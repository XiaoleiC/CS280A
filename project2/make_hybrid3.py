import os
import numpy as np
import matplotlib.pyplot as plt
# plt.rcParams['image.cmap'] = 'gray'

from skimage import transform as sktf
from skimage import img_as_float32


def gaussian_kernel1d(ksize=7, sigma=1.5):
    assert ksize % 2 == 1, "ksize should be odd"
    ax = np.arange(-(ksize//2), ksize//2 + 1, dtype=np.float32)
    k = np.exp(-(ax**2) / (2 * sigma**2)).astype(np.float32)
    k /= np.sum(k)
    return k

def _conv1d_axis0_reflect(img2d, k1d):
    H, W = img2d.shape
    k = k1d.size
    p = k // 2
    padded = np.pad(img2d, ((p, p), (0, 0)), mode='reflect')
    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(H, k, W),
        strides=(s0, s0, s1),
        writeable=False
    )
    out = np.tensordot(windows, k1d, axes=([1], [0]))
    return out.astype(np.float32)

def _conv1d_axis1_reflect(img2d, k1d):
    H, W = img2d.shape
    k = k1d.size
    p = k // 2
    padded = np.pad(img2d, ((0, 0), (p, p)), mode='reflect')
    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded,
        shape=(H, W, k),
        strides=(s0, s1, s1),
        writeable=False
    )
    out = np.tensordot(windows, k1d, axes=([2], [0]))
    return out.astype(np.float32)

def gaussian_filter_custom(img, sigma):
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    k1d = gaussian_kernel1d(ksize, sigma)

    if img.ndim == 2:
        tmp = _conv1d_axis1_reflect(img, k1d)
        out = _conv1d_axis0_reflect(tmp, k1d)
        return out
    elif img.ndim == 3:
        C = img.shape[2]
        out = np.empty_like(img, dtype=np.float32)
        for c in range(C):
            tmp = _conv1d_axis1_reflect(img[..., c], k1d)
            out[..., c] = _conv1d_axis0_reflect(tmp, k1d)
        return out
    else:
        raise ValueError("Unsupported image dimension")

def read_rgb01(path):
    img = plt.imread(path)[..., :3]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img_as_float32(img)
    return img

def to_gray(img):
    if img.ndim == 2:
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

def show_image(img, title="", cmap=None):
    if img.ndim == 2 and cmap is None:
        cmap = 'gray'
    plt.imshow(img, cmap=cmap)
    plt.title(title); plt.axis("off")

def fft_log_mag(gray_or_rgb):
    g = gray_or_rgb if gray_or_rgb.ndim == 2 else np.mean(gray_or_rgb, axis=-1)
    F = np.fft.fftshift(np.fft.fft2(g))
    M = np.log1p(np.abs(F))
    return M / (M.max() + 1e-8)

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def pick_two_points(img, title):
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(title + "\nLeft-click twice, then middle/right-click to confirm.")
    pts = plt.ginput(2, timeout=0)
    plt.close()
    if len(pts) != 2:
        raise RuntimeError("Need exactly 2 points.")
    return np.array(pts, dtype=np.float32)

def align_by_two_points(src_img, dst_img):
    print("Pick 2 points on the SOURCE (high-frequency) image...")
    src_pts = pick_two_points(src_img, "SOURCE image: pick 2 points (e.g., eye centers)")
    print("Pick 2 corresponding points on the TARGET (low-frequency) image...")
    dst_pts = pick_two_points(dst_img, "TARGET image: pick 2 corresponding points")

    tform = sktf.estimate_transform('similarity', src_pts, dst_pts)

    warped_src = sktf.warp(
        src_img, inverse_map=tform.inverse, output_shape=dst_img.shape[:2],
        mode="reflect", preserve_range=True
    ).astype(np.float32)

    src_mask = np.ones(src_img.shape[:2], dtype=np.float32)
    warped_mask = sktf.warp(
        src_mask, inverse_map=tform.inverse, output_shape=dst_img.shape[:2],
        mode="constant", cval=0.0, preserve_range=True
    ).astype(np.float32)
    warped_mask = np.clip(warped_mask, 0.0, 1.0)

    return warped_src, warped_mask

def auto_crop_to_overlap(img_a, mask_a, img_b, mask_b, thresh=0.98):
    assert img_a.shape == img_b.shape
    both = (mask_a > thresh) & (mask_b > thresh)
    rows = np.where(both.any(axis=1))[0]
    cols = np.where(both.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img_a, img_b, both.astype(np.float32)
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    return img_a[r0:r1, c0:c1, ...], img_b[r0:r1, c0:c1, ...], both[r0:r1, c0:c1].astype(np.float32)

def low_pass(img, sigma):
    return gaussian_filter_custom(img, sigma).astype(np.float32)

def high_pass(img, sigma):
    return (img - low_pass(img, sigma)).astype(np.float32)

def smooth_mask(mask, sigma=7):
    m = gaussian_filter_custom(mask, sigma)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m

def apply_taper(hp, mask, sigma=7):
    m = smooth_mask(mask, sigma=sigma)
    if hp.ndim == 3:
        m = m[..., None]
    return hp * m

def combine_balanced(lp, hp, alpha_lo=1.0, alpha_hi=1.0):
    scale = rms(lp) / (rms(hp) + 1e-8)
    return np.clip(alpha_lo * lp + alpha_hi * scale * hp, 0.0, 1.0)

def make_hybrid(img_hi_src, img_lo, sigma_hi=4, sigma_lo=10, alpha_hi=1.0, alpha_lo=1.0,
                crop_overlap=True, feather_sigma=7):
    hi_warp, hi_mask = align_by_two_points(img_hi_src, img_lo)

    lp = low_pass(img_lo, sigma_lo)
    hp_raw = high_pass(hi_warp, sigma_hi)

    hp = apply_taper(hp_raw, hi_mask, sigma=feather_sigma)

    if crop_overlap:
        mask_lo = np.ones(img_lo.shape[:2], dtype=np.float32)
        lp, hp, _ = auto_crop_to_overlap(lp, mask_lo, hp, hi_mask)

    hybrid = combine_balanced(lp, hp, alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    return hybrid, hp, lp

def pyramids(img, N=5, show_plots=True):
    g = [img.astype(np.float32)]
    for _ in range(1, N):
        gb = low_pass(g[-1], sigma=1.0)
        g.append(gb[::2, ::2, ...])

    l = []
    for i in range(N - 1):
        up = np.zeros_like(g[i])
        up[::2, ::2, ...] = g[i + 1]
        up = low_pass(up, sigma=1.0)
        l.append(g[i] - up)
    l.append(g[-1])

    if show_plots:
        plt.figure(figsize=(12, 2.2 * N))
        for i in range(N):
            plt.subplot(N, 2, 2*i+1); show_image(np.clip(g[i], 0, 1), f'Gaussian {i}')
            plt.subplot(N, 2, 2*i+2); show_image(np.clip(l[i] + 0.5, 0, 1), f'Laplacian {i}')
        plt.tight_layout(); plt.show()
    return g, l

if __name__ == "__main__":
    img_lo_path = "./selfie.jpg"
    img_hi_path = "./nutmeg.jpg"

    img_hi = read_rgb01(img_hi_path)
    img_lo = read_rgb01(img_lo_path)
    # img_hi = to_gray(read_rgb01(img_hi_path))
    # img_lo = to_gray(read_rgb01(img_lo_path))

    sigma_high = 2
    sigma_low  = 6
    alpha_hi   = 0.5
    alpha_lo   = 0.5

    hybrid, hp_vis, lp_vis = make_hybrid(
        img_hi, img_lo, sigma_hi=sigma_high, sigma_lo=sigma_low,
        alpha_hi=alpha_hi, alpha_lo=alpha_lo,
        crop_overlap=True, feather_sigma=7
    )

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1); show_image(img_hi, "Source (high-frequency image)")
    plt.subplot(2, 3, 2); show_image(img_lo, "Target (low-frequency image)")
    plt.subplot(2, 3, 3); show_image(np.clip(hp_vis + 0.5, 0, 1), rf"High-pass ($\sigma$={sigma_high})")
    plt.subplot(2, 3, 4); show_image(lp_vis, rf"Low-pass ($\sigma$={sigma_low})")
    plt.subplot(2, 3, 5); show_image(hybrid, "Hybrid")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1); show_image(fft_log_mag(hp_vis), "High-pass FFT")
    plt.subplot(1, 3, 2); show_image(fft_log_mag(lp_vis), "Low-pass FFT")
    plt.subplot(1, 3, 3); show_image(fft_log_mag(hybrid), "Hybrid FFT")
    # plt.subplot(1, 3, 1); show_image(fft_log_mag(hp_vis), "High-pass FFT", cmap = "grey")
    # plt.subplot(1, 3, 2); show_image(fft_log_mag(lp_vis), "Low-pass FFT", cmap = "grey")
    # plt.subplot(1, 3, 3); show_image(fft_log_mag(hybrid), "Hybrid FFT", cmap = "grey")
    plt.tight_layout(); plt.show()

    pyramids(hybrid, N=5, show_plots=True)
