# make_hybrid_final.py
# Robust hybrid-image pipeline:
# - Similarity alignment from 2 correspondences (clicks)
# - Reflect-mode warp (no black triangles)
# - High-pass tapering using a smoothed mask (feathering)
# - RMS energy matching between bands
# - Optional auto-crop to common overlap
# - FFT views + Gaussian/Laplacian pyramids

import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'   # <- makes ALL 2-D images render in gray

from scipy.ndimage import gaussian_filter
from skimage import transform as sktf
from skimage import img_as_float32

# -----------------------------
# I/O & small helpers
# -----------------------------
def read_rgb01(path):
    img = plt.imread(path)[..., :3]  # drop alpha if present
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img_as_float32(img)
    return img

def to_gray(img):
    if img.ndim == 2:
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b  # standard luminance

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

# -----------------------------
# Alignment from two points
# -----------------------------
def pick_two_points(img, title):
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')   # <-- ensure gray in the picker window
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

    # Similarity transform src -> dst
    tform = sktf.estimate_transform('similarity', src_pts, dst_pts)

    # Warp image with reflect padding to avoid black triangles
    warped_src = sktf.warp(
        src_img, inverse_map=tform.inverse, output_shape=dst_img.shape[:2],
        mode="reflect", preserve_range=True
    ).astype(np.float32)

    # Build a validity mask in constant mode to know overlap (1=valid, 0=out)
    src_mask = np.ones(src_img.shape[:2], dtype=np.float32)
    warped_mask = sktf.warp(
        src_mask, inverse_map=tform.inverse, output_shape=dst_img.shape[:2],
        mode="constant", cval=0.0, preserve_range=True
    ).astype(np.float32)
    warped_mask = np.clip(warped_mask, 0.0, 1.0)

    return warped_src, warped_mask

# -----------------------------
# Overlap crop (optional)
# -----------------------------
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

# -----------------------------
# Filters and feathering
# -----------------------------
def low_pass(img, sigma):
    if img.ndim == 3:
        return np.stack([gaussian_filter(img[..., c], sigma=sigma) for c in range(img.shape[2])], axis=-1).astype(np.float32)
    return gaussian_filter(img, sigma=sigma).astype(np.float32)

def high_pass(img, sigma):
    return (img - low_pass(img, sigma)).astype(np.float32)

def smooth_mask(mask, sigma=7):
    m = gaussian_filter(mask, sigma=sigma)
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m

def apply_taper(hp, mask, sigma=7):
    m = smooth_mask(mask, sigma=sigma)
    if hp.ndim == 3:
        m = m[..., None]
    return hp * m

# -----------------------------
# Balanced combine
# -----------------------------
def combine_balanced(lp, hp, alpha_lo=1.0, alpha_hi=1.0):
    # Match RMS energy so contributions are comparable
    scale = rms(lp) / (rms(hp) + 1e-8)
    return np.clip(alpha_lo * lp + alpha_hi * scale * hp, 0.0, 1.0)

# -----------------------------
# Hybrid construction
# -----------------------------
def make_hybrid(img_hi_src, img_lo, sigma_hi=4, sigma_lo=10, alpha_hi=1.0, alpha_lo=1.0,
                crop_overlap=True, feather_sigma=7):
    # Align high-frequency source to low-frequency target
    hi_warp, hi_mask = align_by_two_points(img_hi_src, img_lo)

    # Build bands
    lp = low_pass(img_lo, sigma_lo)
    hp_raw = high_pass(hi_warp, sigma_hi)

    # Feather high-pass near boundaries (avoid seams)
    hp = apply_taper(hp_raw, hi_mask, sigma=feather_sigma)

    # Optionally crop to common overlap to remove any residual edges
    if crop_overlap:
        mask_lo = np.ones(img_lo.shape[:2], dtype=np.float32)
        lp, hp, _ = auto_crop_to_overlap(lp, mask_lo, hp, hi_mask)

    # Energy-balanced sum
    hybrid = combine_balanced(lp, hp, alpha_lo=alpha_lo, alpha_hi=alpha_hi)
    return hybrid, hp, lp

# -----------------------------
# Pyramids (viz)
# -----------------------------
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

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Paths
    img_lo_path = "./DerekPicture.jpg"  # high-frequency provider (close view)
    img_hi_path = "./nutmeg.jpg"        # low-frequency provider (far view)

    # Load
    # img_hi = read_rgb01(img_hi_path)
    # img_lo = read_rgb01(img_lo_path)
    img_hi = to_gray(read_rgb01(img_hi_path))
    img_lo = to_gray(read_rgb01(img_lo_path))

    # Tunable parameters
    sigma_high = 2    # 2–6 recommended
    sigma_low  = 6   # 8–12 recommended
    alpha_hi   = 0.6  # try 1.2 if high-pass looks weak
    alpha_lo   = 0.4

    # Build hybrid
    hybrid, hp_vis, lp_vis = make_hybrid(
        img_hi, img_lo, sigma_hi=sigma_high, sigma_lo=sigma_low,
        alpha_hi=alpha_hi, alpha_lo=alpha_lo,
        crop_overlap=True, feather_sigma=7
    )

    # Spatial results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1); show_image(img_hi, "Source (high-frequency image)")
    plt.subplot(2, 3, 2); show_image(img_lo, "Target (low-frequency image)")
    plt.subplot(2, 3, 3); show_image(np.clip(hp_vis + 0.5, 0, 1), rf"High-pass ($\sigma$={sigma_high})")
    plt.subplot(2, 3, 4); show_image(lp_vis, rf"Low-pass ($\sigma$={sigma_low})")
    plt.subplot(2, 3, 5); show_image(hybrid, "Hybrid")
    plt.tight_layout(); plt.show()

    # Frequency-domain view
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1); show_image(fft_log_mag(hp_vis), "High-pass FFT", cmap="gray")
    plt.subplot(1, 3, 2); show_image(fft_log_mag(lp_vis), "Low-pass FFT", cmap="gray")
    plt.subplot(1, 3, 3); show_image(fft_log_mag(hybrid), "Hybrid FFT", cmap="gray")
    plt.tight_layout(); plt.show()

    # Pyramids (optional)
    pyramids(hybrid, N=5, show_plots=True)
