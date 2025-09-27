# reproduce_figure_342.py
# Reproduce Burt & Adelson Figure 3.42 for Laplacian pyramid blending
# - Custom separable Gaussian (reflect)
# - Laplacian pyramid (decimate/expand)
# - Sigmoid mask
# - Grid (a–l) as in the book figure

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_float32

plt.rcParams['image.cmap'] = 'gray'

# -----------------------------
# sRGB <-> Linear conversions
# -----------------------------
def srgb_to_linear(img):
    a = 0.055
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    lo = img <= 0.04045
    out = np.empty_like(img, dtype=np.float32)
    out[lo] = img[lo] / 12.92
    out[~lo] = ((img[~lo] + a) / (1 + a)) ** 2.4
    return out

def linear_to_srgb(img):
    a = 0.055
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    lo = img <= 0.0031308
    out = np.empty_like(img, dtype=np.float32)
    out[lo] = 12.92 * img[lo]
    out[~lo] = (1 + a) * (img[~lo] ** (1/2.4)) - a
    return np.clip(out, 0.0, 1.0)

# -----------------------------
# Custom Gaussian (separable)
# -----------------------------
def gaussian_kernel1d(ksize=7, sigma=1.5):
    assert ksize % 2 == 1, "ksize must be odd"
    ax = np.arange(-(ksize//2), ksize//2 + 1, dtype=np.float32)
    k = np.exp(-(ax**2) / (2 * sigma * sigma)).astype(np.float32)
    k /= k.sum()
    return k

def _conv1d_axis0_reflect(img2d, k1d):
    H, W = img2d.shape
    k, p = k1d.size, k1d.size // 2
    padded = np.pad(img2d, ((p, p), (0, 0)), mode="reflect")
    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=(H, k, W), strides=(s0, s0, s1), writeable=False
    )
    return np.tensordot(windows, k1d, axes=([1], [0])).astype(np.float32)

def _conv1d_axis1_reflect(img2d, k1d):
    H, W = img2d.shape
    k, p = k1d.size, k1d.size // 2
    padded = np.pad(img2d, ((0, 0), (p, p)), mode="reflect")
    s0, s1 = padded.strides
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=(H, W, k), strides=(s0, s1, s1), writeable=False
    )
    return np.tensordot(windows, k1d, axes=([2], [0])).astype(np.float32)

def gaussian_filter_custom(img, sigma):
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    k1d = gaussian_kernel1d(ksize, sigma)
    if img.ndim == 2:
        return _conv1d_axis0_reflect(_conv1d_axis1_reflect(img, k1d), k1d)
    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        out[..., c] = _conv1d_axis0_reflect(_conv1d_axis1_reflect(img[..., c], k1d), k1d)
    return out

# -----------------------------
# I/O helpers
# -----------------------------
def read_rgb01(path):
    im = imread(path)
    if im.ndim == 2:
        im = np.stack([im]*3, axis=-1)
    im = im[..., :3]
    return img_as_float32(im)

def ensure_size(img, hw):
    H, W = hw
    if img.shape[:2] == (H, W):
        return img.astype(np.float32)
    return resize(img, (H, W), order=1, mode="reflect",
                  anti_aliasing=True, preserve_range=True).astype(np.float32)

def clip01(x): return np.clip(x, 0, 1).astype(np.float32)

# -----------------------------
# Pyramids
# -----------------------------
def decimate2(img, sigma=1.0):
    return gaussian_filter_custom(img, sigma)[::2, ::2, ...].astype(np.float32)

def expand2(img, out_hw, sigma=1.0):
    H, W = out_hw
    up = np.zeros((H, W) + (() if img.ndim == 2 else (img.shape[2],)), dtype=np.float32)
    up[::2, ::2, ...] = img
    return (4.0 * gaussian_filter_custom(up, sigma)).astype(np.float32)

def gaussian_pyramid(img, levels=6, sigma=1.0):
    G = [img.astype(np.float32)]
    for _ in range(1, levels):
        G.append(decimate2(G[-1], sigma))
    return G

def laplacian_pyramid_from_gaussian(G, sigma=1.0):
    L = []
    for i in range(len(G) - 1):
        up = expand2(G[i + 1], G[i].shape[:2], sigma)
        L.append((G[i] - up).astype(np.float32))
    L.append(G[-1].astype(np.float32))
    return L

def reconstruct_from_laplacian_pyr(L, sigma=1.0):
    img = L[-1].astype(np.float32)
    for i in range(len(L) - 2, -1, -1):
        img = expand2(img, L[i].shape[:2], sigma) + L[i]
    return img.astype(np.float32)

# -----------------------------
# Smooth (sigmoid) mask
# -----------------------------
def sigmoid_mask(shape_hw, center=0.45, width=0.10, orientation='vertical', invert=False):
    H, W = shape_hw
    if orientation == 'vertical':
        x = np.linspace(0, 1, W, dtype=np.float32)
        t = (x - center) / (width + 1e-8)
        m1d = 1.0 / (1.0 + np.exp(-t))
        m = np.tile(m1d[None, :], (H, 1))
    else:
        y = np.linspace(0, 1, H, dtype=np.float32)
        t = (y - center) / (width + 1e-8)
        m1d = 1.0 / (1.0 + np.exp(-t))
        m = np.tile(m1d[:, None], (1, W))
    if invert: m = 1.0 - m
    return clip01(m)

# -----------------------------
# Figure 3.42 reproduction
# -----------------------------
if __name__ == "__main__":
    # ---- Inputs ----
    pathA = "./apple.jpeg"
    pathB = "./orange.jpeg"

    # Pyramid params close to the paper
    levels = 6
    pyr_sigma = 1.0

    # Mask (smooth interpolation function)
    mask_center = 0.45
    mask_width  = 0.10
    invert_mask = False

    # Read & match size (sRGB)
    A_srgb = read_rgb01(pathA)
    B_srgb = ensure_size(read_rgb01(pathB), A_srgb.shape[:2])

    # Work in linear light for correct blending
    A = srgb_to_linear(A_srgb)
    B = srgb_to_linear(B_srgb)

    # Smooth mask (1 -> take A, 0 -> take B)
    M = sigmoid_mask(A.shape[:2], center=mask_center, width=mask_width,
                     orientation='vertical', invert=invert_mask)

    # Build pyramids
    GA = gaussian_pyramid(A, levels, pyr_sigma)
    GB = gaussian_pyramid(B, levels, pyr_sigma)
    GM = gaussian_pyramid(M[..., None], levels, pyr_sigma)
    LA = laplacian_pyramid_from_gaussian(GA, pyr_sigma)
    LB = laplacian_pyramid_from_gaussian(GB, pyr_sigma)

    # Level selection: high, mid, low -> 0, 2, 4 (as in Fig. 3.42 caption)
    pick_levels = [0, 2, 4]

    # Per-level contributions
    left_contrib  = []  # A * M
    mid_contrib   = []  # B * (1 - M)
    right_contrib = []  # sum
    for li in pick_levels:
        m = GM[li]                    # [H,W,1]
        a_part = LA[li] * m
        b_part = LB[li] * (1.0 - m)
        left_contrib.append(a_part)
        mid_contrib.append(b_part)
        right_contrib.append(a_part + b_part)

    # Final blend (sum over all levels)
    Lblend = [GM[i] * LA[i] + (1.0 - GM[i]) * LB[i] for i in range(levels)]
    blended_lin = reconstruct_from_laplacian_pyr(Lblend, pyr_sigma)

    # Convert to sRGB for display
    A_disp = linear_to_srgb(A)
    B_disp = linear_to_srgb(B)
    blended = linear_to_srgb(blended_lin)

    # ---------------- Plot a–l ----------------
    # Grid layout: 4 rows x 3 cols
    # Rows 1–3: (a–i) = high/mid/low frequency bands:
    #   col1: A*M, col2: B*(1-M), col3: sum
    # Row 4: (j–l) = A, B, final blend
    labels = [['(a)', '(b)', '(c)'],
              ['(d)', '(e)', '(f)'],
              ['(g)', '(h)', '(i)'],
              ['(j)', '(k)', '(l)']]

    fig = plt.figure(figsize=(12, 14))

    # Helper to show image (auto squeeze)
    def _show(ax, im, title):
        if im.ndim == 3 and im.shape[2] == 1: im = im[..., 0]
        ax.imshow(clip01(im)); ax.set_title(title); ax.axis('off')

    # Row 1–3
    for r, li in enumerate(pick_levels):
        ax1 = plt.subplot(4, 3, r*3 + 1)
        ax2 = plt.subplot(4, 3, r*3 + 2)
        ax3 = plt.subplot(4, 3, r*3 + 3)
        # 为了可视化 Laplacian 贡献，加 0.5 偏置（像书里那样）
        _show(ax1, clip01(left_contrib[r] + 0.5),  labels[r][0])
        _show(ax2, clip01(mid_contrib[r]  + 0.5),  labels[r][1])
        _show(ax3, clip01(right_contrib[r] + 0.5), labels[r][2])

    # Row 4: originals and final blend
    _show(plt.subplot(4,3,10), A_disp, labels[3][0])
    _show(plt.subplot(4,3,11), B_disp, labels[3][1])
    _show(plt.subplot(4,3,12), blended, labels[3][2])

    plt.tight_layout()
    plt.savefig("figure_342_repro.png", dpi=200)
    plt.show()

    print("Saved figure as figure_342_repro.png")
