import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float32
from skimage.io import imread
from skimage.transform import resize

# plt.rcParams['image.cmap'] = 'gray'

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
        padded, shape=(H, k, W), strides=(s0, s0, s1), writeable=False
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
        padded, shape=(H, W, k), strides=(s0, s1, s1), writeable=False
    )
    out = np.tensordot(windows, k1d, axes=([2], [0]))
    return out.astype(np.float32)

def gaussian_filter_custom(img, sigma):
    ksize = int(2 * np.ceil(3 * sigma) + 1)  # cover ~±3σ
    k1d = gaussian_kernel1d(ksize, sigma)
    if img.ndim == 2:
        tmp = _conv1d_axis1_reflect(img, k1d)
        out = _conv1d_axis0_reflect(tmp, k1d)
        return out
    elif img.ndim == 3:
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            tmp = _conv1d_axis1_reflect(img[..., c], k1d)
            out[..., c] = _conv1d_axis0_reflect(tmp, k1d)
        return out
    else:
        raise ValueError("Unsupported image dimension")

def read_rgb01(path):
    im = imread(path)
    if im.ndim == 2:  # gray -> 3ch
        im = np.stack([im]*3, axis=-1)
    im = im[..., :3]
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    else:
        im = img_as_float32(im)
    return im

def ensure_same_size(img, target_shape_hw):
    Ht, Wt = target_shape_hw
    if img.shape[:2] == (Ht, Wt):
        return img
    out = resize(img, (Ht, Wt), order=1, mode='reflect', anti_aliasing=True, preserve_range=True)
    return out.astype(np.float32)

def to_gray(img):
    if img.ndim == 2:
        return img
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.2989*r + 0.5870*g + 0.1140*b).astype(np.float32)

def clip01(x):
    return np.clip(x, 0.0, 1.0).astype(np.float32)

def gaussian_stack(img, levels=6, sigma=2.0):
    G = [img.astype(np.float32)]
    for _ in range(1, levels):
        G.append(gaussian_filter_custom(G[-1], sigma))
    return G

def laplacian_stack_from_gaussian(G):
    L = []
    for i in range(len(G)-1):
        L.append((G[i] - G[i+1]).astype(np.float32))
    L.append(G[-1].astype(np.float32))
    return L

def reconstruct_from_laplacian(L):
    acc = np.zeros_like(L[0], dtype=np.float32)
    for i in range(len(L)-1):
        acc = acc + L[i]
    acc = acc + L[-1]
    return acc.astype(np.float32)

def make_mask(shape_hw, mode='vertical', invert=False, radius_ratio=0.45):
    H, W = shape_hw
    if mode == 'vertical':
        x = np.linspace(0, 1, W, dtype=np.float32)
        m = np.tile(x[None, :], (H, 1))
    elif mode == 'horizontal':
        y = np.linspace(0, 1, H, dtype=np.float32)
        m = np.tile(y[:, None], (1, W))
    elif mode == 'circle':
        cy, cx = H/2.0, W/2.0
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
        r = r / (min(H, W)*radius_ratio)
        m = 1.0 - clip01(r)  # 中心1，外部0
    else:
        raise ValueError("mode must be 'vertical'/'horizontal'/'circle'")
    m = clip01(m)
    if invert:
        m = 1.0 - m
    return m

def load_mask(mask_path, target_hw):
    m = imread(mask_path)
    if m.ndim == 3:
        m = to_gray(m)
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    m = ensure_same_size(m, target_hw)
    m = clip01(m)
    return m

def multires_blend(imgA, imgB, mask01, levels=6, sigma=2.0):
    if imgA.shape != imgB.shape:
        raise ValueError("imgA and imgB must have the same shape")
    if mask01.shape != imgA.shape[:2]:
        raise ValueError("mask must have shape [H,W] matching images")

    is_color = (imgA.ndim == 3)
    if not is_color:
        imgA = imgA[..., None]
        imgB = imgB[..., None]

    GA = gaussian_stack(imgA, levels, sigma)
    GB = gaussian_stack(imgB, levels, sigma)
    LA = laplacian_stack_from_gaussian(GA)
    LB = laplacian_stack_from_gaussian(GB)

    GM = gaussian_stack(mask01[..., None], levels, sigma)

    Lblend = []
    for i in range(levels):
        m = GM[i]
        Lblend.append(m * LA[i] + (1.0 - m) * LB[i])

    blended = reconstruct_from_laplacian(Lblend)

    blended = clip01(blended)
    if not is_color:
        blended = blended[..., 0]
    return blended

def show_stack(stack, title_prefix):
    L = len(stack)
    cols = 5
    rows = int(np.ceil(L / cols))
    plt.figure(figsize=(3.5*cols, 3.5*rows))
    for i, img in enumerate(stack):
        plt.subplot(rows, cols, i+1)
        im = img
        if im.ndim == 3 and im.shape[2] == 1:
            im = im[..., 0]
        plt.imshow(clip01(im))
        plt.title(f"{title_prefix} {i}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    imgA_path = "./apple.jpeg"
    imgB_path = "./orange.jpeg"

    mask_mode  = "circle"     # "vertical" | "horizontal" | "circle" | "file"
    mask_path  = "./mask.png"
    invert_mask = False

    levels = 6
    sigma  = 2.0

    A = read_rgb01(imgA_path)
    B = read_rgb01(imgB_path)
    B = ensure_same_size(B, A.shape[:2])

    if mask_mode == "file":
        M = load_mask(mask_path, A.shape[:2])
    else:
        M = make_mask(A.shape[:2], mode=mask_mode, invert=invert_mask)

    GA = gaussian_stack(A, levels, sigma)
    GB = gaussian_stack(B, levels, sigma)
    LA = laplacian_stack_from_gaussian(GA)
    LB = laplacian_stack_from_gaussian(GB)
    GM = gaussian_stack(M[..., None], levels, sigma)

    show_stack([g for g in GA], "Gaussian A")
    show_stack([g for g in GB], "Gaussian B")
    show_stack([l + 0.5 for l in LA], "Laplacian A (+0.5)")
    show_stack([l + 0.5 for l in LB], "Laplacian B (+0.5)")
    show_stack([gm[...,0] for gm in GM], "Gaussian Mask")

    oraple = multires_blend(A, B, M, levels=levels, sigma=sigma)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(clip01(A)); plt.title("Image A"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(clip01(B)); plt.title("Image B"); plt.axis('off')
    plt.subplot(1,3,3); 
    if M.ndim == 2: plt.imshow(M, cmap='gray')
    else:           plt.imshow(M[...,0], cmap='gray')
    plt.title("Mask (white→A, black→B)"); plt.axis('off')
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,6))
    plt.imshow(oraple); plt.title("Multiresolution Blend (Oraple)")
    plt.axis('off'); plt.tight_layout(); plt.show()

    # from skimage.io import imsave
    # imsave("oraple_result.png", (clip01(oraple)*255).astype(np.uint8))
