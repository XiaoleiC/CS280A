# make_hybrid_headless.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.io import imread
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize, warp, ProjectiveTransform, AffineTransform
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from scipy.ndimage import gaussian_filter

# ---------- utilities ----------
def to_float01(im):
    return np.clip(img_as_float(im), 0.0, 1.0)

def save_fig(im, title, out_path):
    plt.figure(figsize=(5,5))
    if im.ndim == 2:
        plt.imshow(im, cmap="gray")
    else:
        plt.imshow(np.clip(im, 0, 1))
    plt.title(title); plt.axis("off"); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

def fft_log_mag(im):
    g = rgb2gray(im) if im.ndim == 3 else im
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(F))
    mag /= (mag.max() + 1e-8)
    return mag

def lowpass(im, sigma):
    if im.ndim == 2:
        return gaussian_filter(im, sigma=sigma, mode="reflect")
    out = np.empty_like(im, dtype=float)
    for c in range(im.shape[2]):
        out[..., c] = gaussian_filter(im[..., c], sigma=sigma, mode="reflect")
    return out

def highpass(im, sigma):  # image minus blur
    return im - lowpass(im, sigma)

def hybrid_image(im_low_src, im_high_src, sigma_low, sigma_high, alpha=1.0, beta=0.8):
    low  = lowpass(im_low_src,  sigma_low)
    high = highpass(im_high_src, sigma_high)
    hybrid = alpha*low + beta*high
    return np.clip(hybrid, 0, 1), low, high

# ---------- automatic alignment (headless) ----------
def auto_align(src_to_warp, ref, mode="projective", max_dim=1024):
    """
    Align src_to_warp to ref using ORB + RANSAC.
    mode: "projective" (默认，效果最好) 或 "affine"（更稳）
    """
    # downscale for robustness & speed
    def downscale(im):
        h, w = im.shape[:2]
        s = max(h, w)
        if s <= max_dim: return im, 1.0
        scale = max_dim / s
        im2 = resize(im, (int(h*scale), int(w*scale)), anti_aliasing=True)
        return im2, scale

    ref_small, s_ref  = downscale(ref)
    src_small, s_src  = downscale(src_to_warp)

    g_ref = rgb2gray(ref_small) if ref_small.ndim == 3 else ref_small
    g_src = rgb2gray(src_small) if src_small.ndim == 3 else src_small

    orb = ORB(n_keypoints=5000, fast_threshold=0.05)
    orb.detect_and_extract(g_ref); k_ref, d_ref = orb.keypoints, orb.descriptors
    orb.detect_and_extract(g_src); k_src, d_src = orb.keypoints, orb.descriptors

    matches = match_descriptors(d_src, d_ref, cross_check=True)
    if matches.size < 10:
        raise RuntimeError("Not enough matches for alignment.")

    src_pts = k_src[matches[:,0]][:, ::-1]  # (x,y)
    ref_pts = k_ref[matches[:,1]][:, ::-1]

    # pick model
    Model = ProjectiveTransform if mode=="projective" else AffineTransform
    model_robust, inliers = ransac(
        (src_pts, ref_pts), Model, min_samples=4 if mode=="projective" else 3,
        residual_threshold=2.0, max_trials=5000
    )
    if not model_robust:
        raise RuntimeError("RANSAC failed to find a transform.")

    # upscale transform back to original resolution
    scale_back = (1/s_src, 1/s_ref)
    # T = S_ref^-1 * H * S_src
    S_src  = np.array([[1/s_src,0,0],[0,1/s_src,0],[0,0,1]])
    S_refi = np.array([[s_ref,0,0],[0,s_ref,0],[0,0,1]])
    H = model_robust.params
    H_full = S_refi @ H @ S_src

    # warp
    out = warp(src_to_warp, inverse_map=np.linalg.inv(H_full),
               output_shape=ref.shape[:2], mode="edge")
    return out

# ---------- pyramids for report (optional) ----------
from skimage.transform import rescale
def gaussian_pyr(im, N=5, sigma=1.0):
    pyr=[im]
    cur=im
    for _ in range(1,N):
        cur = lowpass(cur, sigma)
        cur = rescale(cur, 0.5, channel_axis=-1 if cur.ndim==3 else None, anti_aliasing=True)
        pyr.append(cur)
    return pyr

def laplacian_pyr(im, N=5, sigma=1.0):
    gp = gaussian_pyr(im,N=N,sigma=sigma)
    lp=[]
    for i in range(len(gp)-1):
        up = resize(gp[i+1], gp[i].shape[:2], anti_aliasing=True)
        lp.append(np.clip(gp[i]-up, -1, 1))
    lp.append(gp[-1])
    return gp, lp

# ---------- main ----------
def main(
    high_path="./DerekPicture.jpg",   # 提供高频
    low_path="./nutmeg.jpg",          # 提供低频
    sigma_low=6.0, sigma_high=3.0, alpha=1.0, beta=0.8,
    align_mode="affine", N_pyr=5
):
    out_dir = Path("hybrid_outputs"); out_dir.mkdir(exist_ok=True, parents=True)

    imH = to_float01(imread(high_path))
    imL = to_float01(imread(low_path))

    # 对齐（无 GUI）
    imH2 = auto_align(imH, imL, mode=align_mode)

    # 统一大小（以低频图为准）
    if imH2.shape[:2] != imL.shape[:2]:
        imH2 = resize(imH2, imL.shape[:2], anti_aliasing=True)

    hybrid, low_img, high_img = hybrid_image(imL, imH2, sigma_low, sigma_high, alpha, beta)

    # 保存结果
    save_fig(imL, "Low source (aligned ref)", out_dir/"aligned_low.png")
    save_fig(imH2, "High source (aligned)", out_dir/"aligned_high.png")
    save_fig(low_img,  f"Low-pass σ={sigma_low}", out_dir/"lowpass.png")
    hp_vis = (high_img - high_img.min())/(high_img.max()-high_img.min()+1e-8)
    save_fig(hp_vis,  f"High-pass σ={sigma_high}", out_dir/"highpass_vis.png")
    save_fig(hybrid,  "Hybrid", out_dir/"hybrid.png")

    # FFT
    save_fig(fft_log_mag(imL),     "FFT low src", out_dir/"fft_low.png")
    save_fig(fft_log_mag(imH2),    "FFT high src", out_dir/"fft_high.png")
    save_fig(fft_log_mag(low_img),"FFT low-pass", out_dir/"fft_lowpass.png")
    save_fig(fft_log_mag(high_img),"FFT high-pass", out_dir/"fft_highpass.png")
    save_fig(fft_log_mag(hybrid),  "FFT hybrid", out_dir/"fft_hybrid.png")

    # 金字塔
    gp, lp = gaussian_pyr(hybrid, N=N_pyr), laplacian_pyr(hybrid, N=N_pyr)[1]
    for i, lvl in enumerate(gp):
        save_fig(lvl, f"Gaussian Pyr L{i}", out_dir/f"gp_L{i}.png")
    for i, lvl in enumerate(lp):
        disp = (lvl - lvl.min())/(lvl.max()-lvl.min()+1e-8)
        save_fig(disp, f"Laplacian Pyr L{i}", out_dir/f"lp_L{i}.png")

    print(f"Done. Results saved to {out_dir.resolve()}")

if __name__ == "__main__":
    # 修改路径为你的两张图片
    main()
