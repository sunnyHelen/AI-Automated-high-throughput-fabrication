#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, math, json
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

try:
    import torch
    _HAS_TORCH = True
    try:
        import lpips as _lpips
        _HAS_LPIPS = True
        _lpips_model = _lpips.LPIPS(net='vgg').eval()
    except Exception:
        _HAS_LPIPS = False
        _lpips_model = None
except Exception:
    _HAS_TORCH = False
    _HAS_LPIPS = False
    _lpips_model = None

_HAS_TORCHVISION = False
try:
    if _HAS_TORCH:
        import torchvision
        from torchvision.models.video import r2plus1d_18
        _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

_HAS_TIMM = False
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

_HAS_OPENCLIP = False
_HAS_CLIP = False
try:
    import open_clip
    _HAS_OPENCLIP = True
except Exception:
    _HAS_OPENCLIP = False
if not _HAS_OPENCLIP:
    try:
        import clip as _clip
        _HAS_CLIP = True
    except Exception:
        _HAS_CLIP = False

def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def load_video(path: str) -> np.ndarray:
    if os.path.isdir(path):
        exts = ('*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp')
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
        if not files:
            raise FileNotFoundError(f"No image frames found in directory: {path}")
        files = sorted(files, key=natural_key)
        frames = []
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {f}")
            frames.append(img)
        return np.stack(frames, axis=0)
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames read from video: {path}")
        return np.stack(frames, axis=0)

def resize_video(v: np.ndarray, size_hw: Tuple[int,int] | None) -> np.ndarray:
    if size_hw is None: return v
    H, W = size_hw
    out = [cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA) for fr in v]
    return np.stack(out, axis=0)

def denoise_frame(frame: np.ndarray, mode: str = "median") -> np.ndarray:
    if mode == "median":
        return cv2.medianBlur(frame, 3)
    elif mode == "gaussian":
        return cv2.GaussianBlur(frame, (3,3), 0)
    elif mode == "nlm":
        return cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
    else:
        return frame

def denoise_video(v: np.ndarray, mode: str = "median") -> np.ndarray:
    return np.stack([denoise_frame(fr, mode) for fr in v], axis=0)

def stabilize_video(v: np.ndarray, ref_channel: str = "G",
                    number_of_iterations: int = 50, termination_eps: float = 1e-4) -> np.ndarray:
    if v.shape[0] < 2:
        return v
    ch_idx = {"B":0, "G":1, "R":2}.get(ref_channel.upper(), 1)
    ref = cv2.GaussianBlur(v[0][:,:,ch_idx], (5,5), 0)
    sz = (v.shape[2], v.shape[1])
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    out = [v[0]]
    for t in range(1, v.shape[0]):
        im = cv2.GaussianBlur(v[t][:,:,ch_idx], (5,5), 0)
        try:
            _, w = cv2.findTransformECC(ref, im, warp_matrix, cv2.MOTION_AFFINE, criteria, None, 5)
            aligned = cv2.warpAffine(v[t], w, sz, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            warp_matrix = w.astype(np.float32)
        except cv2.error:
            aligned = v[t]
        out.append(aligned)
    return np.stack(out, axis=0)

def red_mask(frame_bgr: np.ndarray, use_hsv: bool = True, red_th: float = 0.6) -> np.ndarray:
    if use_hsv:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower1 = cv2.inRange(hsv, (0, 80, int(255*red_th)), (10, 255, 255))
        lower2 = cv2.inRange(hsv, (170, 80, int(255*red_th)), (180, 255, 255))
        mask = cv2.bitwise_or(lower1, lower2)
        mask = (mask > 0).astype(np.uint8)
    else:
        b, g, r = cv2.split(frame_bgr)
        r = r.astype(np.float32) / 255.0
        g = g.astype(np.float32) / 255.0
        b = b.astype(np.float32) / 255.0
        mask = ((r > g) & (r > b) & (r > red_th)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask

def masks_from_video(v: np.ndarray, use_hsv: bool = True, red_th: float = 0.6) -> np.ndarray:
    return np.stack([red_mask(fr, use_hsv=use_hsv, red_th=red_th) for fr in v], axis=0)

def red_coverage_curve(masks: np.ndarray) -> np.ndarray:
    T = masks.shape[0]
    areas = masks.reshape(T, -1).sum(axis=1).astype(np.float64)
    total = masks.shape[1]*masks.shape[2]
    return areas / max(total, 1)

def dtw_distance(a: np.ndarray, b: np.ndarray, cost_fn=None) -> tuple[float, List[tuple[int,int]]]:
    a = np.asarray(a); b = np.asarray(b)
    n, m = len(a), len(b)
    D = np.full((n+1, m+1), np.inf, dtype=np.float64)
    D[0,0] = 0.0
    P = np.zeros((n+1, m+1, 2), dtype=np.int32)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if cost_fn is None:
                cost = float((a[i-1]-b[j-1])**2)
            else:
                cost = float(cost_fn(i-1, j-1))
            idx = np.argmin((D[i-1,j], D[i,j-1], D[i-1,j-1]))
            if idx == 0:
                D[i,j] = cost + D[i-1,j]; P[i,j] = (i-1, j)
            elif idx == 1:
                D[i,j] = cost + D[i,j-1]; P[i,j] = (i, j-1)
            else:
                D[i,j] = cost + D[i-1,j-1]; P[i,j] = (i-1, j-1)
    i, j = n, m
    path = []
    while i>0 and j>0:
        path.append((i-1, j-1))
        i, j = P[i,j]
    path.reverse()
    return float(D[n,m]), path

def _interp_to_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n: return x.copy()
    if len(x) < 2 or n < 2:
        return np.full(n, float(x.mean() if len(x) else 0.0), dtype=np.float64)
    xp = np.linspace(0, 1, len(x)); xq = np.linspace(0, 1, n)
    return np.interp(xq, xp, x).astype(np.float64)

def warp_by_path(seq: np.ndarray, path: List[tuple[int,int]], as_ref: str = "a") -> np.ndarray:
    a_idx = [i for (i,_) in path]
    b_idx = [j for (_,j) in path]
    if as_ref == "a":
        out = []
        for i in range(a_idx[0], a_idx[-1]+1):
            js = [b_idx[k] for k in range(len(path)) if a_idx[k]==i]
            out.append(np.mean([seq[j] for j in js]))
        return np.array(out, dtype=np.float64)
    else:
        out = []
        for j in range(b_idx[0], b_idx[-1]+1):
            is_ = [a_idx[k] for k in range(len(path)) if b_idx[k]==j]
            out.append(np.mean([seq[i] for i in is_]))
        return np.array(out, dtype=np.float64)

def occupancy_heatmap(masks: np.ndarray, blur: int = 5) -> np.ndarray:
    acc = masks.sum(axis=0).astype(np.float32)
    if blur and blur > 1:
        acc = cv2.GaussianBlur(acc, (blur if blur%2==1 else blur+1,)*2, 0)
    acc = acc / (acc.max()+1e-8)
    return acc

def _match_size_B_to_A(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    if b.shape == a.shape: return b
    H, W = a.shape[:2]
    return cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape: b = _match_size_B_to_A(b, a)
    va = a.ravel().astype(np.float64); vb = b.ravel().astype(np.float64)
    na = np.linalg.norm(va) + 1e-12; nb = np.linalg.norm(vb) + 1e-12
    return float(np.dot(va, vb) / (na*nb))

def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape: b = _match_size_B_to_A(b, a)
    pa = a.ravel().astype(np.float64); pb = b.ravel().astype(np.float64)
    pa = pa / (pa.sum() + 1e-12); pb = pb / (pb.sum() + 1e-12)
    ca = np.cumsum(pa); cb = np.cumsum(pb)
    return float(np.mean(np.abs(ca - cb)))

def ssim_safe(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape: b = _match_size_B_to_A(b, a)
    a8 = (np.clip(a,0,1)*255).astype(np.uint8); b8 = (np.clip(b,0,1)*255).astype(np.uint8)
    if _HAS_SKIMAGE:
        try:
            return float(ssim(a8, b8, data_range=255))
        except Exception:
            pass
    mse = np.mean((a - b)**2)
    return float(1.0 / (1.0 + mse))

def occupancy_cube(masks: np.ndarray, grid: Tuple[int,int,int]=(16,16,16)) -> np.ndarray:
    T, H, W = masks.shape
    Tg, Hg, Wg = grid
    t_bins = np.linspace(0, T, Tg+1, dtype=int)
    h_bins = np.linspace(0, H, Hg+1, dtype=int)
    w_bins = np.linspace(0, W, Wg+1, dtype=int)
    cube = np.zeros((Tg,Hg,Wg), dtype=np.float32)
    for ti in range(Tg):
        t0,t1 = t_bins[ti], t_bins[ti+1]
        if t1<=t0: t1 = min(t0+1, T)
        slice_t = masks[t0:t1]
        for hi in range(Hg):
            h0,h1 = h_bins[hi], h_bins[hi+1]
            if h1<=h0: h1 = min(h0+1, H)
            for wi in range(Wg):
                w0,w1 = w_bins[wi], w_bins[wi+1]
                if w1<=w0: w1 = min(w0+1, W)
                block = slice_t[:, h0:h1, w0:w1]
                cube[ti,hi,wi] = block.mean()
    m = cube.max()
    if m > 0: cube /= m
    return cube

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    mse = np.mean((a-b)**2)
    if mse <= 1e-12: return float('inf')
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))

def to_gray(fr):
    if fr.ndim==2: return fr
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)

def frame_psnr_mean(gt: np.ndarray, gen: np.ndarray) -> float:
    T = min(len(gt), len(gen))
    vals = [psnr(gt[t], gen[t]) for t in range(T)]
    return float(np.mean(vals))

def frame_ssim_mean(gt: np.ndarray, gen: np.ndarray) -> float:
    T = min(len(gt), len(gen))
    vals = []
    for t in range(T):
        g = to_gray(gt[t]); f = to_gray(gen[t])
        if _HAS_SKIMAGE:
            vals.append(ssim(g, f, data_range=255))
        else:
            g_ = g.astype(np.float32)/255.0; f_ = f.astype(np.float32)/255.0
            mse = np.mean((g_-f_)**2); vals.append(1.0/(1.0+mse))
    return float(np.mean(vals))

def msssim(a: np.ndarray, b: np.ndarray, levels: int = 4) -> float:
    def ssim_gray(x, y):
        if _HAS_SKIMAGE:
            return ssim(x, y, data_range=255)
        x_ = x.astype(np.float32)/255.0; y_ = y.astype(np.float32)/255.0
        mse = np.mean((x_-y_)**2); return 1.0/(1.0+mse)
    def pyramid(img, levels):
        imgs = [img]
        for _ in range(levels-1):
            img = cv2.GaussianBlur(img, (5,5), 1.5)
            img = cv2.resize(img, (max(1,img.shape[1]//2), max(1,img.shape[0]//2)), interpolation=cv2.INTER_AREA)
            imgs.append(img)
        return imgs
    T = min(len(a), len(b))
    vals = []
    for t in range(T):
        ga, gb = to_gray(a[t]), to_gray(b[t])
        pa, pb = pyramid(ga, levels), pyramid(gb, levels)
        s_list = [max(0.0, min(1.0, ssim_gray(pa[i], pb[i]))) for i in range(levels)]
        eps = 1e-9; v = 1.0
        for s in s_list: v *= max(eps, s)
        vals.append(v ** (1.0/levels))
    return float(np.mean(vals))

def frame_lpips_mean(gt: np.ndarray, gen: np.ndarray) -> float:
    if not _HAS_LPIPS:
        return float('nan')
    T = min(len(gt), len(gen))
    vals = []
    for t in range(T):
        g = cv2.resize(gt[t], (256,256), interpolation=cv2.INTER_AREA)
        f = cv2.resize(gen[t], (256,256), interpolation=cv2.INTER_AREA)
        g = g[:,:,::-1].astype(np.float32)/255.0*2.0 - 1.0
        f = f[:,:,::-1].astype(np.float32)/255.0*2.0 - 1.0
        gt_t = torch.from_numpy(g).permute(2,0,1).unsqueeze(0)
        ge_t = torch.from_numpy(f).permute(2,0,1).unsqueeze(0)
        with torch.no_grad():
            d = _lpips_model(gt_t, ge_t).item()
        vals.append(d)
    return float(np.mean(vals))

def optical_flow_farneback_sequence(v: np.ndarray, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2) -> np.ndarray:
    flows = []
    for t in range(len(v)-1):
        prev = to_gray(v[t]); nxt  = to_gray(v[t+1])
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags=0)
        flows.append(flow.astype(np.float32))
    return np.stack(flows, axis=0) if flows else np.zeros((0, v.shape[1], v.shape[2], 2), dtype=np.float32)

def flow_metrics(gt_flow: np.ndarray, gen_flow: np.ndarray) -> dict:
    Tg = gt_flow.shape[0]; Te = gen_flow.shape[0]
    if Tg==0 or Te==0:
        return {"epe_mean": float('nan'), "vec_cosine": float('nan'), "mag_corr": float('nan')}
    idxs = np.linspace(0, Te-1, Tg).astype(int)
    gen_aligned = gen_flow[idxs]
    diff = gt_flow - gen_aligned
    epe = np.sqrt(np.sum(diff**2, axis=-1)).mean()
    a = gt_flow.reshape(-1,2); b = gen_aligned.reshape(-1,2)
    denom = (np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1) + 1e-9)
    cos = (np.sum(a*b,axis=1)/denom).mean()
    amag = np.linalg.norm(a,axis=1); bmag = np.linalg.norm(b,axis=1)
    if np.std(amag)<1e-9 or np.std(bmag)<1e-9:
        corr = 0.0
    else:
        corr = float(np.corrcoef(amag, bmag)[0,1])
    return {"epe_mean": float(epe), "vec_cosine": float(cos), "mag_corr": float(corr)}

def temporal_diff_sequence(v: np.ndarray) -> np.ndarray:
    diffs = []
    for t in range(len(v)-1):
        diffs.append(cv2.absdiff(v[t], v[t+1]))
    return np.stack(diffs, axis=0) if diffs else np.zeros((0, v.shape[1], v.shape[2], 3), dtype=np.uint8)

def temporal_diff_similarity(gt: np.ndarray, gen: np.ndarray) -> dict:
    gt_d = temporal_diff_sequence(gt)
    gen_d = temporal_diff_sequence(gen)
    Td = min(len(gt_d), len(gen_d))
    if Td==0:
        return {"t_ssim_mean": float('nan'), "t_cosine_rgb": float('nan')}
    vals = []; cos_vals = []
    for i in range(Td):
        g = to_gray(gt_d[i]); f = to_gray(gen_d[i])
        if _HAS_SKIMAGE:
            vals.append(ssim(g, f, data_range=255))
        else:
            g_ = g.astype(np.float32)/255.0; f_ = f.astype(np.float32)/255.0
            mse = np.mean((g_-f_)**2); vals.append(1.0/(1.0+mse))
        gr = gt_d[i].astype(np.float32).ravel(); fr = gen_d[i].astype(np.float32).ravel()
        cos = float(np.dot(gr, fr) / ((np.linalg.norm(gr)+1e-9)*(np.linalg.norm(fr)+1e-9)))
        cos_vals.append(cos)
    return {"t_ssim_mean": float(np.mean(vals)), "t_cosine_rgb": float(np.mean(cos_vals))}


def _torch_cuda_ok(device: str) -> bool:
    if device.lower() == "cuda":
        return torch.cuda.is_available()
    return True

def _get_r2plus1d18_feature_extractor(device="cpu", debug=False):
    if not (_HAS_TORCH and _HAS_TORCHVISION):
        return None, 0
    import torchvision
    from torch import nn
    model = None
    try:
        from torchvision.models.video import R2Plus1D_18_Weights
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        model = torchvision.models.video.r2plus1d_18(weights=weights)
    except Exception:
        pass
    if model is None:
        try:
            model = torchvision.models.video.r2plus1d_18(pretrained=True)
        except Exception:
            return None, 0
    model.eval().to(device)
    class FeatWrapper(nn.Module):
        def __init__(self, base):
            super().__init__(); self.base = base
        def forward(self, x):
            x = self.base.stem(x)
            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)
            x = self.base.avgpool(x)
            x = torch.flatten(x, 1)
            return x
    return FeatWrapper(model), 512

def _video_to_clips(frames_bgr: np.ndarray, clip_len=16, stride=8, size=(112,112), debug=False) -> np.ndarray:
    T = int(frames_bgr.shape[0])
    clips = []
    for start in range(0, max(1, T - clip_len + 1), stride):
        end = start + clip_len
        if end > T: break
        clip = frames_bgr[start:end]
        clip = np.stack([cv2.resize(fr, size, interpolation=cv2.INTER_AREA) for fr in clip], axis=0)
        clip = clip[:,:,:,::-1].astype(np.float32)/255.0
        clips.append(clip)
    if not clips:
        pad = max(0, clip_len - T)
        pad_stack = np.repeat(frames_bgr[-1:,:,:,:], pad, axis=0) if T>0 else np.zeros((clip_len, size[0], size[1], 3), np.uint8)
        clip = frames_bgr if T>0 else pad_stack
        if T>0: clip = np.concatenate([clip, pad_stack], axis=0)
        clip = np.stack([cv2.resize(fr, size, interpolation=cv2.INTER_AREA) for fr in clip], axis=0)
        clip = clip[:,:,:,::-1].astype(np.float32)/255.0
        clips = [clip]
    out = np.stack(clips, axis=0)
    return out

def _normalize_kinetics(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=x.device).view(1,3,1,1,1)
    std  = torch.tensor([0.22803, 0.22145, 0.216989], device=x.device).view(1,3,1,1,1)
    return (x - mean) / std

def _extract_features_r2p1d(frames_bgr: np.ndarray, device="cpu", clip_len=16, stride=8, size=(112,112), debug=False) -> np.ndarray:
    model, dim = _get_r2plus1d18_feature_extractor(device=device, debug=debug)
    if model is None: return np.zeros((0, 512), dtype=np.float32)
    with torch.no_grad():
        clips = _video_to_clips(frames_bgr, clip_len=clip_len, stride=stride, size=size, debug=debug)
        clips = torch.from_numpy(np.transpose(clips, (0,4,1,2,3))).to(device)
        clips = _normalize_kinetics(clips)
        feats = []
        bs = 8
        for i in range(0, clips.shape[0], bs):
            f = model(clips[i:i+bs])
            feats.append(f.detach().cpu().numpy().astype(np.float32))
        feats = np.concatenate(feats, axis=0)
    return feats

def _cov_bias0(X: np.ndarray) -> np.ndarray:
    N = X.shape[0]
    if N == 0:
        return np.eye(X.shape[1], dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / max(N, 1)
    return C.astype(np.float64)

def _frechet_distance(m1, C1, m2, C2):
    diff = (m1 - m2).astype(np.float64)
    diff_sq = float(diff @ diff)
    eps = 1e-6
    C1 = ((C1 + C1.T)*0.5) + np.eye(C1.shape[0])*eps
    C2 = ((C2 + C2.T)*0.5) + np.eye(C2.shape[0])*eps
    w1, V1 = np.linalg.eigh(C1); w1 = np.clip(w1, 0, None)
    sqrtC1 = V1 @ np.diag(np.sqrt(w1)) @ V1.T
    invsqrtC1 = V1 @ np.diag(1.0/np.sqrt(np.clip(w1, eps, None))) @ V1.T
    M = invsqrtC1 @ (C1 @ C2) @ invsqrtC1
    wM, VM = np.linalg.eigh((M + M.T)*0.5); wM = np.clip(wM, 0, None)
    sqrtM = VM @ np.diag(np.sqrt(wM)) @ VM.T
    tr_term = float(np.trace(sqrtC1 @ sqrtM @ sqrtC1))
    return diff_sq + float(np.trace(C1 + C2 - 2.0 * (sqrtC1 @ sqrtM @ sqrtC1)))



# -------- semantic ----------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def _resize_rgb(img_bgr: np.ndarray, size=(224,224), norm_mean=_IMAGENET_MEAN, norm_std=_IMAGENET_STD) -> np.ndarray:
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32)/255.0
    img = (img - norm_mean) / norm_std
    return img

class _ViTWrapper:
    def __init__(self, device="cpu"):
        self.device = device
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True)
        self.model.eval().to(device)

    @torch.no_grad()
    def frame_tokens(self, img_bgr: np.ndarray) -> np.ndarray:
        img = _resize_rgb(img_bgr, (224,224), _IMAGENET_MEAN, _IMAGENET_STD)
        x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
        out = self.model.forward_features(x)
        if isinstance(out, dict) and 'x' in out:
            tokens = out['x']
        else:
            tokens = out
        tokens = tokens[:, 1:, :]
        return tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def frame_global(self, img_bgr: np.ndarray) -> np.ndarray:
        toks = self.frame_tokens(img_bgr)
        return toks.mean(axis=0)

class _CLIPWrapper:
    def __init__(self, device="cpu"):
        self.device = device
        self.size = 224
        if _HAS_OPENCLIP:
            self.model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
            self.model.eval()
            self.use_openclip = True
        elif _HAS_CLIP:
            self.model, _ = _clip.load("ViT-B/32", device=device)
            self.model.eval()
            self.use_openclip = False
        else:
            self.model = None
            self.use_openclip = None

    @torch.no_grad()
    def frame_embed(self, img_bgr: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros((512,), dtype=np.float32)
        img = _resize_rgb(img_bgr, (self.size,self.size), _CLIP_MEAN, _CLIP_STD)
        x = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
        return feat.squeeze(0).detach().cpu().numpy().astype(np.float32)

def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12; nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na*nb))

def _vit_score_pair(frameA: np.ndarray, frameB: np.ndarray, vit: _ViTWrapper) -> float:
    Xa = vit.frame_tokens(frameA)
    Yb = vit.frame_tokens(frameB)
    Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-12)
    Yb = Yb / (np.linalg.norm(Yb, axis=1, keepdims=True) + 1e-12)
    S = Xa @ Yb.T
    R = float(S.max(axis=1).mean())
    P = float(S.max(axis=0).mean())
    if (R+P) <= 1e-12: return 0.0
    return float(2 * R * P / (R + P))

def semantic_scores(gt: np.ndarray, gen: np.ndarray, device="cpu") -> dict:
    T = min(len(gt), len(gen))
    vit_lin, clip_lin = [], []
    vit_emb_gt, vit_emb_ge = [], []
    clip_emb_gt, clip_emb_ge = [], []
    vit = None; clip = None
    if _HAS_TORCH and _HAS_TIMM:
        try:
            vit = _ViTWrapper(device=device)
        except Exception:
            vit = None
    if _HAS_TORCH and (_HAS_OPENCLIP or _HAS_CLIP):
        try:
            clip = _CLIPWrapper(device=device)
        except Exception:
            clip = None
    if vit is None and clip is None:
        return {
            "vit_score_mean_linear": float('nan'),
            "clip_score_mean_linear": float('nan'),
            "vit_score_mean_dtw": float('nan'),
            "clip_score_mean_dtw": float('nan'),
            "vit_backbone": "vit_base_patch16_224" if _HAS_TIMM else "NA",
            "clip_backbone": "ViT-B-32" if (_HAS_OPENCLIP or _HAS_CLIP) else "NA"
        }
    for t in range(T):
        fgt = gt[t]; fge = gen[t]
        if vit is not None:
            try:
                s = _vit_score_pair(fgt, fge, vit)
                vit_lin.append(s)
                vit_emb_gt.append(vit.frame_global(fgt))
                vit_emb_ge.append(vit.frame_global(fge))
            except Exception:
                vit_lin.append(np.nan)
        if clip is not None:
            try:
                eg = clip.frame_embed(fgt); ee = clip.frame_embed(fge)
                clip_lin.append(_cos_sim(eg, ee))
                clip_emb_gt.append(eg); clip_emb_ge.append(ee)
            except Exception:
                clip_lin.append(np.nan)
    vit_mean_linear = float(np.nanmean(vit_lin)) if vit_lin else float('nan')
    clip_mean_linear = float(np.nanmean(clip_lin)) if clip_lin else float('nan')

    def dtw_mean_sim(seqA, seqB):
        if len(seqA)==0 or len(seqB)==0: return float('nan')
        A = np.stack(seqA, axis=0); B = np.stack(seqB, axis=0)
        A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        def cost(i,j):
            return float(1.0 - np.dot(A[i], B[j]))
        _, path = dtw_distance(np.zeros(len(A)), np.zeros(len(B)), cost_fn=cost)
        sims = [1.0 - cost(i,j) for (i,j) in path]
        return float(np.mean(sims)) if sims else float('nan')

    vit_mean_dtw  = dtw_mean_sim(vit_emb_gt, vit_emb_ge) if vit is not None else float('nan')
    clip_mean_dtw = dtw_mean_sim(clip_emb_gt, clip_emb_ge) if clip is not None else float('nan')

    return {
        "vit_score_mean_linear": vit_mean_linear,
        "clip_score_mean_linear": clip_mean_linear,
        "vit_score_mean_dtw": vit_mean_dtw,
        "clip_score_mean_dtw": clip_mean_dtw,
        "vit_backbone": "vit_base_patch16_224" if vit is not None else ("vit_base_patch16_224" if _HAS_TIMM else "NA"),
        "clip_backbone": "ViT-B-32" if clip is not None else ("ViT-B-32" if (_HAS_OPENCLIP or _HAS_CLIP) else "NA"),
    }


def evaluate_pair(
        gt: np.ndarray, gen: np.ndarray,
        use_hsv: bool, red_th: float, cube_grid: Tuple[int,int,int],
        want_semantic: bool, semantic_device: str
    ) -> Dict[str, Any]:

    pix_psnr = frame_psnr_mean(gt, gen)
    pix_ssim = frame_ssim_mean(gt, gen)
    pix_msssim = msssim(gt, gen)
    pix_lpips = frame_lpips_mean(gt, gen)

    gt_flow = optical_flow_farneback_sequence(gt)
    gen_flow = optical_flow_farneback_sequence(gen)
    flow_stats = flow_metrics(gt_flow, gen_flow)
    tdiff_stats = temporal_diff_similarity(gt, gen)

    gt_masks = masks_from_video(gt, use_hsv=use_hsv, red_th=red_th)
    gen_masks = masks_from_video(gen, use_hsv=use_hsv, red_th=red_th)
    gt_curve = red_coverage_curve(gt_masks)
    gen_curve = red_coverage_curve(gen_masks)
    dtw_dist, path = dtw_distance(gt_curve, gen_curve)
    gen_curve_warp = warp_by_path(gen_curve, path, as_ref="a")
    gt_curve_warp  = _interp_to_len(gt_curve, len(gen_curve_warp))

    def safe_corr(a,b):
        if len(a)<2 or len(b)<2: return 0.0
        if np.std(a)<1e-9 or np.std(b)<1e-9: return 0.0
        return float(np.corrcoef(a,b)[0,1])

    pearson_r = safe_corr(gt_curve_warp, gen_curve_warp)
    mae_curve = float(np.mean(np.abs(gt_curve_warp - gen_curve_warp)))
    rmse_curve = float(np.sqrt(np.mean((gt_curve_warp - gen_curve_warp)**2)))

    gt_heat = occupancy_heatmap(gt_masks, blur=5)
    gen_heat = occupancy_heatmap(gen_masks, blur=5)
    heat_ssim = ssim_safe(gt_heat, gen_heat)
    heat_cos  = cosine_similarity(gt_heat, gen_heat)

    gt_cube = occupancy_cube(gt_masks, grid=cube_grid)
    gen_cube = occupancy_cube(gen_masks, grid=cube_grid)
    cube_cos = cosine_similarity(gt_cube, gen_cube)

    out = {
        "pixel_perceptual": {
            "psnr_mean": pix_psnr,
            "ssim_mean": pix_ssim,
            "ms_ssim_mean": pix_msssim,
            "lpips_mean": pix_lpips
        },
        "motion_dynamics": {
            **flow_stats,
            **tdiff_stats
        },
        "temporal_bio": {
            "pearson_r": pearson_r,
            "dtw_distance": dtw_dist,
            "mae_curve": mae_curve,
            "rmse_curve": rmse_curve,
            "gt_curve_len": int(len(gt_curve)),
            "gen_curve_len": int(len(gen_curve)),
        },
        "spatial_bio": {
            "heatmap_ssim": heat_ssim,
            "heatmap_cosine": heat_cos,
        },
        "spatiotemporal_bio": {
            "cube_cosine": cube_cos,
            "cube_grid": list(cube_grid),
        }
    }

    if want_semantic:
        sem = semantic_scores(gt, gen, device=semantic_device)
        out["semantic"] = sem

    return out


CSV_HEADER = [
    "dataset", "method", "video_name",
    "psnr", "ssim", "ms_ssim", "lpips",
    "epe", "flow_cos", "flow_mag_corr", "t_ssim", "t_cos_rgb",
    "pearson_r", "dtw", "mae", "rmse", "gt_len", "gen_len",
    "heat_ssim", "heat_cos", "cube_cos",
    "vit_linear", "clip_linear", "vit_dtw", "clip_dtw"
]

def flatten_metrics(dataset_name: str, method: str, video_name: str, m: Dict[str, Any]) -> Dict[str, Any]:
    p = m["pixel_perceptual"]
    md = m["motion_dynamics"]
    tb = m["temporal_bio"]
    sb = m["spatial_bio"]
    stb = m["spatiotemporal_bio"]
    sem = m.get("semantic", {})
    return {
        "dataset": dataset_name,
        "method": method,
        "video_name": video_name,
        "psnr": p["psnr_mean"],
        "ssim": p["ssim_mean"],
        "ms_ssim": p["ms_ssim_mean"],
        "lpips": p["lpips_mean"],
        "epe": md["epe_mean"],
        "flow_cos": md["vec_cosine"],
        "flow_mag_corr": md["mag_corr"],
        "t_ssim": md["t_ssim_mean"],
        "t_cos_rgb": md["t_cosine_rgb"],
        "pearson_r": tb["pearson_r"],
        "dtw": tb["dtw_distance"],
        "mae": tb["mae_curve"],
        "rmse": tb["rmse_curve"],
        "gt_len": tb["gt_curve_len"],
        "gen_len": tb["gen_curve_len"],
        "heat_ssim": sb["heatmap_ssim"],
        "heat_cos": sb["heatmap_cosine"],
        "cube_cos": stb["cube_cosine"],
        "vit_linear": sem.get("vit_score_mean_linear", float("nan")),
        "clip_linear": sem.get("clip_score_mean_linear", float("nan")),
        "vit_dtw": sem.get("vit_score_mean_dtw", float("nan")),
        "clip_dtw": sem.get("clip_score_mean_dtw", float("nan")),
    }

def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(CSV_HEADER) + "\n")
        for r in rows:
            vals = []
            for k in CSV_HEADER:
                v = r.get(k, "")
                if isinstance(v, float):
                    if np.isnan(v):
                        vals.append("nan")
                    else:
                        vals.append(f"{v:.6f}")
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")

def is_video_file(name: str) -> bool:
    name_low = name.lower()
    return name_low.endswith(".mp4") or name_low.endswith(".avi") or name_low.endswith(".mov") or name_low.endswith(".mkv")

def find_gt_path(root: str, gt_stem: str) -> str:
    """
    root: E:\\07
    gt_stem: 07-GT
    try: dir, .mp4, .avi
    """
    cand_dir = os.path.join(root, gt_stem)
    if os.path.isdir(cand_dir):
        return cand_dir
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        cand = os.path.join(root, gt_stem + ext)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"GT not found in {root} with stem={gt_stem}")

def collect_method_videos(root: str, dataset_stem: str) -> Dict[str, List[str]]:
    """
    root = E:\\01-ICA
    video names like 01-ICA-CLOT-1.mp4
    return: {CLOT: [path1, path2, ...], Wan: [...], ...}
    """
    method_dict: Dict[str, List[str]] = {}
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if os.path.isdir(full):

            parts = name.split("-")
            if len(parts) >= 3 and parts[0] == dataset_stem.split("-")[0]:

                method = parts[-2]
                if method.upper() == "GT":  # 跳过GT
                    continue
                method_dict.setdefault(method, []).append(full)
        else:
            if not is_video_file(name):
                continue
            base = os.path.splitext(name)[0]
            parts = base.split("-")
            if len(parts) < 3:
                continue
            # e.g. 01-ICA-CLOT-1 -> ["RC","ICA","CLOT","1"]
            #      07-CLOT-1     -> ["07","CLOT","1"]
            method = parts[-2]
            if method.upper() == "GT":
                continue
            method_dict.setdefault(method, []).append(full)
    return method_dict

def evaluate_one_dataset(
        dataset_name: str,
        root_dir: str,
        gt_stem: str,
        resize_hw=None,
        denoise="median",
        stabilize=True,
        ref_channel="G",
        use_hsv=True,
        red_th=0.6,
        cube_grid=(16,16,16),
        want_semantic=False,
        semantic_device="cpu",
        out_root="eval_outputs"
    ) -> List[Dict[str, Any]]:
    """
    return record for each video
    """
    print(f"[INFO] === Dataset: {dataset_name} ===")
    gt_path = find_gt_path(root_dir, gt_stem)
    print(f"[INFO] GT = {gt_path}")

    gt = load_video(gt_path)
    if resize_hw is not None:
        gt = resize_video(gt, resize_hw)
    if denoise and denoise.lower() != "none":
        gt = denoise_video(gt, denoise.lower())
    if stabilize:
        gt = stabilize_video(gt, ref_channel=ref_channel)
    H, W = gt.shape[1], gt.shape[2]

    method_videos = collect_method_videos(root_dir, dataset_name)
    print(f"[INFO] found methods in {dataset_name}:", list(method_videos.keys()))

    rows: List[Dict[str, Any]] = []

    for method, video_list in method_videos.items():

        video_list = sorted(video_list, key=natural_key)
        for video_path in video_list:
            print(f"[INFO]   eval {dataset_name} / {method} / {os.path.basename(video_path)}")
            gen = load_video(video_path)

            if resize_hw is None and (gen.shape[1] != H or gen.shape[2] != W):
                gen = resize_video(gen, (H, W))
            if resize_hw is not None:
                gen = resize_video(gen, resize_hw)
            if denoise and denoise.lower() != "none":
                gen = denoise_video(gen, denoise.lower())
            if stabilize:
                gen = stabilize_video(gen, ref_channel=ref_channel)

            metrics = evaluate_pair(
                gt=gt, gen=gen,
                use_hsv=use_hsv, red_th=red_th, cube_grid=cube_grid,
                want_semantic=want_semantic, semantic_device=semantic_device
            )
            flat = flatten_metrics(
                dataset_name=dataset_name,
                method=method,
                video_name=os.path.basename(video_path),
                m=metrics
            )
            rows.append(flat)


    out_csv = os.path.join(out_root, f"{dataset_name}_metrics.csv")
    write_csv(out_csv, rows)
    print(f"[OK] dataset-level csv written to {out_csv}")
    return rows

def average_across_datasets(all_rows: List[Dict[str, Any]], out_path: str):
    """
    all_rows to make a table
    """
    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_method.setdefault(r["method"], []).append(r)

    avg_rows: List[Dict[str, Any]] = []
    for method, rows in by_method.items():
        avg_row = {"dataset": "ALL", "method": method, "video_name": "AVG"}
        for k in CSV_HEADER:
            if k in ["dataset", "method", "video_name"]:
                continue

            vals = []
            for r in rows:
                v = r.get(k, None)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                    vals.append(float(v))
            if vals:
                avg_row[k] = float(np.mean(vals))
            else:
                avg_row[k] = float("nan")
        avg_rows.append(avg_row)

    write_csv(out_path, avg_rows)
    print(f"[OK] global average csv written to {out_path}")

def main():

    datasets = [
        ("07-CCA",     r"E:\07-CCA",     "07-GT-0"),
        ("01-CCA", r"E:\01-CCA", "01-CCA-GT-0"),
        ("01-ICA", r"E:\01-ICA", "01-ICA-GT-0"),
    ]


    resize_hw = None
    denoise   = "median"
    stabilize = True
    ref_channel = "G"
    use_hsv = True
    red_th = 0.6
    cube_grid = (16,16,16)
    want_semantic = True
    semantic_device = "cpu"
    out_root = "eval_all_out1"

    os.makedirs(out_root, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for (ds_name, ds_root, ds_gt) in datasets:
        rows = evaluate_one_dataset(
            dataset_name=ds_name,
            root_dir=ds_root,
            gt_stem=ds_gt,
            resize_hw=resize_hw,
            denoise=denoise,
            stabilize=stabilize,
            ref_channel=ref_channel,
            use_hsv=use_hsv,
            red_th=red_th,
            cube_grid=cube_grid,
            want_semantic=want_semantic,
            semantic_device=semantic_device,
            out_root=out_root
        )
        all_rows.extend(rows)

    # calculating average
    avg_csv = os.path.join(out_root, "ALL_datasets_method_avg_dl.csv")
    average_across_datasets(all_rows, avg_csv)

if __name__ == "__main__":
    main()
