import torch
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import re, math
import torch
import torch.nn as nn
from contextlib import suppress

def _extract_shapes(x):
    if isinstance(x, torch.Tensor):
        return [tuple(x.shape)]
    if isinstance(x, (list, tuple)):
        return [tuple(t.shape) for t in x if isinstance(t, torch.Tensor)]
    if isinstance(x, dict):
        if isinstance(x.get("last_hidden_state", None), torch.Tensor):
            return [tuple(x["last_hidden_state"].shape)]
        for v in x.values():
            if isinstance(v, torch.Tensor):
                return [tuple(v.shape)]
        return []

    if hasattr(x, "last_hidden_state") and isinstance(x.last_hidden_state, torch.Tensor):
        return [tuple(x.last_hidden_state.shape)]
    return []

def _count_params(m: nn.Module):
    try:
        return sum(p.numel() for p in m.parameters())
    except Exception:
        return 0

def register_key_module_hooks(root_module: nn.Module,
                              include_patterns=None,
                              verbose=False):
    if include_patterns is None:
        include_patterns = [
            # ---- Text encoders  ----
            r"(?:^|[.\-_])(text|txt)\w*(enc|encoder)\b",
            r"(?:^|[.\-_])(umt5|t5|bart|bert|roberta)\b",

            # ---- Vision / Image encoders (CLIP / OpenCLIP / ViT / projector) ----
            r"(?:^|[.\-_])(clip|openclip|clip_model|clipvision|vision|vit)\b",
            r"image[\s_\-]?(enc|encoder|proj|project|projector)\b",

            # ---- Video / I2V  ----
            r"(?:^|[.\-_])video[\s_\-]?(enc|encoder|proj|projector)\b",
            r"(?:^|[.\-_])(i2v|image2video)\b",
            r"(?:^|[.\-_])(spatio[\w\-]*temporal|temporal)[\w\-]*(enc|block|proj|projector|transformer)\b",

            # ---- VAE family (first-stage / encoder / decoder / prior / posterior) ----
            r"(?:^|[.\-_])(vae|autoenc|autoencoder|first[\s_\-]?stage|prior|posterior)\b",
            r"(?:^|[.\-_])vae[\w\-]*[\./]?(encoder|decoder)\b",  #  encoder/decoder VAE

            # ---- Diffusion backbone (DiT / STDiT / WanDiT / UNet / Transformer2D/3D / denoiser) ----
            r"(?:^|[.\-_])(dit|stdit|wan[\w\-]*dit)\b",
            r"(?:^|[.\-_])diffusion[\w\-]*(transformer|model)?\b",
            r"(?:^|[.\-_])unet[\w\-]*(2d|3d|condition|spatio|temporal)?\b",
            r"(?:^|[.\-_])transformer[23]d\b",
            r"(?:^|[.\-_])denois[\w\-]+|noise[\s_\-]?pred[\w\-]*",
        ]

    regs = [re.compile(p, re.I) for p in include_patterns]


    cand = []
    for name, m in root_module.named_modules():
        fullkey = f"{name}::{m.__class__.__name__}".lower()
        if any(r.search(fullkey) for r in regs):
            cand.append((name, m))


    cand.sort(key=lambda x: x[0].count('.'))  # 先短路径
    selected = []
    prefixes = []
    for name, m in cand:
        if any(name.startswith(p + '.') or name == p for p in prefixes):
            continue
        selected.append((name, m))
        prefixes.append(name)

    if verbose:
        print("[shape-hook] selected modules:", [n for n,_ in selected])

    def _hook(m, inp, out):
        if getattr(m, "_shape_logged_once", False):
            return
        in_shapes = []
        for t in inp:
            if isinstance(t, torch.Tensor):
                in_shapes.append(tuple(t.shape))
        out_shapes = _extract_shapes(out)

        print(f"[KEY] {path_map[m]} ({m.__class__.__name__}) "
              f"in {in_shapes} -> out {out_shapes} ; "
              f"params={_count_params(m)/1e6:.2f}M")
        m._shape_logged_once = True

    path_map = {m: n for n, m in root_module.named_modules()}
    for name, m in selected:
        with suppress(Exception):
            m.register_forward_hook(_hook)

# Load models
model_manager = ModelManager(device="cpu")
model_manager.load_models(
    ["models/Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32, # Image Encoder is loaded with float32
)
model_manager.load_models(
    [
        [
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00001-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00002-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00003-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00004-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00005-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00006-of-00007.safetensors",
            "models/Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "models/Wan-AI/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
        "models/Wan-AI/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
)
model_manager.load_lora("models/lightning_logs/version_72/checkpoint.ckpt", lora_alpha=1.0)
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=6*10**9) # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
register_key_module_hooks(pipe)
#geometry-1
image1 = Image.open("data/examples/wan/G1-original.jpg")
image2 = Image.open("data/examples/wan/G1-ablation.jpg")

prompt = "The video shows a 2D microscopy image of a patient's carotid artery bifurcation. The blood flow is from left to right (streaming from the single larger blood vessel into the 2 bifurcated branches). The video has a black background, and the vessel shape is visualised using green meshes, which represent endothelial cells lining the vessel walls. There is a laser cut injury on the vessel wall, which is visualised with a black gap in the green meshes, where endothelial cells have fallen off. Blood is flown through the vessel structure, red fluorescent signals representing platelets, from the larger single blood vessel branch to the bifurcated branches.  Platelets accumulate and detach at the laser injury zone and fallen off."

negative_prompt = "bright colors, overexposed, static, blurred details, subtitles, style, work, painting, picture, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, deformed limbs, fused fingers, static picture, cluttered background, three legs, many people in the background, walking backwards"


# Image-to-video
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    input_image= image2,
    input_video=[image1, image2],
    num_inference_steps=50,
    num_frames=65,
    height=512,
    width=512,
    seed=0, tiled=True
)
print("prompt: ", prompt)
print("negative_prompt: ", negative_prompt)
video_output = "Geometry-G1-f65.mp4"
print("video output: ", video_output)
save_video(video, video_output, fps=14, quality=5)
