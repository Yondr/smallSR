#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended optimized pipeline:
- Stages: DENOISE (NAFNet-SIDD), DEBLUR (NAFNet-REDS), SR (NAFSSR-L_4x)
- Any stage can be Torch or ONNX: --denoise-weights/--denoise-onnx, --deblur-weights/--deblur-onnx, --sr-weights/--sr-onnx
- AMP/FP16, tiling, kornia unsharp on GPU, threaded I/O
- Default order: Denoise -> Deblur -> SR
"""
import sys
import argparse
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn.functional as F
try:
    import onnxruntime as ort
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False
try:
    import kornia as K
except Exception:
    K = None
import cv2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
try:
    from image_pipeline.nafssr_arch import NAFNetSR  # SR backbone
except Exception:
    NAFNetSR = None
try:
    from image_pipeline.nafnet_arch import NAFNet  # restoration backbone
except Exception:
    NAFNet = None

def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def _imsave(path: str, rgb: np.ndarray, fmt: str = "png", jpg_quality: int = 90, png_level: int = 1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ext = fmt.lower()
    if ext in ["jpg", "jpeg"]:
        cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])[1].tofile(path)
    elif ext == "webp":
        cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), int(jpg_quality)])[1].tofile(path)
    else:
        cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_level)])[1].tofile(path)

def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().float().div_(255.0)
    return t

def _to_image(t: torch.Tensor) -> np.ndarray:
    t = t.detach().clamp_(0.0, 1.0)
    t = (t * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return t

def parse_size(s: str):
    if s is None:
        return None
    if "x" not in s.lower():
        raise ValueError("Size must be WxH")
    w, h = s.lower().split("x")
    return (int(w), int(h))

def hann2d(h, w):
    wy = np.hanning(h).reshape(-1, 1)
    wx = np.hanning(w).reshape(1, -1)
    w2 = (wy @ wx).astype(np.float32)
    m = float(w2.max()) if w2.size else 1.0
    if m > 0:
        w2 /= m
    return w2

@torch.inference_mode()
def tiled_forward_torch(model, x, tile: int, overlap: int, use_amp: bool):
    device = x.device
    _, _, H, W = x.shape
    if tile <= 0 or tile >= max(H, W):
        with torch.cuda.amp.autocast(enabled=use_amp):
            return model(x)
    stride = tile - overlap
    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))
    if xs and xs[-1] + tile > W:
        xs[-1] = max(0, W - tile)
    if ys and ys[-1] + tile > H:
        ys[-1] = max(0, H - tile)
    with torch.cuda.amp.autocast(enabled=use_amp):
        y0 = model(x[:, :, ys[0]:ys[0]+tile, xs[0]:xs[0]+tile])
    _, c2, Ht, Wt = y0.shape
    scale_h = Ht / float(tile)
    scale_w = Wt / float(tile)
    out = torch.zeros((1, c2, int(round(H*scale_h)), int(round(W*scale_w))), dtype=y0.dtype, device=device)
    weight = torch.zeros_like(out, dtype=torch.float32)
    base_win = hann2d(int(tile*scale_h), int(tile*scale_w))
    for yy in ys:
        for xx in xs:
            with torch.cuda.amp.autocast(enabled=use_amp):
                ytile = model(x[:, :, yy:yy+tile, xx:xx+tile])
            y_h, y_w = ytile.shape[-2:]
            y0o = int(round(yy * scale_h)); x0o = int(round(xx * scale_w))
            y1o = y0o + y_h; x1o = x0o + y_w
            wpatch = base_win
            if wpatch.shape != (y_h, y_w):
                wpatch = cv2.resize(wpatch, (y_w, y_h), interpolation=cv2.INTER_LINEAR)
            wpatch_t = torch.from_numpy(wpatch).to(device=device, dtype=ytile.dtype).unsqueeze(0).unsqueeze(0)
            out[:, :, y0o:y1o, x0o:x1o] += ytile * wpatch_t
            weight[:, :, y0o:y1o, x0o:x1o] += wpatch_t.to(weight.dtype)
    out = out / weight.clamp_min(1e-6)
    return out

def forward_onnx(sess, x_np: np.ndarray, tile: int, overlap: int):
    if tile <= 0:
        return sess.run(["output"], {"input": x_np})[0]
    _, _, H, W = x_np.shape
    stride = tile - overlap
    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))
    if xs and xs[-1] + tile > W:
        xs[-1] = max(0, W - tile)
    if ys and ys[-1] + tile > H:
        ys[-1] = max(0, H - tile)
    y0 = sess.run(["output"], {"input": x_np[:, :, ys[0]:ys[0]+tile, xs[0]:xs[0]+tile]})[0]
    _, c2, Ht, Wt = y0.shape
    scale_h = Ht / float(tile)
    scale_w = Wt / float(tile)
    out = np.zeros((1, c2, int(round(H*scale_h)), int(round(W*scale_w))), dtype=y0.dtype)
    weight = np.zeros_like(out, dtype=np.float32)
    base_win = hann2d(int(tile*scale_h), int(tile*scale_w))
    for yy in ys:
        for xx in xs:
            ytile = sess.run(["output"], {"input": x_np[:, :, yy:yy+tile, xx:xx+tile]})[0]
            y_h, y_w = ytile.shape[-2:]
            y0o = int(round(yy * scale_h)); x0o = int(round(xx * scale_w))
            y1o = y0o + y_h; x1o = x0o + y_w
            wpatch = base_win
            if wpatch.shape != (y_h, y_w):
                wpatch = cv2.resize(wpatch, (y_w, y_h), interpolation=cv2.INTER_LINEAR)
            for c in range(c2):
                out[0, c, y0o:y1o, x0o:x1o] += ytile[0, c] * wpatch
                weight[0, c, y0o:y1o, x0o:x1o] += wpatch
    out = out / np.clip(weight, 1e-6, None)
    return out

def kornia_unsharp(t: torch.Tensor, sigma: float = 1.0, amount: float = 1.0):
    if K is None or amount <= 1e-6:
        return t
    return K.filters.unsharp_mask(t, kernel_size=(3,3), sigma=(sigma, sigma), amount=amount)

def load_sr_torch(weights: str, device: torch.device):
    if NAFNetSR is None:
        raise RuntimeError("NAFNetSR not found")
    m = NAFNetSR()
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("params", ckpt)
    m.load_state_dict(state, strict=False)
    m.eval().to(device)
    return m

def load_restorer_torch(weights: str, device: torch.device, width: int):
    if NAFNet is None:
        raise RuntimeError("NAFNet not found")
    # Reasonable default blocks; many public NAFNet weights use these counts
    m = NAFNet(width=width, enc_blk_nums=[1,1,1,28], middle_blk_num=1, dec_blk_nums=[1,1,1,1])
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("params", ckpt)
    m.load_state_dict(state, strict=False)
    m.eval().to(device)
    return m

def make_ort_session(path: str):
    if not _HAS_ORT:
        raise RuntimeError("onnxruntime not installed. pip install onnxruntime-gpu")
    providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1"})]
    return ort.InferenceSession(path, providers=providers)

class Stage:
    def __init__(self, name, torch_model=None, onnx_sess=None, fp16=True, device="cuda", tile=0, overlap=16):
        self.name = name
        self.torch_model = torch_model
        self.onnx_sess = onnx_sess
        self.fp16 = fp16
        self.device = device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"
        self.tile = tile
        self.overlap = overlap
    def run(self, x: torch.Tensor) -> torch.Tensor:
        if self.onnx_sess is not None:
            x_np = x.float().cpu().numpy()
            if self.fp16:
                x_np = x_np.astype(np.float16)
            y_np = forward_onnx(self.onnx_sess, x_np, tile=self.tile, overlap=self.overlap)
            y = torch.from_numpy(y_np).to(self.device if self.device == "cuda" else "cpu")
            return y
        elif self.torch_model is not None:
            x = x.to(self.device, non_blocking=True)
            if self.fp16 and self.device == "cuda":
                x = x.half()
            return tiled_forward_torch(self.torch_model, x, tile=self.tile, overlap=self.overlap, use_amp=(self.fp16 and self.device=="cuda"))
        else:
            return x

def process_images(
    inputs,
    output_dir: str,
    denoise_weights: str = None, denoise_onnx: str = None, denoise_width: int = 32,
    deblur_weights: str = None,  deblur_onnx: str = None,  deblur_width: int = 64,
    sr_weights: str = None,      sr_onnx: str = None,
    device_str: str = "cuda",
    fp16: bool = True,
    tile: int = 0,
    overlap: int = 16,
    sharpen_amount: float = 1.0,
    sharpen_sigma: float = 1.0,
    final_size: str = None,
    save_fmt: str = "png",
    jpg_quality: int = 90,
    png_level: int = 1,
    io_workers: int = 4,
):
    device = torch.device(device_str if torch.cuda.is_available() and device_str.startswith("cuda") else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    stages = []
    if denoise_onnx:
        stages.append(Stage("denoise", onnx_sess=make_ort_session(denoise_onnx), fp16=fp16, device=device_str, tile=tile, overlap=overlap))
    elif denoise_weights:
        stages.append(Stage("denoise", torch_model=load_restorer_torch(denoise_weights, device, width=denoise_width), fp16=fp16, device=device_str, tile=tile, overlap=overlap))
    if deblur_onnx:
        stages.append(Stage("deblur", onnx_sess=make_ort_session(deblur_onnx), fp16=fp16, device=device_str, tile=tile, overlap=overlap))
    elif deblur_weights:
        stages.append(Stage("deblur", torch_model=load_restorer_torch(deblur_weights, device, width=deblur_width), fp16=fp16, device=device_str, tile=tile, overlap=overlap))
    if sr_onnx:
        stages.append(Stage("sr", onnx_sess=make_ort_session(sr_onnx), fp16=fp16, device=device_str, tile=tile, overlap=overlap))
    elif sr_weights:
        stages.append(Stage("sr", torch_model=load_sr_torch(sr_weights, device), fp16=fp16, device=device_str, tile=tile, overlap=overlap))

    size_tuple = parse_size(final_size) if final_size else None

    def _run_one(img_path: str):
        name = Path(img_path).stem
        out_path = str(Path(output_dir) / f"{name}.{save_fmt}")
        img = _imread_rgb(img_path)
        t = _to_tensor(img)
        y = t
        for st in stages:
            y = st.run(y)
        if sharpen_amount > 0:
            if device.type == "cuda":
                y = kornia_unsharp(y, sigma=sharpen_sigma, amount=sharpen_amount)
            else:
                img_y = _to_image(y)
                blurred = cv2.GaussianBlur(img_y, (0, 0), sharpen_sigma)
                img_y = cv2.addWeighted(img_y, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
                y = _to_tensor(img_y)
        if size_tuple is not None:
            W, H = size_tuple
            y = F.interpolate(y, size=(H, W), mode="bicubic", align_corners=False)
        out_img = _to_image(y)
        _imsave(out_path, out_img, fmt=save_fmt, jpg_quality=jpg_quality, png_level=png_level)
        return out_path

    results = []
    with ThreadPoolExecutor(max_workers=max(1, io_workers)) as ex:
        futures = {ex.submit(_run_one, p): p for p in inputs}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                outp = fut.result()
                results.append((src, outp, None))
            except Exception as e:
                results.append((src, None, str(e)))
    return results

def main():
    ap = argparse.ArgumentParser(description="Extended optimized restoration/SR pipeline")
    ap.add_argument("--input-images", nargs="+", required=True, help="List of input image paths")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--denoise-weights", type=str, default=None, help="NAFNet-SIDD .pth")
    ap.add_argument("--denoise-onnx", type=str, default=None, help="NAFNet-SIDD .onnx")
    ap.add_argument("--denoise-width", type=int, default=32, help="Width for NAFNet SIDD (default 32)")
    ap.add_argument("--deblur-weights", type=str, default=None, help="NAFNet-REDS .pth")
    ap.add_argument("--deblur-onnx", type=str, default=None, help="NAFNet-REDS .onnx")
    ap.add_argument("--deblur-width", type=int, default=64, help="Width for NAFNet REDS (default 64)")
    ap.add_argument("--sr-weights", type=str, default=None, help="NAFSSR-L_4x .pth")
    ap.add_argument("--sr-onnx", type=str, default=None, help="NAFSSR .onnx")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--sharpen-amount", type=float, default=1.0)
    ap.add_argument("--sharpen-sigma", type=float, default=1.0)
    ap.add_argument("--final-size", type=str, default=None)
    ap.add_argument("--format", type=str, default="png", choices=["png","jpg","jpeg","webp"])
    ap.add_argument("--jpg-quality", type=int, default=90)
    ap.add_argument("--png-level", type=int, default=1)
    ap.add_argument("--io-workers", type=int, default=4)
    args = ap.parse_args()
    results = process_images(
        inputs=args.input_images,
        output_dir=args.output_dir,
        denoise_weights=args.denoise_weights,
        denoise_onnx=args.denoise_onnx,
        denoise_width=int(args.denoise_width),
        deblur_weights=args.deblur_weights,
        deblur_onnx=args.deblur_onnx,
        deblur_width=int(args.deblur_width),
        sr_weights=args.sr_weights,
        sr_onnx=args.sr_onnx,
        device_str=args.device,
        fp16=bool(args.fp16),
        tile=int(args.tile),
        overlap=int(args.overlap),
        sharpen_amount=float(args.sharpen_amount),
        sharpen_sigma=float(args.sharpen_sigma),
        final_size=args.final_size,
        save_fmt=args.format,
        jpg_quality=int(args.jpg_quality),
        png_level=int(args.png_level),
        io_workers=int(args.io_workers),
    )
    ok = sum(1 for _, outp, err in results if err is None and outp is not None)
    fail = [ (src, err) for src, _, err in results if err is not None ]
    print(f"Done. Success: {ok}, Failed: {len(fail)}")
    if fail:
        for src, err in fail[:10]:
            print(f" - {src}: {err}")

if __name__ == "__main__":
    main()
