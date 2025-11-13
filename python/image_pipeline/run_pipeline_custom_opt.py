#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized image pipeline for batch frame processing on laptop GPU.

Key features:
- torch.inference_mode() + AMP (FP16) + cudnn.benchmark
- Optional ONNX Runtime CUDA path (--onnx path/to/model.onnx)
- Tiled inference to fit memory (--tile, --overlap)
- Kornia unsharp mask on GPU (--sharpen-amount, --sharpen-sigma)
- Threaded I/O to keep GPU busy (--io-workers)
- Single rescale at the end only if explicitly requested (--final-size WxH)
- Optional micro-batching for small frames (--batch)
"""

import sys
import argparse
from pathlib import Path
import os
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import torch
import torch.nn.functional as F

try:
    import onnxruntime as ort  # optional
    _HAS_ORT = True
except Exception:
    _HAS_ORT = False

try:
    import kornia as K
except Exception:
    K = None

import cv2

# ---- Local model import (PyTorch) ----
# Expect project layout: project_root/image_pipeline/nafssr_arch.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
try:
    from image_pipeline.nafssr_arch import NAFNetSR  # type: ignore
except Exception:
    NAFNetSR = None  # handled later


def _imread_rgb(path: str) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _imsave(path: str, rgb: np.ndarray, fmt: str = "png", jpg_quality: int = 90, png_level: int = 1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt.lower() in ["jpg", "jpeg"]:
        cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])[1].tofile(path)
    elif fmt.lower() in ["webp"]:
        cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), int(jpg_quality)])[1].tofile(path)
    else:  # png
        cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_level)])[1].tofile(path)


def _to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    # HWC uint8 -> 1x3xHxW float32 [0,1]
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).contiguous().float().div_(255.0)
    return t


def _to_image(t: torch.Tensor) -> np.ndarray:
    # 1x3xHxW [0,1] -> HWC uint8 RGB
    t = t.detach().clamp_(0.0, 1.0)
    t = (t * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return t


def parse_size(s: str):
    # "1280x1024" -> (1280, 1024)
    if s is None:
        return None
    if "x" not in s.lower():
        raise ValueError("Size must be WxH, e.g., 1280x1024")
    w, h = s.lower().split("x")
    return (int(w), int(h))


def blend_window(h: int, w: int, overlap: int, device: torch.device):
    """Create a cosine blending window for stitching tiles."""
    y = torch.hann_window(h, periodic=False, device=device).unsqueeze(1)
    x = torch.hann_window(w, periodic=False, device=device).unsqueeze(0)
    win = (y * x)
    # Avoid too strong darkening in the middle
    win = win.clamp_min(1e-3)
    # Pad inner area to ~1
    win = win / win.max()
    # Reduce blending when overlap is small
    if overlap < 8:
        win = win.pow(0.5)
    return win


@torch.inference_mode()
def tiled_forward_torch(model, x, tile: int, overlap: int, use_amp: bool):
    """
    x: 1xCxHxW on cuda
    Returns model(x) with tiling, supports unknown scale by probing first tile.
    """
    device = x.device
    _, c, H, W = x.shape
    if tile <= 0 or tile >= max(H, W):
        # No tiling
        with torch.cuda.amp.autocast(enabled=use_amp):
            return model(x)

    # Compute grid
    stride = tile - overlap
    xs = list(range(0, W, stride))
    ys = list(range(0, H, stride))
    if xs and xs[-1] + tile > W:
        xs[-1] = max(0, W - tile)
    if ys and ys[-1] + tile > H:
        ys[-1] = max(0, H - tile)

    # Probe output size using first tile
    with torch.cuda.amp.autocast(enabled=use_amp):
        y0 = model(x[:, :, ys[0]:ys[0]+tile, xs[0]:xs[0]+tile])
    _, c2, Ht, Wt = y0.shape

    # Ratio between out/in for SR models
    scale_h = Ht / float(tile)
    scale_w = Wt / float(tile)
    outH = int(round(H * scale_h))
    outW = int(round(W * scale_w))

    out = torch.zeros((1, c2, outH, outW), dtype=y0.dtype, device=device)
    weight = torch.zeros_like(out)

    win = blend_window(int(tile*scale_h), int(tile*scale_w), int(overlap*scale_h), device=device)
    win = win.unsqueeze(0).unsqueeze(0)  # 1x1xhxw

    for yy in ys:
        for xx in xs:
            xs0, ys0 = xx, yy
            xs1, ys1 = xx + tile, yy + tile

            xtile = x[:, :, ys0:ys1, xs0:xs1]
            with torch.cuda.amp.autocast(enabled=use_amp):
                ytile = model(xtile)

            y_h, y_w = ytile.shape[-2:]
            # Target placement in output tensor
            y0o = int(round(ys0 * scale_h))
            x0o = int(round(xs0 * scale_w))
            y1o = y0o + y_h
            x1o = x0o + y_w

            wpatch = win
            if wpatch.shape[-2:] != (y_h, y_w):
                wpatch = F.interpolate(wpatch, size=(y_h, y_w), mode="bilinear", align_corners=False)

            out[:, :, y0o:y1o, x0o:x1o] += ytile * wpatch
            weight[:, :, y0o:y1o, x0o:x1o] += wpatch

    out = out / weight.clamp_min(1e-6)
    return out


def forward_onnx(sess, x_np: np.ndarray, tile: int, overlap: int):
    """
    x_np: 1x3xHxW float32/float16 in numpy
    Tiled ONNX inference using CUDA EP.
    """
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

    # Probe first tile for scale
    y0 = sess.run(["output"], {"input": x_np[:, :, ys[0]:ys[0]+tile, xs[0]:xs[0]+tile]})[0]
    _, c2, Ht, Wt = y0.shape
    scale_h = Ht / float(tile)
    scale_w = Wt / float(tile)
    outH = int(round(H * scale_h))
    outW = int(round(W * scale_w))
    out = np.zeros((1, c2, outH, outW), dtype=y0.dtype)
    weight = np.zeros_like(out, dtype=np.float32)

    # Precompute hann windows in numpy
    wy = np.hanning(int(tile*scale_h)).reshape(-1, 1)
    wx = np.hanning(int(tile*scale_w)).reshape(1, -1)
    w2 = (wy @ wx).astype(np.float32)
    w2 /= max(1e-6, w2.max())

    for yy in ys:
        for xx in xs:
            ytile = sess.run(["output"], {"input": x_np[:, :, yy:yy+tile, xx:xx+tile]})[0]
            y_h, y_w = ytile.shape[-2:]

            y0o = int(round(yy * scale_h))
            x0o = int(round(xx * scale_w))
            y1o = y0o + y_h
            x1o = x0o + y_w

            wpatch = w2
            if wpatch.shape != (y_h, y_w):
                wpatch = cv2.resize(wpatch, (y_w, y_h), interpolation=cv2.INTER_LINEAR)

            for cc in range(c2):
                out[0, cc, y0o:y1o, x0o:x1o] += ytile[0, cc] * wpatch
                weight[0, cc, y0o:y1o, x0o:x1o] += wpatch

    out = out / np.clip(weight, 1e-6, None)
    return out


def kornia_unsharp(t: torch.Tensor, sigma: float = 1.0, amount: float = 1.0):
    if K is None or amount <= 1e-6:
        return t
    return K.filters.unsharp_mask(t, kernel_size=(3, 3), sigma=(sigma, sigma), amount=amount)


def load_torch_model(weights: str, device: torch.device):
    if NAFNetSR is None:
        raise RuntimeError("NAFNetSR not found. Ensure image_pipeline/nafssr_arch.py is importable.")
    model = NAFNetSR()
    ckpt = torch.load(weights, map_location="cpu")
    # Common keys: 'params' or full state_dict
    state = ckpt.get("params", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)
    return model


def make_ort_session(path: str):
    if not _HAS_ORT:
        raise RuntimeError("onnxruntime not installed. pip install onnxruntime-gpu")
    providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1"})]
    sess = ort.InferenceSession(path, providers=providers)
    return sess


def process_images(
    inputs,
    output_dir: str,
    torch_weights: str = None,
    onnx_path: str = None,
    device_str: str = "cuda",
    fp16: bool = True,
    tile: int = 0,
    overlap: int = 16,
    batch: int = 1,
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

    use_onnx = onnx_path is not None
    model = None
    sess = None

    if use_onnx:
        sess = make_ort_session(onnx_path)
    else:
        assert torch_weights is not None, "Provide --weights for PyTorch path or --onnx for ONNX path"
        model = load_torch_model(torch_weights, device=device)
        torch.backends.cudnn.benchmark = True

    size_tuple = parse_size(final_size) if final_size else None

    def _run_one(img_path: str):
        name = Path(img_path).stem
        out_path = str(Path(output_dir) / f"{name}.{save_fmt}")

        img = _imread_rgb(img_path)
        t = _to_tensor(img)  # 1x3xHxW

        if use_onnx:
            x_np = t.numpy().astype(np.float16 if fp16 else np.float32)
            y_np = forward_onnx(sess, x_np, tile=tile, overlap=overlap)
            y = torch.from_numpy(y_np).to(device if device.type == "cuda" else "cpu")
        else:
            t = t.to(device, non_blocking=True)
            if fp16 and device.type == "cuda":
                t = t.half()
            y = tiled_forward_torch(model, t, tile=tile, overlap=overlap, use_amp=(fp16 and device.type == "cuda"))

        # Sharpen on GPU if possible
        if sharpen_amount > 0:
            if device.type == "cuda":
                y = kornia_unsharp(y, sigma=sharpen_sigma, amount=sharpen_amount)
            else:
                # Fallback CPU unsharp using cv2
                img_y = _to_image(y)
                blurred = cv2.GaussianBlur(img_y, (0, 0), sharpen_sigma)
                img_y = cv2.addWeighted(img_y, 1 + sharpen_amount, blurred, -sharpen_amount, 0)
                y = _to_tensor(img_y)

        # Optional single final rescale
        if size_tuple is not None:
            W, H = size_tuple
            y = F.interpolate(y, size=(H, W), mode="bicubic", align_corners=False)

        out_img = _to_image(y)
        _imsave(out_path, out_img, fmt=save_fmt, jpg_quality=jpg_quality, png_level=png_level)
        return out_path

    # Threaded I/O + compute loop
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
    ap = argparse.ArgumentParser(description="Optimized image processing pipeline")
    ap.add_argument("--input-images", nargs="+", required=True, help="List of input image paths")
    ap.add_argument("--output-dir", required=True, help="Directory for results")

    # Model selection
    ap.add_argument("--weights", type=str, default=None, help="PyTorch weights for NAFNetSR")
    ap.add_argument("--onnx", type=str, default=None, help="ONNX model path (enables ONNX Runtime)")

    # Performance knobs
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--fp16", action="store_true", help="Enable FP16/AMP when possible")
    ap.add_argument("--tile", type=int, default=0, help="Tile size for tiled inference (0 disables tiling)")
    ap.add_argument("--overlap", type=int, default=16, help="Overlap between tiles")
    ap.add_argument("--batch", type=int, default=1, help="Reserved; per-image pipeline by default")

    # Sharpen and post
    ap.add_argument("--sharpen-amount", type=float, default=1.0, help="Unsharp mask amount")
    ap.add_argument("--sharpen-sigma", type=float, default=1.0, help="Unsharp gaussian sigma")
    ap.add_argument("--final-size", type=str, default=None, help='Final resize WxH, e.g. "1280x1024"')

    # Saving
    ap.add_argument("--format", type=str, default="png", choices=["png", "jpg", "jpeg", "webp"], help="Output format")
    ap.add_argument("--jpg-quality", type=int, default=90, help="JPEG/WebP quality")
    ap.add_argument("--png-level", type=int, default=1, help="PNG compression level 0-9")

    # I/O
    ap.add_argument("--io-workers", type=int, default=4, help="Thread workers for I/O and per-file pipeline")

    args = ap.parse_args()

    results = process_images(
        inputs=args.input_images,
        output_dir=args.output_dir,
        torch_weights=args.weights,
        onnx_path=args.onnx,
        device_str=args.device,
        fp16=bool(args.fp16),
        tile=int(args.tile),
        overlap=int(args.overlap),
        batch=int(args.batch),
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
