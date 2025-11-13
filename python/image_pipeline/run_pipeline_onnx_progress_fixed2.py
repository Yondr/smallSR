
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX-only restoration/SR pipeline with progress bars (fixed2).
Adds:
  - Per-stage dtype detection from model input (tensor(float) vs tensor(float16))
  - Auto-detect input/output tensor names (no hardcoded 'input'/'output')
  - Normalize model outputs to NCHW with batch (1,C,H,W)
  - Correct broadcasting of blending window (wpatch[None, None, ...])
Retains:
  - Absolute glob handling (glob.glob)
  - CUDAExecutionProvider with CPU fallback
  - Proper tiling for small images (no broadcast errors)
"""
import argparse, os, glob
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import onnxruntime as ort

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ---------------- utilities ----------------

def list_images(inputs, input_dir=None):
    files = []
    if input_dir:
        p = Path(input_dir)
        if not p.is_dir():
            raise FileNotFoundError(f"Input dir not found: {input_dir}")
        for ext in IMG_EXTS:
            files.extend(p.glob(f"*{ext}"))
    if inputs:
        for item in inputs:
            if any(ch in item for ch in "*?[]"):
                for m in glob.glob(item):
                    mp = Path(m)
                    if mp.is_dir():
                        for ext in IMG_EXTS:
                            files.extend(mp.glob(f"*{ext}"))
                    elif mp.suffix.lower() in IMG_EXTS and mp.exists():
                        files.append(mp)
            else:
                fp = Path(item)
                if fp.is_dir():
                    for ext in IMG_EXTS:
                        files.extend(fp.glob(f"*{ext}"))
                elif fp.suffix.lower() in IMG_EXTS and fp.exists():
                    files.append(fp)
    uniq, seen = [], set()
    for f in files:
        f = f.resolve()
        if f.suffix.lower() in IMG_EXTS and f.exists() and f not in seen:
            seen.add(f); uniq.append(f)
    return [str(f) for f in sorted(uniq)]

def imread_rgb(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None: raise RuntimeError(f"Failed to read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imsave(path: str, rgb: np.ndarray, fmt: str = "png", jpg_quality: int = 90, png_level: int = 1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    fmt = fmt.lower()
    if fmt in ("jpg","jpeg"):
        cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])[1].tofile(path)
    elif fmt == "webp":
        cv2.imencode(".webp", bgr, [int(cv2.IMWRITE_WEBP_QUALITY), int(jpg_quality)])[1].tofile(path)
    else:
        cv2.imencode(".png", bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), int(png_level)])[1].tofile(path)

def make_session(path: str):
    try:
        return ort.InferenceSession(path, providers=[("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": "1"})])
    except Exception:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

def get_io_names_and_dtype(sess: ort.InferenceSession):
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    in_name = inp.name
    out_name = out.name
    # dtype mapping
    in_type = inp.type  # e.g., 'tensor(float16)' or 'tensor(float)'
    if "float16" in in_type:
        in_dtype = np.float16
    else:
        in_dtype = np.float32
    return in_name, out_name, in_dtype

def ensure_nchw1(y: np.ndarray) -> np.ndarray:
    # Accepts (C,H,W), (1,C,H,W), (H,W,C), (1,H,W,C)
    if y.ndim == 4:
        n,c_or_h,h_or_w,w_or_c = y.shape
        # Heuristic: if second dim is channels <=4, assume NCHW
        if n == 1 and (y.shape[1] <= 4):
            return y
        # else NHWC -> transpose
        if n == 1 and y.shape[-1] <= 4:
            return y.transpose(0,3,1,2)
    elif y.ndim == 3:
        # (C,H,W) or (H,W,C)
        if y.shape[0] <= 4:  # C,H,W
            return y[None, ...]
        else:  # H,W,C
            return y.transpose(2,0,1)[None, ...]
    else:
        raise ValueError(f"Unexpected output shape: {y.shape}")
    return y  # fallback

def hann2d(h, w):
    if h <= 1 or w <= 1:
        return np.ones((h, w), dtype=np.float32)
    wy = np.hanning(h).reshape(-1, 1)
    wx = np.hanning(w).reshape(1, -1)
    w2 = (wy @ wx).astype(np.float32)
    m = float(w2.max()) if w2.size else 1.0
    if m > 0: w2 /= m
    return w2

def forward_onnx_tiled(sess, x_np: np.ndarray, tile: int, overlap: int, progress_cb=None):
    in_name, out_name, _ = get_io_names_and_dtype(sess)
    if tile <= 0:
        y = sess.run([out_name], {in_name: x_np})[0]
        return ensure_nchw1(y)

    _, _, H, W = x_np.shape
    tile_h = min(tile, H)
    tile_w = min(tile, W)
    stride_h = max(1, tile_h - overlap)
    stride_w = max(1, tile_w - overlap)

    xs = list(range(0, W, stride_w))
    ys = list(range(0, H, stride_h))
    if xs and xs[-1] + tile_w > W: xs[-1] = max(0, W - tile_w)
    if ys and ys[-1] + tile_h > H: ys[-1] = max(0, H - tile_h)

    h_eff0 = min(tile_h, H - ys[0])
    w_eff0 = min(tile_w,  W - xs[0])
    y0 = sess.run([out_name], {in_name: x_np[:, :, ys[0]:ys[0]+h_eff0, xs[0]:xs[0]+w_eff0]})[0]
    y0 = ensure_nchw1(y0)
    _, c2, Ht0, Wt0 = y0.shape

    scale_h = Ht0 / float(h_eff0)
    scale_w = Wt0 / float(w_eff0)
    outH = int(round(H * scale_h)); outW = int(round(W * scale_w))
    out    = np.zeros((1, c2, outH, outW), dtype=y0.dtype)
    weight = np.zeros_like(out, dtype=np.float32)
    base_win = hann2d(int(tile_h * scale_h), int(tile_w * scale_w))

    total_tiles = len(xs) * len(ys); done = 0
    for yy in ys:
        for xx in xs:
            h_eff = min(tile_h, H - yy)
            w_eff = min(tile_w,  W - xx)
            ytile = sess.run([out_name], {in_name: x_np[:, :, yy:yy+h_eff, xx:xx+w_eff]})[0]
            ytile = ensure_nchw1(ytile)
            _, _, y_h, y_w = ytile.shape

            y0o = int(round(yy * scale_h)); x0o = int(round(xx * scale_w))
            y1o = y0o + y_h; x1o = x0o + y_w

            wpatch = base_win
            if wpatch.shape != (y_h, y_w):
                wpatch = cv2.resize(wpatch, (y_w, y_h), interpolation=cv2.INTER_LINEAR)
            # Ensure proper broadcasting: (1,1,h,w)
            wpatch = wpatch.astype(ytile.dtype, copy=False)[None, None, ...]

            out[0, :, y0o:y1o, x0o:x1o] += (ytile * wpatch)[0]
            weight[0, :, y0o:y1o, x0o:x1o] += wpatch[0]

            done += 1
            if progress_cb:
                progress_cb(done, total_tiles)

    out = out / np.clip(weight, 1e-6, None)
    return out

# ---------------- pipeline ----------------

def process_one(img_path, out_dir, denoise_sess, deblur_sess, sr_sess,
                fp16_flag, tile, overlap, final_size, sharpen_amount, sharpen_sigma,
                fmt, jpg_quality, png_level):
    name = Path(img_path).stem
    out_path = str(Path(out_dir) / f"{name}.{fmt}")
    rgb = imread_rgb(img_path)
    base = rgb.astype(np.float32) / 255.0
    base = np.transpose(base, (2,0,1))[None, ...]  # 1x3xHxW

    def run_stage(sess, x_np, desc):
        if sess is None:
            return x_np
        in_name, out_name, stage_dtype = get_io_names_and_dtype(sess)
        # If user passed --fp16, prefer fp16, else match model; but always match model if mismatch
        x_cast = x_np.astype(stage_dtype, copy=False) if (x_np.dtype != stage_dtype) else x_np
        bar = tqdm(total=1 if tile<=0 else None, desc=f"{desc} tiles", leave=False, unit="tile")
        def cb(a,b):
            if bar.total is None: bar.total = b
            bar.n = a; bar.refresh()
        y = forward_onnx_tiled(sess, x_cast, tile, overlap, cb)
        bar.close()
        return y.astype(x_np.dtype, copy=False)  # keep pipeline dtype stable between stages

    # initial dtype based on --fp16
    x = base.astype(np.float16 if fp16_flag else np.float32, copy=False)

    x = run_stage(denoise_sess, x, "Denoise")
    x = run_stage(deblur_sess,  x, "Deblur")
    x = run_stage(sr_sess,      x, "SR")

    out = np.transpose(x[0], (1,2,0))
    out = np.clip(out, 0.0, 1.0)
    if final_size:
        W,H = final_size
        out = cv2.resize(out, (W,H), interpolation=cv2.INTER_CUBIC)
    if sharpen_amount > 0:
        blurred = cv2.GaussianBlur(out, (0,0), sharpen_sigma)
        out = cv2.addWeighted(out, 1+sharpen_amount, blurred, -sharpen_amount, 0)
    out_u8 = (np.clip(out,0,1)*255.0).round().astype(np.uint8)
    imsave(out_path, out_u8, fmt=fmt, jpg_quality=jpg_quality, png_level=png_level)
    return out_path

def parse_size(s):
    if not s: return None
    s = s.lower()
    if "x" not in s: raise ValueError("Size must be WxH")
    w,h = s.split("x"); return (int(w), int(h))

def main():
    ap = argparse.ArgumentParser(description="ONNX-only restoration/SR pipeline with per-stage progress bars (fixed2)")
    ap.add_argument("--input-images", nargs="*", help="Files or globs")
    ap.add_argument("--input-dir", type=str, default=None, help="Directory with images")
    ap.add_argument("--output-dir", required=True, help="Where to save results")
    ap.add_argument("--denoise-onnx", type=str, default=None)
    ap.add_argument("--deblur-onnx",  type=str, default=None)
    ap.add_argument("--sr-onnx",      type=str, default=None)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--final-size", type=str, default=None)
    ap.add_argument("--sharpen-amount", type=float, default=1.0)
    ap.add_argument("--sharpen-sigma", type=float, default=1.0)
    ap.add_argument("--format", type=str, default="png", choices=["png","jpg","jpeg","webp"])
    ap.add_argument("--jpg-quality", type=int, default=90)
    ap.add_argument("--png-level", type=int, default=1)
    args = ap.parse_args()

    inputs = list_images(args.input_images, args.input_dir)
    if not inputs:
        raise SystemExit("No input images found. Use --input-dir or --input-images with files/globs.")
    os.makedirs(args.output_dir, exist_ok=True)

    denoise_sess = make_session(args.denoise_onnx) if args.denoise_onnx else None
    deblur_sess  = make_session(args.deblur_onnx)  if args.deblur_onnx  else None
    sr_sess      = make_session(args.sr_onnx)      if args.sr_onnx      else None

    final_size = parse_size(args.final_size) if args.final_size else None

    with tqdm(total=len(inputs), desc="Images", unit="img") as bar:
        for img in inputs:
            try:
                process_one(img, args.output_dir,
                            denoise_sess, deblur_sess, sr_sess,
                            fp16_flag=args.fp16, tile=args.tile, overlap=args.overlap,
                            final_size=final_size,
                            sharpen_amount=args.sharpen_amount, sharpen_sigma=args.sharpen_sigma,
                            fmt=args.format, jpg_quality=args.jpg_quality, png_level=args.png_level)
            except Exception as e:
                tqdm.write(f"Failed: {img} :: {e}")
            bar.update(1)

if __name__ == "__main__":
    main()
