
import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import os
import cv2
from torchvision import transforms

# --- Local Imports ---
# Ensure the local libraries can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))
from image_pipeline.nafssr_arch import NAFNetSR
try:
    from nafnetlib.core import DeblurProcessor, DenoiseProcessor
except ImportError:
    print("Error: nafnetlib is not installed. Please run 'pip install nafnetlib'")
    sys.exit(1)


def run_custom_pipeline(image_paths, output_dir_path):
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    project_dir = Path(__file__).resolve().parent.parent.parent
    model_dir = project_dir / "weights"
    output_dir = Path(output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        print(f"No images provided for processing.")
        return

    print(f"Found {len(image_paths)} images to process.")

    # --- Initialize Models (do this once outside the loop) ---
    print("Initializing models...")
    denoise_processor = DenoiseProcessor(model_id='sidd_width64', model_dir=str(model_dir), device=device)
    deblur_processor = DeblurProcessor(model_id='reds_width64', model_dir=str(model_dir), device=device)
    
    # SR Model
    weights_path_sr = model_dir / "NAFSSR-L_4x.pth"
    sr_model = NAFNetSR(up_scale=4, width=128, num_blks=128, drop_path_rate=0.3, drop_out_rate=0., dual=False).to(device)
    state_dict_sr = torch.load(str(weights_path_sr), map_location=device)["params"]
    sr_model.load_state_dict(state_dict_sr, strict=False)
    sr_model.eval()
    print("All models loaded!")

    # --- Process each image ---
    for i, input_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)}: {os.path.basename(input_path)} ---")
        
        try:
            # --- 1. Load Image ---
            print("Step 1: Loading...")
            input_image = Image.open(input_path).convert("RGB")

            # --- 2. Super-Resolution (4x) ---
            print("Step 2: Super-Resolution (4x)...")
            transform_sr = transforms.ToTensor()
            input_tensor_sr = transform_sr(input_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output_tensor_sr = sr_model(input_tensor_sr)
            upscaled_image = transforms.ToPILImage()(output_tensor_sr.squeeze(0).cpu().clamp(0, 1))

            # --- 3. Downscale ---
            print("Step 3: Downscaling to 1280x1024...")
            downscaled_image = upscaled_image.resize((1280, 1024), Image.Resampling.LANCZOS)

            # --- 4. Deblur ---
            print("Step 4: Deblurring (REDS)...")
            deblurred_image = deblur_processor.process(downscaled_image)

            # --- 5. Denoise ---
            print("Step 5: Denoising (SIDD)...")
            denoised_image = denoise_processor.process(deblurred_image)

            # --- 6. Sharpen ---
            print("Step 6: Sharpening...")
            output_image_cv = cv2.cvtColor(np.array(deblurred_image), cv2.COLOR_RGB2BGR)
            sharpen_kernel = np.array([[-1, -1, -1],
                                       [-1,  9, -1],
                                       [-1, -1, -1]])
            sharpened_image_cv = cv2.filter2D(output_image_cv, -1, sharpen_kernel)
            final_image = Image.fromarray(cv2.cvtColor(sharpened_image_cv, cv2.COLOR_BGR2RGB))

            # --- 6. Save Final Image ---
            input_filename = os.path.basename(input_path)
            output_filename = f"{os.path.splitext(input_filename)[0]}_processed.png"
            output_path = output_dir / output_filename
            
            final_image.save(str(output_path), "PNG", quality=98)
            print(f"Final result SAVED -> {output_path}")

        except Exception as e:
            print(f"Failed to process {os.path.basename(input_path)}. Error: {e}")

    print("\nAll processing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a full image processing pipeline on a directory of images.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to the directory containing input images.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='The directory where processed images will be saved.')

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    image_files = []
    if input_path.is_dir():
        # Add more extensions if needed
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        print(f"Searching for images in {input_path}...")
        for ext in supported_extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    elif input_path.is_file():
        image_files.append(input_path)

    if not image_files:
        print(f"No image files found in {args.input_dir}")
        sys.exit(1)

    run_custom_pipeline(image_files, args.output_dir)
