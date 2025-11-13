import sys
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np
import os
import cv2
from torchvision import transforms
import kornia as K

# --- Local Imports ---
# Ensure the local libraries can be found
sys.path.append(str(Path(__file__).resolve().parent.parent))
from image_pipeline.nafssr_arch import NAFNetSR
# The nafnetlib processors are no longer used in this simplified pipeline
# try:
#     from nafnetlib.core import DeblurProcessor, DenoiseProcessor
# except ImportError:
#     print("Error: nafnetlib is not installed. Please run 'pip install nafnetlib'")
#     sys.exit(1)


def run_simplified_pipeline(image_paths, output_dir_path):
    # --- Configuration & Optimizations ---
    torch.backends.cudnn.benchmark = True
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
    print("Initializing models for FP16 inference...")
    
    # SR Model
    weights_path_sr = model_dir / "NAFSSR-L_4x.pth"
    sr_model = NAFNetSR(up_scale=4, width=128, num_blks=128, drop_path_rate=0.3, drop_out_rate=0., dual=False)
    # Load weights with weights_only=True for security
    state_dict_sr = torch.load(str(weights_path_sr), map_location=device, weights_only=True)["params"]
    sr_model.load_state_dict(state_dict_sr, strict=False)
    sr_model.eval().half().to(device) # Move to GPU and set to FP16
    print("SR model loaded!")

    transform_to_tensor = transforms.ToTensor()
    
    # --- Process each image ---
    for i, input_path in enumerate(image_paths):
        print(f"\n--- Processing image {i+1}/{len(image_paths)}: {os.path.basename(input_path)} ---")
        
        try:
            # --- 1. Load Image and convert to tensor ---
            print("Step 1: Loading and preparing tensor...")
            input_image = Image.open(input_path).convert("RGB")
            # Move tensor to GPU and convert to FP16
            input_tensor = transform_to_tensor(input_image).unsqueeze(0).half().to(device, non_blocking=True)

            # --- 2. Run SR and Sharpening on GPU ---
            print("Step 2: Running Super-Resolution and Sharpening with AMP...")
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda"):
                # Super-Resolution
                upscaled_tensor = sr_model(input_tensor)
                
                # Clamp values to [0, 1] range before sharpening
                upscaled_tensor = upscaled_tensor.clamp(0, 1)
                
                # Sharpen using Kornia
                sharpened_tensor = K.filters.unsharp_mask(upscaled_tensor, kernel_size=(3, 3), sigma=(1.0, 1.0), amount=1.0)

            # --- 3. Save Final Image ---
            print("Step 3: Saving image...")
            # Convert final tensor to PIL image for saving
            # We need to move tensor to CPU and convert from FP16 to Float
            final_image = transforms.ToPILImage()(sharpened_tensor.squeeze(0).float().cpu())
            
            input_filename = os.path.basename(input_path)
            output_filename = f"{os.path.splitext(input_filename)[0]}_processed.png"
            output_path = output_dir / output_filename
            
            final_image.save(str(output_path), "PNG")
            print(f"Final result SAVED -> {output_path}")

        except Exception as e:
            print(f"Failed to process {os.path.basename(input_path)}. Error: {e}")

    print("\nAll processing complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a simplified, accelerated image processing pipeline.')
    parser.add_argument('--input-images', nargs='+', required=True,
                        help='A list of paths to the input images.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='The directory where processed images will be saved.')
    
    args = parser.parse_args()
    
    run_simplified_pipeline(args.input_images, args.output_dir)
