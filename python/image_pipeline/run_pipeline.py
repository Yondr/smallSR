
##### (torch201) PS Y:\gemini\project\python> python -m image_pipeline.run_pipeline      


import sys
from pathlib import Path
# добавить корень проекта (два уровня вверх, если запускаешь внутри image_pipeline)
sys.path.append(str(Path(__file__).resolve().parent))
# или если запускаешь из image_pipeline папки и relative imports ищут родителя - добавь parent:
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
import numpy as np
import os
import glob
import cv2
from torchvision import transforms

# --- Local Imports ---
from .nafssr_arch import NAFNetSR
from nafnetlib.core import DeblurProcessor, DenoiseProcessor




def run_full_pipeline():
    # --- Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_dir = os.path.join(project_dir, "data", "ps3noize_input")
    output_dir = os.path.join(project_dir, "data", "ps3noize_out_ip")
    model_dir = os.path.join(project_dir, "weights")
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files in the input directory
    image_paths = glob.glob(os.path.join(input_dir, '*.[pP][nN][gG]')) + \
                  glob.glob(os.path.join(input_dir, '*.[jJ][pP][gG]')) + \
                  glob.glob(os.path.join(input_dir, '*.[jJ][pP][eE][gG]'))

    if not image_paths:
        print(f"Не знайдено зображень у папці: {input_dir}")
        return

    print(f"Знайдено {len(image_paths)} зображень для обробки.")

    # --- Initialize Models (do this once outside the loop) ---
    print("Ініціалізація моделей...")
    try:
        from nafnetlib.core import DeblurProcessor, DenoiseProcessor
    except ImportError:
        print("Please install nafnetlib: pip install nafnetlib")
        return
    denoise_processor = DenoiseProcessor(model_id='sidd_width64', model_dir=model_dir, device=device)
    deblur_processor = DeblurProcessor(model_id='reds_width64', model_dir=model_dir, device=device)
    
    # SR Model
    weights_path_sr = os.path.join(project_dir, "weights", "NAFSSR-L_4x.pth")
    sr_model = NAFNetSR(up_scale=4, width=128, num_blks=128, drop_path_rate=0.3, drop_out_rate=0., dual=False).to(device)
    state_dict_sr = torch.load(weights_path_sr, map_location=device)["params"]
    sr_model.load_state_dict(state_dict_sr, strict=False)
    sr_model.eval()
    print("Всі моделі завантажено!")

    # --- Process each image ---
    for i, input_path in enumerate(image_paths):
        print(f"\n--- Обробка зображення {i+1}/{len(image_paths)}: {os.path.basename(input_path)} ---")
        
        # --- 1. Load Image ---
        print("Крок 1: Завантаження...")
        input_image = Image.open(input_path).convert("RGB")

        # --- 2. Denoise ---
        print("Крок 2: Усунення шуму (SIDD)...")
        denoised_image = denoise_processor.process(input_image)

        # --- 3. Deblur ---
        print("Крок 3: Усунення розмиття (REDS)...")
        deblurred_image = deblur_processor.process(denoised_image)

        # --- 4. Super-Resolution (4x) ---
        print("Крок 4: Збільшення зображення (4x)...")
        transform_sr = transforms.ToTensor()
        input_tensor_sr = transform_sr(deblurred_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output_tensor_sr = sr_model(input_tensor_sr)
        
        output_image_pil = transforms.ToPILImage()(output_tensor_sr.squeeze(0).cpu().clamp(0, 1))

        # --- 5. Sharpen ---
        print("Крок 5: Збільшення різкості...")
        # Convert PIL image to OpenCV format
        output_image_cv = cv2.cvtColor(np.array(output_image_pil), cv2.COLOR_RGB2BGR)
        # Create a sharpening kernel
        sharpen_kernel = np.array([[-1, -1, -1],
                                   [-1,  9, -1],
                                   [-1, -1, -1]])
        # Apply the sharpening kernel
        sharpened_image_cv = cv2.filter2D(output_image_cv, -1, sharpen_kernel)
        # Convert back to PIL format
        final_image = Image.fromarray(cv2.cvtColor(sharpened_image_cv, cv2.COLOR_BGR2RGB))

        # --- 6. Save Final Image ---
        input_filename = os.path.basename(input_path)
        output_filename = f"{os.path.splitext(input_filename)[0]}_processed.png"
        output_path = os.path.join(output_dir, output_filename)
        
        final_image.save(output_path, quality=98)
        print(f"Фінальний результат ЗБЕРЕЖЕНО → {output_path}")

    print("\nВся обробка завершена!")

if __name__ == '__main__':
    run_full_pipeline()