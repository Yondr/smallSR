import torch
import cv2
import numpy as np
from train.train_student_sr import StudentSR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StudentSR(num_blocks=8, ch=64).to(device)
ckpt = torch.load("Y:/gemini/project/weights/student_sr.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

TARGET_W, TARGET_H = 1280, 1024

@torch.no_grad()
def run_student(frame_bgr: np.ndarray) -> np.ndarray:
    # ресайз до 1280x1024 перед моделью
    frame_bgr = cv2.resize(frame_bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_CUBIC)

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]  # NCHW

    x_t = torch.from_numpy(x).to(device)

    out = model(x_t).clamp(0, 1)
    out_np = (out[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    img = cv2.imread("Y:/gemini/project/data/test.png")
    out = run_student(img)
    print("Input shape:", img.shape)
    print("Output shape:", out.shape)  # должно быть (1024, 1280, 3)
    cv2.imwrite("Y:/gemini/project/data/test_student_1280x1024_ep20.png", out)
