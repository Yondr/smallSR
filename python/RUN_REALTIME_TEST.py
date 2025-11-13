import torch
import cv2
import numpy as np
from train.train_student_sr import StudentSR

# -------------------------------
#        Load model
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = StudentSR(num_blocks=8, ch=64).to(device)
ckpt = torch.load("Y:/gemini/project/weights/student_sr.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

SCALE = 2  # 640x480 -> 1280x960


@torch.no_grad()
def run_student(frame_bgr: np.ndarray) -> np.ndarray:
    # 1) апскейлим оригинальный кадр x2 (корректный 4:3 → 4:3)
    h, w = frame_bgr.shape[:2]
    target_w, target_h = w * SCALE, h * SCALE

    frame_bgr_up = cv2.resize(
        frame_bgr,
        (target_w, target_h),
        interpolation=cv2.INTER_CUBIC
    )

    # 2) прогоним через студента как реставратор
    rgb = cv2.cvtColor(frame_bgr_up, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = x.transpose(2, 0, 1)[None]  # NCHW

    x_t = torch.from_numpy(x).to(device)
    out = model(x_t).clamp(0, 1)

    out_np = (out[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


# -------------------------------
#      Real-Time Pipeline
# -------------------------------
def open_ps3_eye():
    # пробуем несколько индексов камеры
    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"PS3 Eye найдена на index {idx}")
            return cap
    raise RuntimeError("PS3 Eye не найдена. Подключи камеру и попробуй снова.")


def main():
    cap = open_ps3_eye()

    # выставляем режим камеры 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    print("Нажми ESC для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Камера перестала отдавать кадры.")
            break

        out = run_student(frame)  # 1280x960

        cv2.imshow("PS3 Eye → Student SR x2 (1280x960)", out)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
