import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train.train_student_sr import StudentSR  # если путь другой — поправь импорт

# Пути
CKPT_PATH = "../../weights/student_sr.pth"          # от папки python/optimize
ONNX_PATH = "../../weights/student_sr_1280x960.onnx"

# Фиксированное разрешение под наш пайплайн: 1280x960 (W x H)
H, W = 960, 1280   # NCHW -> (1, 3, 960, 1280)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1. Собираем модель и грузим веса
    model = StudentSR(num_blocks=8, ch=64).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)

    # чекпоинт, как мы делали раньше: {"model_state": ...}
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.eval()

    # 2. Дамми-вход для трассировки
    dummy = torch.randn(1, 3, H, W, device=device)

    # 3. Экспорт в ONNX
    torch.onnx.export(
        model,
        dummy,
        ONNX_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            # если хочешь строгий фиксированный размер — можно удалить dynamic_axes
            "input": {0: "N", 2: "H", 3: "W"},
            "output": {0: "N", 2: "H", 3: "W"},
        },
    )

    print(f"ONNX сохранён в: {ONNX_PATH}")


if __name__ == "__main__":
    main()
