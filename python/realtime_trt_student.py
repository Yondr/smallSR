import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

ENGINE_PATH = "Y:/gemini/project_clean/weights/student_sr_1280x960_fp16.engine"

# Target size for SR
OUT_W = 1280
OUT_H = 960


# -----------------------------
# TensorRT Loader
# -----------------------------
class TRTModule:
    def __init__(self, engine_path: str):
        logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # имена входа/выхода (TensorRT 10.x io_tensors API)
        self.input_name = None
        self.output_name = None

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Could not find input/output tensors in engine")

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))

    def infer(self, img_chw: np.ndarray) -> np.ndarray:
        """
        img_chw: (1, 3, H, W), float32, [0,1]
        """
        img_chw = np.ascontiguousarray(img_chw.astype(np.float32))

        engine_in_shape = self.engine.get_tensor_shape(self.input_name)
        if any(d == -1 for d in engine_in_shape):
            self.context.set_input_shape(self.input_name, img_chw.shape)

        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out_size = int(np.prod(out_shape))

        d_input = cuda.mem_alloc(img_chw.nbytes)
        out_host = np.empty(out_size, dtype=np.float32)
        d_output = cuda.mem_alloc(out_host.nbytes)

        self.context.set_tensor_address(self.input_name, int(d_input))
        self.context.set_tensor_address(self.output_name, int(d_output))

        cuda.memcpy_htod_async(d_input, img_chw, self.stream)

        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT execute_async_v3 failed")

        cuda.memcpy_dtoh_async(out_host, d_output, self.stream)
        self.stream.synchronize()

        out = out_host.reshape(out_shape)
        return out


# -----------------------------
# Camera helpers
# -----------------------------
def open_ps3_eye():
    # пробуем разные backend'ы
    for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF]:
        for idx in [0, 1, 2, 3]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"PS3 Eye найдена: index {idx}, backend {backend}")
                # пробуем MJPG
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 60)

                # выводим фактические параметры
                try:
                    print("Backend name:", cap.getBackendName())
                except Exception:
                    pass
                print("Width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print("FPS   :", cap.get(cv2.CAP_PROP_FPS))
                return cap

            cap.release()

    raise RuntimeError("Камера PS3 Eye не найдена ни на одном index/backend")


# -----------------------------
# Real-time pipeline
# -----------------------------
def main():
    trt_model = TRTModule(ENGINE_PATH)
    gpu_name = pycuda.autoinit.device.name()
    device_info = f"TensorRT | {gpu_name}"

    cap = open_ps3_eye()
    print("ESC = выход")

    frame_count = 0
    fps = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Нет кадров с камеры (ret=False или frame=None)")
            continue

        # upscaled RAW (bicubic)
        up = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)

        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        chw = x.transpose(2, 0, 1)[None]  # (1, 3, H, W)

        out = trt_model.infer(chw)        # (1, 3, H, W)

        out_img = (out[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        # FPS counter
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()

        # Display info
        cv2.putText(out_bgr, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(out_bgr, device_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Original 640x480", frame)
        cv2.imshow("Student SR TensorRT (1280x960)", out_bgr)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
