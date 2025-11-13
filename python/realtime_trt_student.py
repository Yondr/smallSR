import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "Y:/gemini/project/weights/student_sr_1280x960_fp16.engine"

# Target size for SR
OUT_W = 1280
OUT_H = 960


# -----------------------------
# TensorRT Loader
# -----------------------------
class TRTModule:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.INFO)

        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = self.context.get_binding_shape(i)

            size = np.prod(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({"host": host_mem, "device": device_mem, "shape": shape})
            else:
                self.outputs.append({"host": host_mem, "device": device_mem, "shape": shape})

    def infer(self, img_chw):
        inp = self.inputs[0]

        np.copyto(inp["host"], img_chw.ravel())
        cuda.memcpy_htod(inp["device"], inp["host"])

        self.context.execute_v2(self.bindings)

        out = self.outputs[0]
        cuda.memcpy_dtoh(out["host"], out["device"])

        out_img = out["host"].reshape(out["shape"])
        return out_img


# -----------------------------
# Real-time pipeline
# -----------------------------
def open_ps3_eye():
    for idx in [0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"PS3 Eye найдена на index {idx}")
            return cap
    raise RuntimeError("Камера PS3 Eye не найдена")


def main():
    trt_model = TRTModule(ENGINE_PATH)

    cap = open_ps3_eye()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    print("ESC = выход")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Нет кадров с камеры")
            break

        # апскейл bicubic → 1280x960
        up = cv2.resize(frame, (OUT_W, OUT_H), interpolation=cv2.INTER_CUBIC)

        # формат в CHW float32 [0..1]
        rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        chw = x.transpose(2, 0, 1)[None]  # NCHW

        # инференс TensorRT
        out = trt_model.infer(chw)

        # NCWH → HWC uint8
        out_img = (out[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        cv2.imshow("Student SR TensorRT (1280x960)", out_bgr)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
