import os
import subprocess

# Paths
ONNX_PATH = "../../weights/student_sr_1280x960.onnx"
ENGINE_PATH = "../../weights/student_sr_1280x960_fp16.engine"
TRTEXEC_PATH = "Y:/gemini/TensorRT-10.12.0.36/bin/trtexec.exe"

# Workspace size in GB
WORKSPACE_GB = 4

def main():
    print("Converting ONNX to TensorRT engine...")

    if not os.path.exists(TRTEXEC_PATH):
        print(f"Error: trtexec not found at {TRTEXEC_PATH}")
        print("Please specify the correct path to trtexec.exe")
        return

    if not os.path.exists(ONNX_PATH):
        print(f"Error: ONNX model not found at {ONNX_PATH}")
        return

    # trtexec command
    command = [
        TRTEXEC_PATH,
        f"--onnx={ONNX_PATH}",
        f"--saveEngine={ENGINE_PATH}",
        "--fp16",
        f"--memPoolSize=workspace:{WORKSPACE_GB * 1024}",
        "--minShapes=input:1x3x960x1280",
        "--optShapes=input:1x3x960x1280",
        "--maxShapes=input:1x3x960x1280",
        "--verbose"
    ]

    print("Running command:")
    print(" ".join(command))

    # Run the command
    try:
        subprocess.run(command, check=True)
        print(f"\nTensorRT engine saved to: {ENGINE_PATH}")
    except subprocess.CalledProcessError as e:
        print(f"\nError during conversion: {e}")
    except FileNotFoundError:
        print(f"\nError: Could not find {TRTEXEC_PATH}. Please ensure it is installed and the path is correct.")


if __name__ == "__main__":
    main()
