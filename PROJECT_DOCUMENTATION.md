# Project Documentation: Real-time Super-Resolution Pipeline

## 1. Project Overview

The primary goal of this project is to develop a lightweight, real-time image enhancement model capable of running on resource-constrained hardware. The model takes raw, potentially noisy and blurry video frames as input and produces a clean, high-quality output.

To achieve this, we employed a **knowledge distillation** strategy, often referred to as a **"Teacher-Student"** approach. We first built a powerful but slow "Teacher" pipeline using multiple large, state-of-the-art models. Then, we trained a small, fast "Student" model to mimic the output of this powerful teacher.

---

## 2. The Step-by-Step Pipeline

The project is divided into five main stages:

### Step 1: Data Acquisition
- **Action:** Raw video frames are captured from a hardware source (a PS3 camera in this case) using the `python/capture_ps3_camera.py` script.
- **Output:** A directory of raw, low-quality images (e.g., `ps3_captures`).

### Step 2: The "Teacher" Pipeline (Offline Image Enhancement)
- **Action:** The raw images are processed by a sophisticated, multi-stage enhancement pipeline, run by the `python/image_pipeline/run_pipeline_custom_V2_mtread.py` script. This script is slow and computationally expensive, but produces very high-quality results.
- **The stages inside the teacher pipeline are:**
    1.  **4x Super-Resolution:** The input image is first upscaled by 4x using a **NAFSSR** model (`NAFSSR-L_4x.pth`).
    2.  **Downscaling:** The upscaled image is then downscaled to the final target resolution (1280x1024). This process helps to consolidate details and remove artifacts.
    3.  **Deblurring:** Motion blur is removed from the image using a **NAFNet** model pre-trained on the REDS dataset (`NAFNet-REDS-width64.pth`).
    4.  **Denoising:** Image sensor noise is removed using another **NAFNet** model pre-trained on the SIDD dataset (`NAFNet-SIDD-width64.pth`).
    5.  **Sharpening:** A final sharpening filter is applied to enhance fine details.
- **Output:** A directory of high-quality "teacher" images (e.g., `ps3_out_multithread`).

### Step 3: Dataset Preparation
- **Action:** We prepare the data for training the student model. This involves two sub-steps:
    1.  **Creating Corresponding Pairs:** We ensure that for every high-quality "teacher" image, we have the original corresponding "raw" image.
    2.  **Train/Validation Split:** We split this paired dataset into a training set (90%) and a validation set (10%). The validation set is crucial for evaluating the model's performance on unseen data and preventing overfitting. The validation files were moved to separate `_VALIDATION` folders to ensure they are not used during training.
- **Output:** Separate training and validation folders for both the raw (LR) and teacher (HR) images.

### Step 4: "Student" Model Training
- **Action:** The `python/train/train_student_sr.py` script is used to train our lightweight "student" model.
- **Process:**
    - The student model (`StudentSR`) takes a raw image from the training set as input.
    - It tries to generate an output image that is as close as possible to the corresponding high-quality "teacher" image.
    - The difference between the student's output and the teacher's output is measured by an **L1 Loss** function. The model's weights are adjusted through backpropagation to minimize this loss.
- **Output:** A single, lightweight trained model file: `weights/student_sr.pth`.

### Step 5: Inference (Real-time Testing)
- **Action:** The trained `student_sr.pth` model is used for real-time inference on new images or video streams. This is done using the `python/RUN_STUDENT_TEST.py` and `python/RUN_REALTIME_TEST.py` scripts.
- **Process:** Because the student model is small and fast, it can perform the entire enhancement task in a single pass, making it suitable for real-time applications.
- **Output:** A clean, enhanced video stream or output image.

---

## 3. Rationale: Why We Chose This Approach

- **Teacher-Student Model:**
    - **Advantage:** This is the core strength of the project. It allows us to combine the **quality** of a large, complex model (the teacher) with the **speed** of a small, efficient model (the student). We get the best of both worlds.
- **NAFNet Architecture:**
    - **Why:** NAFNet is a highly-regarded, state-of-the-art architecture for image restoration. By using pre-trained NAFNet models for deblurring and denoising, we leveraged existing research to build a very powerful teacher pipeline without having to train these large models from scratch.
- **Lightweight Student Model (`StudentSR`):**
    - **Why:** The student model uses a simple and effective ResNet-style architecture. This design is well-understood, easy to implement, and, most importantly, computationally cheap, which is essential for achieving real-time performance.
- **L1 Loss:**
    - **Why:** In image-to-image translation tasks, L1 loss often produces visually sharper and more pleasing results compared to L2 (MSE) loss, which can sometimes lead to blurry outputs.

---

## 4. Advantages and Disadvantages

### Advantages
- **High Performance at Low Cost:** The final student model is fast and efficient, capable of running in real-time.
- **Excellent Image Quality:** The student model learns to replicate the output of a very powerful teacher pipeline, resulting in high-quality image enhancement.
- **Modularity:** The teacher pipeline is modular. We can swap out or improve any stage (e.g., use a better deblurring model) and simply retrain the student to learn from the new, improved teacher.

### Disadvantages
- **Complexity:** The overall workflow is complex, requiring careful data management and multiple processing stages.
- **Performance is Capped by the Teacher:** The student can never be better than the teacher. Any artifacts or flaws in the teacher's output will also be learned by the student.
- **Training Dependency:** The project requires a significant amount of training data (LR/HR pairs) and time to train the student model effectively.

---

## 5. Training Metrics and Visualization

To evaluate the performance of the student model during training, we track two key metrics:

-   **L1 Loss:** The Mean Absolute Error between the student's output and the teacher's output. Lower is better.
-   **PSNR (Peak Signal-to-Noise Ratio):** A standard metric for image quality. Higher is better.

### Generating Metrics
The `train_student_sr.py` script has been updated to log these metrics for both the training and validation datasets after each epoch. The metrics are saved to a CSV file, specified by the `--log-path` argument (e.g., `training_log.csv`).

### Visualizing Metrics
A new script, `python/plot_metrics.py`, has been added to visualize the training progress.

-   **How to run:**
    ```bash
    python python/plot_metrics.py --log-path path/to/your/training_log.csv
    ```
-   **Output:** This script reads the CSV log file and generates two plots:
    -   `loss_plot.png`: Shows the training and validation loss over epochs.
    -   `psnr_plot.png`: Shows the training and validation PSNR over epochs.

These plots are saved in the same directory as the log file and are essential for analyzing the model's learning progress and for use in presentations. The `training_logs/` directory in the repository contains an example of these output files.