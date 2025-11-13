
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_metrics(log_path):
    """
    Reads a training log CSV file and generates plots for loss and PSNR.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        print(f"Error: Log file not found at '{log_path}'")
        return

    df = pd.read_csv(log_path)

    output_dir = log_path.parent
    
    # --- Plot 1: Loss vs. Epoch ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = output_dir / 'loss_plot.png'
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to: {loss_plot_path}")

    # --- Plot 2: PSNR vs. Epoch ---
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_psnr'], label='Train PSNR', marker='o')
    if 'val_psnr' in df.columns and df['val_psnr'].notna().any():
        plt.plot(df['epoch'], df['val_psnr'], label='Validation PSNR', marker='o')
    plt.title('PSNR vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)
    psnr_plot_path = output_dir / 'psnr_plot.png'
    plt.savefig(psnr_plot_path)
    print(f"Saved PSNR plot to: {psnr_plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training metrics from a log file.')
    parser.add_argument('--log-path', type=str, required=True,
                        help='Path to the training_log.csv file.')
    
    args = parser.parse_args()
    plot_metrics(args.log_path)
