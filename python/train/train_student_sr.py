#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# =========================
#  Dataset
# =========================

class SRPairDataset(Dataset):
    def __init__(self, raw_dir, hires_dir, patch_size=128, random_crop=True,
                 target_size=(1280, 1024)):
        """
        raw_dir   : папка с RAW кадрами
        hires_dir : папка с HIRES (teacher output)
        patch_size: размер патча; если <=0, берем полный кадр
        random_crop:
            True  - случайный кроп (train)
            False - центрированный кроп (val)
        target_size:
            (W, H) — рабочий размер, к которому приводим и raw, и hires.
            По умолчанию (1280, 1024).
        """
        self.raw_dir = Path(raw_dir)
        self.hires_dir = Path(hires_dir)
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.target_size = target_size  # (W, H)

        raw_names = {p.name for p in self.raw_dir.iterdir() if p.is_file()}
        hr_names = {p.name for p in self.hires_dir.iterdir() if p.is_file()}
        self.names = sorted(list(raw_names & hr_names))
        if not self.names:
            raise RuntimeError(
                f"No common files between raw ({self.raw_dir}) and hires ({self.hires_dir})"
            )

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def _crop_pair(self, lr_img, hr_img):
        """Оба изображения уже одного размера (после target_size)."""
        if self.patch_size is None or self.patch_size <= 0:
            return lr_img, hr_img

        w, h = hr_img.size
        ps = self.patch_size

        if w < ps or h < ps:
            # если кадр меньше патча — без кропа
            return lr_img, hr_img

        if self.random_crop:
            # TRAIN: случайный кроп
            x = np.random.randint(0, w - ps + 1)
            y = np.random.randint(0, h - ps + 1)
        else:
            # VAL: центрированный кроп
            x = (w - ps) // 2
            y = (h - ps) // 2

        hr_crop = hr_img.crop((x, y, x + ps, y + ps))
        lr_crop = lr_img.crop((x, y, x + ps, y + ps))
        return lr_crop, hr_crop

    def __getitem__(self, idx):
        name = self.names[idx]
        raw_path = self.raw_dir / name
        hr_path = self.hires_dir / name

        raw = Image.open(raw_path).convert("RGB")
        hr = Image.open(hr_path).convert("RGB")

        # приводим оба к target_size (W, H) = (1280, 1024)
        if self.target_size is not None:
            tw, th = self.target_size
            raw = raw.resize((tw, th), Image.BICUBIC)
            hr = hr.resize((tw, th), Image.BICUBIC)

        lr_patch, hr_patch = self._crop_pair(raw, hr)

        lr_t = self.to_tensor(lr_patch)   # [3,H,W], 0..1
        hr_t = self.to_tensor(hr_patch)

        return lr_t, hr_t



# =========================
#  Простая student-модель
# =========================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        return x + out


class StudentSR(nn.Module):
    """
    Лёгкая CNN: 3→64, несколько residual-блоков, 64→3.
    Масштаб = 1 (вход и выход одного размера).
    """
    def __init__(self, num_blocks=8, ch=64):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x):
        x0 = self.head(x)
        out = self.body(x0)
        out = self.tail(out)
        return out + x  # residual к входу


# =========================
#  Метрики и обучение
# =========================

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse.item() == 0:
        return torch.tensor(99.0, device=pred.device)
    return 10 * torch.log10(1.0 / mse)


def train(
    raw_dir,
    hires_dir,
    raw_val_dir=None,
    hires_val_dir=None,
    out_path="student_sr.pth",
    epochs=10,
    batch_size=8,
    lr=1e-4,
    patch_size=128,
    num_workers=4,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # --- train dataset ---
    train_dataset = SRPairDataset(
        raw_dir, hires_dir, patch_size=patch_size, random_crop=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- val dataset (опционально) ---
    val_loader = None
    if raw_val_dir is not None and hires_val_dir is not None:
        val_dataset = SRPairDataset(
            raw_val_dir, hires_val_dir,
            patch_size=patch_size,  # можно 0, если хочешь full-frame
            random_crop=False       # центрированный кроп
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print("Train size:", len(train_dataset), "pairs | Val size:", len(val_dataset))
    else:
        print("Dataset size:", len(train_dataset), "pairs (no validation set)")

    model = StudentSR(num_blocks=8, ch=64).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        # ---------- TRAIN ----------
        model.train()
        train_loss_sum = 0.0
        train_psnr_sum = 0.0
        n_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", unit="batch")
        for lr_img, hr_img in pbar:
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad()
            sr = model(lr_img)
            loss = criterion(sr, hr_img)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_psnr = psnr(sr.clamp(0.0, 1.0), hr_img)

            bs = lr_img.size(0)
            train_loss_sum += loss.item() * bs
            train_psnr_sum += batch_psnr.item() * bs
            n_train += bs

            pbar.set_postfix(loss=loss.item(), psnr=batch_psnr.item())

        train_loss = train_loss_sum / n_train
        train_psnr = train_psnr_sum / n_train

        # ---------- VALIDATION ----------
        val_loss = None
        val_psnr = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_psnr_sum = 0.0
            n_val = 0

            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]", unit="batch")
                for lr_img, hr_img in pbar_val:
                    lr_img = lr_img.to(device, non_blocking=True)
                    hr_img = hr_img.to(device, non_blocking=True)

                    sr = model(lr_img)
                    loss = criterion(sr, hr_img)
                    batch_psnr = psnr(sr.clamp(0.0, 1.0), hr_img)

                    bs = lr_img.size(0)
                    val_loss_sum += loss.item() * bs
                    val_psnr_sum += batch_psnr.item() * bs
                    n_val += bs

                    pbar_val.set_postfix(loss=loss.item(), psnr=batch_psnr.item())

            val_loss = val_loss_sum / n_val
            val_psnr = val_psnr_sum / n_val

        # ---------- LOG + SAVE ----------
        if val_loader is not None:
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_loss={train_loss:.6f}, train_PSNR={train_psnr:.2f} dB | "
                f"val_loss={val_loss:.6f}, val_PSNR={val_psnr:.2f} dB"
            )
        else:
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_loss={train_loss:.6f}, train_PSNR={train_psnr:.2f} dB"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_psnr": train_psnr,
                "val_loss": val_loss,
                "val_psnr": val_psnr,
            },
            out_path,
        )
        print("Saved:", out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Train student SR model on raw/hires pairs with optional validation set"
    )
    ap.add_argument("--raw-dir", type=str, required=True)
    ap.add_argument("--hires-dir", type=str, required=True)
    ap.add_argument("--raw-val-dir", type=str, default=None,
                    help="RAW validation directory (optional)")
    ap.add_argument("--hires-val-dir", type=str, default=None,
                    help="HIRES validation directory (optional)")
    ap.add_argument(
        "--out",
        type=str,
        default="student_sr.pth",
        help="Path to save checkpoint",
    )
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)

    args = ap.parse_args()

    train(
        raw_dir=args.raw_dir,
        hires_dir=args.hires_dir,
        raw_val_dir=args.raw_val_dir,
        hires_val_dir=args.hires_val_dir,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
    )
