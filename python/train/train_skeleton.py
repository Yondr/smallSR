#Це механізм для підготовки і подачі даних у модель.

#Автоматично вирізає патчі, робить аугментації, конвертує в тензори.
#Важлива ланка між твоїм LR-HR пайплайном і тренуванням NAFNetSR.


from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class SRPairsDataset(Dataset):
    def __init__(self, root_dir, scale=4, patch_size=256):
        self.lr_dir = os.path.join(root_dir, 'LR')
        self.hr_dir = os.path.join(root_dir, 'HR')
        self.names = sorted(os.listdir(self.hr_dir))
        self.to_tensor = transforms.ToTensor()
        self.scale = scale
        self.patch_size = patch_size

    def __len__(self):
        return len(self.names)

    def random_crop(self, lr, hr):
        # lr, hr: PIL Images
        lw, lh = lr.size
        pw = self.patch_size // self.scale
        px = random.randint(0, max(0, lw - pw))
        py = random.randint(0, max(0, lh - pw))
        lr_patch = lr.crop((px, py, px + pw, py + pw))
        hr_patch = hr.crop((px * self.scale, py * self.scale,
                            (px + pw) * self.scale, (py + pw) * self.scale))
        return lr_patch, hr_patch

    def __getitem__(self, idx):
        name = self.names[idx]
        lr = Image.open(os.path.join(self.lr_dir, name)).convert('RGB')
        hr = Image.open(os.path.join(self.hr_dir, name)).convert('RGB')
        lr, hr = self.random_crop(lr, hr)
        # optional augment flips/rotations
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            lr = lr.transpose(Image.FLIP_TOP_BOTTOM)
            hr = hr.transpose(Image.FLIP_TOP_BOTTOM)
        return self.to_tensor(lr), self.to_tensor(hr)
