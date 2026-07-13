import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from PIL import Image


class SyntheticBrainDataset(Dataset):
    def __init__(self, size=200, img_size=256, seed=42):
        self.size = size
        self.img_size = img_size
        self.seed = seed

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed + idx)

        s = self.img_size
        img = rng.uniform(0.0, 0.35, (s, s)).astype(np.float32)
        cx, cy = rng.integers(s // 4, 3 * s // 4, size=2)
        rx, ry = rng.integers(s // 5, s // 3, size=2)
        y_grid, x_grid = np.ogrid[:s, :s]
        brain_mask = ((x_grid - cx) ** 2 / rx**2 + (y_grid - cy) ** 2 / ry**2) < 1
        img[brain_mask] += rng.uniform(0.3, 0.7, brain_mask.sum())

        tx, ty = cx + rng.integers(-rx // 3, rx // 3), cy + rng.integers(-ry // 3, ry // 3)
        tr, ts = rng.integers(8, 30), rng.uniform(0.6, 1.0)
        tumor_mask = ((x_grid - tx) ** 2 + (y_grid - ty) ** 2) < tr**2
        tumor_mask &= brain_mask
        img[tumor_mask] = np.clip(img[tumor_mask] * ts + rng.uniform(0.4, 0.8), 0, 1)

        img = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)

        image = torch.tensor(img).unsqueeze(0)
        mask = torch.tensor(tumor_mask.astype(np.int64))
        return image, mask


class KaggleBrainMRIDataset(Dataset):
    def __init__(self, root: str, img_size: int = 256, augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        self.samples = []

        for patient_dir in sorted(Path(root).iterdir()):
            if not patient_dir.is_dir():
                continue
            masks = sorted(patient_dir.glob("*_mask.tif"))
            for mask_path in masks:
                img_path = Path(str(mask_path).replace("_mask.tif", ".tif"))
                if img_path.exists():
                    self.samples.append((img_path, mask_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No image/mask pairs found under '{root}'. "
                "Download from: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation"
            )

    def __len__(self):
        return len(self.samples)

    def _augment(self, image, mask):
        mask = mask.unsqueeze(0)
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        angle = random.uniform(-20, 20)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)
        return image, mask.squeeze(0)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = np.array(
            Image.open(img_path).convert("L").resize((self.img_size, self.img_size), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        msk = np.array(
            Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST),
            dtype=np.int64,
        )
        msk = (msk > 127).astype(np.int64)

        image = TF.to_tensor(img.copy())
        mask = torch.from_numpy(msk)

        if self.augment:
            image, mask = self._augment(image, mask)

        return image, mask


def get_dataloaders(dataset_type="synthetic", root=None, batch_size=8, val_split=0.2, num_workers=0, img_size=256):
    if dataset_type == "synthetic":
        full_ds = SyntheticBrainDataset(size=500, img_size=img_size)
    elif dataset_type == "kaggle":
        full_ds = KaggleBrainMRIDataset(root=root, img_size=img_size, augment=True)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
