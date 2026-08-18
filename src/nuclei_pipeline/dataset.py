"""PyTorch Dataset for the real nuclei train/val/test split (replaces the
synthetic generator used in Lab 4 — same tensor shapes/contract so the
U-Net and training loop are unchanged)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .data_prep import DatasetPaths


class NucleiDataset(Dataset):
    """Loads (grayscale image, binary mask) pairs for one split.

    Images and masks are already 256x256 in this dataset; grayscale
    conversion + [0,1] scaling happens here for the image, and the mask is
    thresholded at 127 to guarantee a strict {0,1} target regardless of any
    PNG compression artefacts.
    """

    def __init__(self, root: str | Path, split: str, augment: bool = False):
        self.paths = DatasetPaths(Path(root))
        self.split = split
        self.augment = augment
        self.ids = sorted(p.stem for p in self.paths.images_dir(split).glob("*.png"))

    def __len__(self) -> int:
        return len(self.ids)

    def _load(self, iid: str) -> tuple[np.ndarray, np.ndarray]:
        img = Image.open(self.paths.images_dir(self.split) / f"{iid}.png").convert("L")
        msk = Image.open(self.paths.masks_dir(self.split) / f"{iid}.png").convert("L")
        img = np.asarray(img, dtype=np.float32) / 255.0
        msk = (np.asarray(msk, dtype=np.float32) > 127).astype(np.float32)
        return img, msk

    def __getitem__(self, idx: int):
        iid = self.ids[idx]
        img, msk = self._load(iid)

        if self.augment:
            if np.random.rand() < 0.5:
                img, msk = np.fliplr(img).copy(), np.fliplr(msk).copy()
            if np.random.rand() < 0.5:
                img, msk = np.flipud(img).copy(), np.flipud(msk).copy()
            k = np.random.randint(4)
            img, msk = np.rot90(img, k).copy(), np.rot90(msk, k).copy()

        img_t = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        msk_t = torch.from_numpy(msk).unsqueeze(0)
        return img_t, msk_t

    def image_id(self, idx: int) -> str:
        return self.ids[idx]
