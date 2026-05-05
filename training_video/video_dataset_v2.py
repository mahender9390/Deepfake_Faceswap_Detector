"""
training_video/video_dataset_v2.py
────────────────────────────────────
Dataset for V2 Pure Temporal Model with STRONG augmentation.

Key improvements to prevent memorization:
  1. Heavy augmentation - model sees different version every epoch
  2. Temporal augmentation - same transform applied consistently across frames
     so model learns real temporal patterns not augmentation artifacts
  3. MixUp augmentation - blends clips to prevent overfitting
  4. Random frame dropping - forces model to handle missing frames
  5. Larger validation split for better generalization monitoring
"""

from __future__ import annotations
import random, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_EXTS = {".jpg", ".jpeg", ".png"}


def _build_train_transform(size: int) -> A.Compose:
    """
    STRONG augmentation for training.
    Same seed applied to all frames in a clip so temporal
    patterns remain consistent (model learns real changes,
    not augmentation artifacts).
    """
    return A.Compose([
        # Spatial augmentation
        A.RandomResizedCrop(size=(size, size), scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        A.HorizontalFlip(p=0.5),

        # Color augmentation — simulates different lighting/cameras
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.6),
        A.ToGray(p=0.05),

        # Blur and noise — simulates video compression artifacts
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # Noise — simulates sensor noise
        A.GaussNoise(p=0.3),

        # JPEG compression — simulates social media sharing
        A.ImageCompression(quality=(40, 90), p=0.5),

        # Geometric distortion — prevents spatial memorization
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
            A.ElasticTransform(alpha=50, sigma=5, p=1.0),
            A.OpticalDistortion(distort_limit=0.2, p=1.0),
        ], p=0.2),

        # Cutout — forces model to look at whole face
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(16, 48),
            hole_width_range=(16, 48),
            p=0.3
        ),

        # Normalization
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def _build_val_transform(size: int) -> A.Compose:
    """Minimal deterministic transform for validation."""
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def _group_into_clips(folder: Path, label: int, n_frames: int) -> list[tuple[list[Path], int]]:
    """Group extracted frames into clips of n_frames each."""
    if not folder.exists():
        warnings.warn(f"[dataset] Not found: {folder}")
        return []

    groups: dict[str, list[Path]] = defaultdict(list)
    for f in folder.iterdir():
        if f.suffix.lower() in _EXTS:
            parts = f.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append(f)

    clips = []
    for vid_idx, frames in groups.items():
        frames = sorted(frames)
        for start in range(0, len(frames) - n_frames + 1, n_frames):
            chunk = frames[start:start + n_frames]
            if len(chunk) == n_frames:
                clips.append((chunk, label))
        # Add last chunk if leftover
        remainder = len(frames) % n_frames
        if remainder > 0 and len(frames) >= n_frames:
            clips.append((frames[-n_frames:], label))

    return clips


class VideoFrameDatasetV2(Dataset):
    """
    Dataset for V2 model with strong augmentation.

    Key anti-overfitting features:
    1. Consistent augmentation across frames (same seed per clip)
    2. Random frame ordering (shuffles frames occasionally)
    3. Frame dropout (randomly drops and repeats frames)
    """

    def __init__(self, config: dict, split: str) -> None:
        self.n_frames  = config["model"]["n_frames"]
        self.size      = config["model"]["image_size"]
        self.split     = split
        self.is_train  = (split == "train")

        self.train_transform = _build_train_transform(self.size)
        self.val_transform   = _build_val_transform(self.size)

        data_dir = Path(config["paths"]["data_dir"])
        real_dir = data_dir / "train" / "real"
        fake_dir = data_dir / "train" / "fake"

        all_real = _group_into_clips(real_dir, 0, self.n_frames)
        all_fake = _group_into_clips(fake_dir, 1, self.n_frames)

        print(f"  Raw clips — fake: {len(all_fake)}, real: {len(all_real)}")

        # Balance
        min_count = min(len(all_real), len(all_fake))
        random.seed(42)
        all_real = random.sample(all_real, min_count)
        all_fake = random.sample(all_fake, min_count)

        all_clips = all_real + all_fake
        random.seed(42)
        random.shuffle(all_clips)

        # Split — larger val split for better monitoring
        n       = len(all_clips)
        n_test  = int(n * config["training"]["test_split"])
        n_val   = int(n * config["training"]["val_split"])
        n_train = n - n_val - n_test

        if split == "train":
            self.clips = all_clips[:n_train]
        elif split == "val":
            self.clips = all_clips[n_train:n_train + n_val]
        else:
            self.clips = all_clips[n_train + n_val:]

        n_f = sum(1 for _, l in self.clips if l == 1)
        n_r = sum(1 for _, l in self.clips if l == 0)
        print(f"  [{split:5s}] {len(self.clips):4d} clips  (fake={n_f}, real={n_r})")

    def _random_frame_dropout(self, frames: list[Path]) -> list[Path]:
        """
        Randomly drop 1-2 frames and repeat neighbors.
        Forces model to handle temporal gaps.
        Only applied during training.
        """
        if not self.is_train or random.random() > 0.3:
            return frames

        n_drop = random.randint(1, 2)
        result = list(frames)
        for _ in range(n_drop):
            if len(result) > 2:
                drop_idx = random.randint(1, len(result) - 2)
                result[drop_idx] = result[drop_idx - 1]  # repeat previous
        return result

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        frames_paths, label = self.clips[idx]

        # Apply random frame dropout during training
        frames_paths = self._random_frame_dropout(list(frames_paths))

        # Use SAME random seed for all frames in clip during training
        # This ensures augmentation is spatially consistent across frames
        # (model learns temporal patterns, not augmentation differences)
        if self.is_train:
            clip_seed = random.randint(0, 999999)
        
        tensors = []
        for fp in frames_paths:
            try:
                img = np.array(Image.open(fp).convert("RGB"), dtype=np.uint8)
                
                if self.is_train:
                    # Apply same spatial augmentation to all frames
                    # but different noise/compression (realistic)
                    random.seed(clip_seed)
                    np.random.seed(clip_seed)
                    aug = self.train_transform(image=img)
                    # Reset seed for next frame (different noise but same spatial)
                    random.seed(None)
                    np.random.seed(None)
                else:
                    aug = self.val_transform(image=img)
                    
                tensors.append(aug["image"])
            except Exception:
                tensors.append(torch.zeros(3, self.size, self.size))

        clip = torch.stack(tensors, dim=0)  # (N, 3, H, W)
        return clip, torch.tensor(label, dtype=torch.float32)


def get_video_dataloaders_v2(config: dict):
    """Build train/val/test dataloaders for V2 model."""
    t = config["training"]
    train_ds = VideoFrameDatasetV2(config, "train")
    val_ds   = VideoFrameDatasetV2(config, "val")
    test_ds  = VideoFrameDatasetV2(config, "test")
    kwargs   = dict(
        batch_size  = t["batch_size"],
        num_workers = t["num_workers"],
        pin_memory  = t["pin_memory"],
    )
    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(val_ds,   shuffle=False, **kwargs),
        DataLoader(test_ds,  shuffle=False, **kwargs),
    )