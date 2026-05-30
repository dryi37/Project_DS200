import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Image-net mean / std used for both encoder inputs
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def _train_augmentations() -> A.Compose:
    """
    Standard augmentation suite for medical polyp images.
    Đã fix các cảnh báo của Albumentations phiên bản mới nhất.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-15, 15),
            p=0.4, # Đã xóa 'mode' gây cảnh báo
        ),
        A.ElasticTransform(alpha=80, sigma=10, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(1, 32),
            hole_width_range=(1, 32),
            p=0.1, # Đã xóa 'fill_value' gây cảnh báo
        ),
    ], additional_targets={"mask": "mask"})


def _val_augmentations() -> A.Compose:
    """No spatial or photometric augmentation at validation time."""
    return A.Compose([], additional_targets={"mask": "mask"})


def _to_tensor_normalise(size: int) -> A.Compose:
    """
    Resize to *size* × *size*, convert to float32 tensor in [0, 1],
    then apply ImageNet normalisation.
    """
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def _mask_resize(size: int) -> A.Compose:
    """Resize a binary mask using nearest-neighbour interpolation."""
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_NEAREST),
    ])


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class PolypDataset(Dataset):
    """
    Dataset class robust to varying image extensions (.jpg, .png, .tif).
    Matches images to masks based purely on the filename stem.
    """

    def __init__(
        self,
        root: str | Path,
        mode: Literal["train", "val", "test"] = "train",
        img_size_sam: int = 1024,
        img_size_cnx: int = 448,
        mask_size: int = 352,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.mode = mode
        
        # --- LOGIC TỰ DÒ THƯ MỤC ẢNH ---
        if (self.root / "images").exists():
            self.img_dir = self.root / "images"
        elif (self.root / "image").exists():
            self.img_dir = self.root / "image"
        else:
            raise RuntimeError(f"Không tìm thấy thư mục 'images' hoặc 'image' trong {self.root}")

        # --- LOGIC TỰ DÒ THƯ MỤC MASK ---
        if (self.root / "masks").exists():
            self.mask_dir = self.root / "masks"
        elif (self.root / "mask").exists():
            self.mask_dir = self.root / "mask"
        else:
            raise RuntimeError(f"Không tìm thấy thư mục 'masks' hoặc 'mask' trong {self.root}")

        self.img_paths = sorted([p for p in self.img_dir.iterdir() if p.is_file()])
        self.stems = [p.stem for p in self.img_paths]
        
        if len(self.img_paths) == 0:
            raise RuntimeError(f"Không tìm thấy ảnh nào trong {self.img_dir}")

        self.aug = _train_augmentations() if mode == "train" else _val_augmentations()
        self.to_sam = _to_tensor_normalise(img_size_sam)
        self.to_cnx = _to_tensor_normalise(img_size_cnx)
        self.resize_mask = _mask_resize(mask_size)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.img_paths[idx]
        stem = img_path.stem

        # Quét thư mục masks để tìm file có chung tên (stem) bất chấp đuôi
        mask_candidates = list(self.mask_dir.glob(f"{stem}.*"))
        if not mask_candidates:
            raise FileNotFoundError(f"Không tìm thấy mask tương ứng cho ảnh: {stem}")
        mask_path = mask_candidates[0]

        # --- Load image (BGR → RGB) and mask ---
        img  = cv2.cvtColor(cv2.imread(str(img_path),  cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise RuntimeError(f"Không thể đọc file ảnh: {img_path}")
        if mask is None:
            raise RuntimeError(f"Không thể đọc file mask: {mask_path}")

        # Binarise mask
        mask = (mask > 127).astype(np.uint8)

        # --- Augmentation ---
        augmented = self.aug(image=img, mask=mask)
        aug_img   = augmented["image"]
        aug_mask  = augmented["mask"]

        # --- Tensors generation ---
        img_sam: torch.Tensor = self.to_sam(image=aug_img)["image"]
        img_cnx: torch.Tensor = self.to_cnx(image=aug_img)["image"]

        resized_mask = self.resize_mask(image=aug_mask)["image"]
        gt_mask: torch.Tensor = torch.from_numpy(resized_mask.astype(np.float32)).unsqueeze(0)

        return {
            "img_sam" : img_sam,    # (3, 1024, 1024)
            "img_cnx" : img_cnx,    # (3,  448,  448)
            "mask"    : gt_mask,    # (1,  352,  352)
            "stem"    : stem,       # string
        }


# ---------------------------------------------------------------------------
# Convenience DataLoader factory
# ---------------------------------------------------------------------------

def get_loader(
    root: str | Path,
    batch_size: int = 4,
    mode: Literal["train", "val", "test"] = "train",
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    dataset = PolypDataset(root, mode=mode, **dataset_kwargs)
    shuffle = (mode == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=(mode == "train"),
    )
    print(
        f"[DataLoader] mode={mode} | samples={len(dataset)} | "
        f"batches={len(loader)} | batch_size={batch_size}"
    )
    return loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/TrainDataset"
    
    try:
        loader = get_loader(root, batch_size=2, mode="train", num_workers=0)
        batch = next(iter(loader))
        print("img_sam :", batch["img_sam"].shape,  batch["img_sam"].dtype)
        print("img_cnx :", batch["img_cnx"].shape,  batch["img_cnx"].dtype)
        print("mask    :", batch["mask"].shape,    batch["mask"].dtype)
        print("stems   :", batch["stem"])
        print("✓ Dataset hoạt động hoàn hảo!")
    except Exception as e:
        print(f"Lỗi khởi tạo dataset: {e}")