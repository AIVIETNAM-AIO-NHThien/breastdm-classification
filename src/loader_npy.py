import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from typing import Tuple

class BreastDMNpyDataset(Dataset):
    """
    Dataset cho dữ liệu đã được tiền xử lý và lưu dưới dạng file .npy.
    Mỗi file .npy chứa một mảng (C, 96, 96) đã stack 9 hoặc 17 kênh,
    đã được chuẩn hóa cường độ và resize.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        augment: bool = False,
    ):
        self.root_dir = root_dir
        self.split = split
        self.augment = augment

        self.samples = []
        self.labels = []
        label_map = {"Benign": 0, "Malignant": 1}

        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {split_dir}")

        for label_name in ["Benign", "Malignant"]:
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label = label_map[label_name]
            for patient_id in os.listdir(label_dir):
                patient_dir = os.path.join(label_dir, patient_id)
                if not os.path.isdir(patient_dir):
                    continue
                for fname in os.listdir(patient_dir):
                    if fname.endswith(".npy"):
                        self.samples.append(os.path.join(patient_dir, fname))
                        self.labels.append(label)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """Augmentation: flip và rotation (không crop vì dữ liệu đã 96x96)."""
        if torch.rand(1) > 0.5:
            img = TF.hflip(img)
        if torch.rand(1) > 0.5:
            img = TF.vflip(img)
        angle = torch.randint(0, 4, (1,)).item() * 90
        if angle != 0:
            img = TF.rotate(img, angle, fill=0.0)
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        arr = np.load(self.samples[index])          # shape (C, 96, 96)
        img = torch.from_numpy(arr).float()         # đã là float, không cần /255

        if self.augment:
            img = self._augment(img)

        return img, self.labels[index]


def create_dataloaders_npy(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train/val/test từ dữ liệu .npy.
    """
    train_dataset = BreastDMNpyDataset(root_dir, split="train", augment=True)
    val_dataset   = BreastDMNpyDataset(root_dir, split="val", augment=False)
    test_dataset  = BreastDMNpyDataset(root_dir, split="test", augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test nhanh
    root = "/kaggle/input/breast-dm/cls/img9Se"   # ví dụ cho Exp-1
    train_loader, val_loader, test_loader = create_dataloaders_npy(root, batch_size=8)
    for imgs, labels in train_loader:
        print("Batch shape:", imgs.shape)   # [B, 9, 96, 96]
        print("Labels:", labels)
        break