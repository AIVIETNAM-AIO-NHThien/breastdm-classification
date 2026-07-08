import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from typing import Tuple

class BreastDMNpyDataset(Dataset):
    """
    Dataset cho dữ liệu đã được lưu dưới dạng file .npy.
    Mỗi file .npy chứa một mảng (C, H, W) hoặc (H, W, C) – có thể không đồng nhất kích thước.
    Sẽ tự động chuyển về (C, 96, 96).
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
        """Augmentation: chỉ flip và rotation, KHÔNG crop vì ảnh đã nhỏ."""
        if torch.rand(1) > 0.5:
            img = TF.hflip(img)
        if torch.rand(1) > 0.5:
            img = TF.vflip(img)
        angle = torch.randint(0, 4, (1,)).item() * 90
        if angle != 0:
            img = TF.rotate(img, angle, fill=0.0)
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        arr = np.load(self.samples[index])   # có thể là (H, W, C) hoặc (C, H, W)

        # Xác định và chuyển về dạng (C, H, W)
        if arr.ndim == 3:
            # Nếu chiều cuối là 9 hoặc 17 => kênh ở cuối => transpose
            if arr.shape[-1] in [9, 17]:
                arr = np.transpose(arr, (2, 0, 1))
            # Ngược lại, nếu chiều đầu là 9 hoặc 17 => đã đúng (C, H, W)
            elif arr.shape[0] in [9, 17]:
                pass  # giữ nguyên
            else:
                # Đoán: nếu chiều thứ nhất nhỏ hơn 10 => coi là kênh
                if arr.shape[0] < 10:
                    pass  # coi như (C, H, W)
                else:
                    # Mặc định coi kênh ở cuối
                    arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected array shape: {arr.shape}")

        img = torch.from_numpy(arr).float()

        # Resize về 96x96 (dùng antialias để giữ chất lượng)
        img = TF.resize(img, [96, 96], antialias=True)

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
    root = "/kaggle/input/breast-dm/cls/img9Se"
    train_loader, val_loader, test_loader = create_dataloaders_npy(root, batch_size=8)
    for imgs, labels in train_loader:
        print("Batch shape:", imgs.shape)   # Kỳ vọng [B, 9, 96, 96]
        print("Labels:", labels)
        break