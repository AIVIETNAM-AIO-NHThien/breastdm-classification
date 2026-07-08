import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from typing import Tuple, List


class BreastDMDataset(Dataset):
    """
    Dataset cho bài toán phân loại u vú (BreastDM) với dữ liệu đa chuỗi.
    Hỗ trợ hai thí nghiệm:
    - Exp-1: 9 kênh (VIBRANT + VIBRANT+C1 ... +C8)
    - Exp-2: 17 kênh (VIBRANT + 8 post-contrast + 8 subtraction)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        experiment: str = "Exp-1",
        augment: bool = False,
    ):
        """
        Args:
            root_dir: Đường dẫn tới thư mục gốc ROI-classification
            split: 'train', 'val', hoặc 'test'
            experiment: 'Exp-1' hoặc 'Exp-2'
            augment: True nếu áp dụng augmentation (chỉ dùng cho tập train)
        """
        self.root_dir = root_dir
        self.split = split
        self.experiment = experiment
        self.augment = augment

        # Xác định các thư mục con cần đọc
        if experiment == "Exp-1":
            self.folders = ["VIBRANT"] + [f"VIBRANT+C{i}" for i in range(1, 9)]
        elif experiment == "Exp-2":
            self.folders = ["VIBRANT"] + [f"VIBRANT+C{i}" for i in range(1, 9)] + [f"SUB{i}" for i in range(1, 9)]
        else:
            raise ValueError("Experiment phải là 'Exp-1' hoặc 'Exp-2'")

        self.num_channels = len(self.folders)  # 9 hoặc 17

        # Tập nhãn: Benign -> 0, Malignant -> 1
        self.label_dict = {"Benign": 0, "Malignant": 1}

        # Xây dựng danh sách mẫu
        self.samples = self._build_samples()

    def _build_samples(self) -> List[dict]:
        """
        Duyệt qua cấu trúc thư mục để thu thập tất cả các lát cắt có đủ các kênh.
        Mỗi mẫu lưu: patient_dir, slice_name, label
        """
        samples = []
        split_dir = os.path.join(self.root_dir, self.split)

        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục split: {split_dir}")

        for label_name in os.listdir(split_dir):
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.isdir(label_dir):
                continue
            label = self.label_dict.get(label_name)
            if label is None:
                continue  # bỏ qua thư mục không đúng

            for patient_id in os.listdir(label_dir):
                patient_path = os.path.join(label_dir, patient_id)
                if not os.path.isdir(patient_path):
                    continue

                # Lấy danh sách slice từ thư mục VIBRANT (đại diện)
                vibrant_dir = os.path.join(patient_path, "VIBRANT")
                if not os.path.exists(vibrant_dir):
                    continue

                slice_names = [
                    f for f in os.listdir(vibrant_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                for slice_name in slice_names:
                    # Kiểm tra xem slice này có tồn tại trong TẤT CẢ các thư mục cần thiết không
                    valid = True
                    for folder in self.folders:
                        img_path = os.path.join(patient_path, folder, slice_name)
                        if not os.path.exists(img_path):
                            valid = False
                            break
                    if valid:
                        samples.append({
                            "patient_dir": patient_path,
                            "slice_name": slice_name,
                            "label": label,
                        })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_and_stack(self, patient_dir: str, slice_name: str) -> torch.Tensor:
        """
        Đọc các ảnh cùng tên từ các thư mục khác nhau và xếp chồng thành tensor (C, H, W).
        """
        channels = []
        for folder in self.folders:
            img_path = os.path.join(patient_dir, folder, slice_name)
            # Đọc ảnh grayscale
            img = Image.open(img_path).convert("L")
            # Chuyển thành tensor float (0-1)
            img_tensor = TF.to_tensor(img)  # shape (1, H, W)
            channels.append(img_tensor)

        # Stack theo chiều kênh -> (C, H, W)
        img_stack = torch.cat(channels, dim=0)  # (num_channels, H, W)
        return img_stack

    def _intensity_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Chuẩn hóa cường độ: clip 0.1% đuôi, sau đó z-score trên các voxel còn lại.
        """
        # Chuyển sang numpy để tính percentile dễ dàng
        arr = tensor.numpy()
        low = np.percentile(arr, 0.1)
        high = np.percentile(arr, 99.9)
        arr_clipped = np.clip(arr, low, high)

        # Z-score trên toàn bộ mẫu
        mean = arr_clipped.mean()
        std = arr_clipped.std()
        if std == 0:
            std = 1e-8
        arr_norm = (arr_clipped - mean) / std
        return torch.from_numpy(arr_norm).float()

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        """
        Áp dụng augmentation: RandomFlip, RandomRotation và RandomResizedCrop (scaling).
        Toàn bộ các kênh cùng chịu chung một phép biến đổi (giữ tương quan không gian).
        """
        # Random crop + resize để mô phỏng scaling (đúng như paper mô tả)
        img = TF.resized_crop(img, 0, 0, img.shape[1], img.shape[2],
                              [96, 96], scale=(0.8, 1.0), ratio=(1.0, 1.0))
        # Random horizontal flip
        if torch.rand(1) > 0.5:
            img = TF.hflip(img)
        # Random vertical flip
        if torch.rand(1) > 0.5:
            img = TF.vflip(img)
        # Random rotation (0, 90, 180, 270)
        angle = torch.randint(0, 4, (1,)).item() * 90
        if angle != 0:
            img = TF.rotate(img, angle, fill=0.0)
        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        patient_dir = sample["patient_dir"]
        slice_name = sample["slice_name"]
        label = sample["label"]

        # 1. Đọc và xếp chồng kênh
        img = self._load_and_stack(patient_dir, slice_name)  # (C, H, W)

        # 2. Augmentation (chỉ áp dụng cho tập train, đã bao gồm resize)
        if self.augment:
            img = self._augment(img)

        # 3. Chuẩn hóa cường độ
        img = self._intensity_normalize(img)

        # 4. Resize về 96x96 (đảm bảo kích thước cho cả val/test)
        img = TF.resize(img, [96, 96], antialias=True)

        return img, label


def create_dataloaders(
    root_dir: str,
    experiment: str = "Exp-1",
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train, val, test với augmentation chỉ trên train.
    """
    train_dataset = BreastDMDataset(
        root_dir=root_dir,
        split="train",
        experiment=experiment,
        augment=True,
    )
    val_dataset = BreastDMDataset(
        root_dir=root_dir,
        split="val",
        experiment=experiment,
        augment=False,
    )
    test_dataset = BreastDMDataset(
        root_dir=root_dir,
        split="test",
        experiment=experiment,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


# Ví dụ sử dụng
if __name__ == "__main__":
    # Giả sử dữ liệu nằm ở ./ROI-classification
    root = "/kaggle/input/roi-classification"
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=root,
        experiment="Exp-2",
        batch_size=8,
        num_workers=2,
    )
    # Kiểm tra một batch
    for imgs, labels in train_loader:
        print(f"Batch shape: {imgs.shape}")   # [B, 17, 96, 96] với Exp-2
        print(f"Labels: {labels}")
        break