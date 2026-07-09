import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from typing import Tuple, List
import random


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
        self.root_dir = root_dir
        self.split = split
        self.experiment = experiment
        self.augment = augment

        if experiment == "Exp-1":
            self.folders = ["VIBRANT"] + [f"VIBRANT+C{i}" for i in range(1, 9)]
        elif experiment == "Exp-2":
            self.folders = ["VIBRANT"] + [f"VIBRANT+C{i}" for i in range(1, 9)] + [f"SUB{i}" for i in range(1, 9)]
        else:
            raise ValueError("Experiment phải là 'Exp-1' hoặc 'Exp-2'")

        self.num_channels = len(self.folders)
        self.label_dict = {"Benign": 0, "Malignant": 1}

        # Khởi tạo RandomResizedCrop cho augmentation (scaling)
        self.random_crop = transforms.RandomResizedCrop(
            size=96, scale=(0.8, 1.0), ratio=(1.0, 1.0)
        )

        self.samples = self._build_samples()

        # Lưu danh sách nhãn để sampler dùng
        self.labels = [s["label"] for s in self.samples]

    def _build_samples(self) -> List[dict]:
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
                continue

            for patient_id in os.listdir(label_dir):
                patient_path = os.path.join(label_dir, patient_id)
                if not os.path.isdir(patient_path):
                    continue

                vibrant_dir = os.path.join(patient_path, "VIBRANT")
                if not os.path.exists(vibrant_dir):
                    continue

                slice_names = [
                    f for f in os.listdir(vibrant_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]

                for slice_name in slice_names:
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
        channels = []
        for folder in self.folders:
            img_path = os.path.join(patient_dir, folder, slice_name)
            img = Image.open(img_path).convert("L")
            img_tensor = TF.to_tensor(img)  # (1, H, W)
            channels.append(img_tensor)
        return torch.cat(channels, dim=0)  # (C, H, W)

    def _intensity_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        arr = tensor.numpy()
        low = np.percentile(arr, 0.1)
        high = np.percentile(arr, 99.9)
        arr_clipped = np.clip(arr, low, high)
        mean = arr_clipped.mean()
        std = arr_clipped.std()
        if std == 0:
            std = 1e-8
        arr_norm = (arr_clipped - mean) / std
        return torch.from_numpy(arr_norm).float()

    def _augment(self, img: torch.Tensor) -> torch.Tensor:
        # Random crop + resize (scaling) – dùng RandomResizedCrop
        img = self.random_crop(img)

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

        # 2. Augmentation (chỉ tập train, đã bao gồm resize về 96x96)
        if self.augment:
            img = self._augment(img)

        # 3. Chuẩn hóa cường độ
        img = self._intensity_normalize(img)

        # 4. Resize lần cuối (đảm bảo 96x96 cho cả val/test)
        img = TF.resize(img, [96, 96], antialias=True)

        return img, label


class TripletBatchSampler(Sampler):
    """
    BatchSampler cho Triplet Loss:
    Mỗi batch gồm batch_size mẫu, chia đều cho 2 lớp (batch_size/2 mỗi lớp).
    Đảm bảo mỗi batch có cả benign và malignant để tạo triplet.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_per_class = batch_size // 2  # mỗi lớp lấy 8 mẫu (với batch 16)

        # Lấy danh sách chỉ số theo lớp
        self.class_indices = {}
        for idx, label in enumerate(self.dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        self.num_batches = min(
            len(self.class_indices[0]) // self.num_per_class,
            len(self.class_indices[1]) // self.num_per_class
        )

    def __iter__(self):
        # Shuffle các chỉ số trong mỗi lớp
        indices0 = self.class_indices[0].copy()
        indices1 = self.class_indices[1].copy()
        if self.shuffle:
            random.shuffle(indices0)
            random.shuffle(indices1)

        batches = []
        for i in range(self.num_batches):
            batch = []
            # Lấy num_per_class mẫu từ mỗi lớp
            start = i * self.num_per_class
            batch.extend(indices0[start:start+self.num_per_class])
            batch.extend(indices1[start:start+self.num_per_class])
            random.shuffle(batch)
            batches.append(batch)

        if self.shuffle:
            random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return self.num_batches


def create_dataloaders(
    root_dir: str,
    experiment: str = "Exp-1",
    batch_size: int = 16,
    num_workers: int = 4,
    use_triplet: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train, val, test.
    Nếu use_triplet=True, train_loader sẽ dùng TripletBatchSampler.
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

    if use_triplet:
        sampler = TripletBatchSampler(train_dataset, batch_size=batch_size, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
        )
    else:
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


if __name__ == "__main__":
    root = "/kaggle/input/roi-classification"
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=root,
        experiment="Exp-2",
        batch_size=8,
        num_workers=2,
        use_triplet=False,
    )
    for imgs, labels in train_loader:
        print(f"Batch shape: {imgs.shape}")
        print(f"Labels: {labels}")
        break