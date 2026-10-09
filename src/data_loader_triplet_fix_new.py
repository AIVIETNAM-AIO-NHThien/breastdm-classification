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
        self.samples = self._build_samples()

        # Lưu danh sách nhãn để sampler dùng
        self.labels = [s["label"] for s in self.samples]

        # Augmentation giống data_loader thường (Resize 256 -> RandomCrop 224 -> Resize 96)
        if augment:
            self.augmentation = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.Resize([96, 96]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
            ])
        else:
            self.augmentation = transforms.Compose([
                transforms.Resize([96, 96]),])

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
            img_tensor = TF.to_tensor(img)
            channels.append(img_tensor)
        return torch.cat(channels, dim=0)

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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[index]
        patient_dir = sample["patient_dir"]
        slice_name = sample["slice_name"]
        label = sample["label"]

        # 1. Đọc và xếp chồng kênh
        img = self._load_and_stack(patient_dir, slice_name)

        # 2. Augmentation (chỉ train)
        if self.augmentation is not None:
            img = self.augmentation(img)
        else:
            img = TF.resize(img, [96, 96])

        # 3. Chuẩn hóa cường độ
        img = self._intensity_normalize(img)

        return img, label


# ============================================================
# PKSAMPLER — P classes × K samples per class
# ============================================================
class PKSampler(Sampler):
    """
    PKSampler: mỗi batch có P class × K samples.

    Class thiểu số được lặp lại (wrap-around) để đủ batch.
    Không cắt bớt slices như TripletBatchSampler.

    Ví dụ: P=2, K=16 → mỗi batch 32 slices (16 benign + 16 malignant).
    Số batch = max(số batch có thể tạo từ mỗi class).
    """
    def __init__(self, dataset, P=2, K=16, shuffle=True):
        self.P = P
        self.K = K
        self.batch_size = P * K
        self.shuffle = shuffle

        # Nhóm indices theo label
        self.class_indices = {}
        for idx, label in enumerate(dataset.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)

        # Số batch = max(số batch có thể tạo từ mỗi class)
        self.num_batches = max(
            len(indices) // K for indices in self.class_indices.values()
        )

    def __iter__(self):
        # Shuffle trong mỗi class
        class_indices = {
            c: idxs.copy() for c, idxs in self.class_indices.items()
        }
        if self.shuffle:
            for idxs in class_indices.values():
                random.shuffle(idxs)

        # Con trỏ vòng lặp cho mỗi class
        pointers = {c: 0 for c in class_indices}

        for _ in range(self.num_batches):
            batch = []
            for c in class_indices.keys():
                idxs = class_indices[c]
                n = len(idxs)

                # Lấy K samples, wrap-around nếu cần
                for j in range(self.K):
                    pos = (pointers[c] + j) % n
                    batch.append(idxs[pos])

                # Cập nhật pointer
                pointers[c] = (pointers[c] + self.K) % n

            if self.shuffle:
                random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches


# ============================================================
# TRIPLETBATCHSAMPLER — Giữ lại để so sánh (không dùng mặc định)
# ============================================================
class TripletBatchSampler(Sampler):
    """
    BatchSampler cho Triplet Loss (cách cũ):
    Mỗi batch gồm batch_size mẫu, chia đều cho 2 lớp.

    NHƯỢC ĐIỂM: Cắt bớt slices vì giới hạn bởi class nhỏ hơn.
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_per_class = batch_size // 2

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
        indices0 = self.class_indices[0].copy()
        indices1 = self.class_indices[1].copy()
        if self.shuffle:
            random.shuffle(indices0)
            random.shuffle(indices1)

        batches = []
        for i in range(self.num_batches):
            batch = []
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


# ============================================================
# CREATE DATALOADERS
# ============================================================
def create_dataloaders(
    root_dir: str,
    experiment: str = "Exp-1",
    batch_size: int = 16,
    num_workers: int = 4,
    use_triplet: bool = False,
    sampler_type: str = "pk",       # "pk" hoặc "triplet"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo DataLoader cho train, val, test.

    Args:
        sampler_type: "pk" (mặc định, không cắt bớt) hoặc "triplet" (cũ, cắt bớt).

    Nếu use_triplet=True, train_loader sẽ dùng sampler_type chỉ định.
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
        if sampler_type == "pk":
            sampler = PKSampler(
                train_dataset,
                P=2,
                K=batch_size // 2,
                shuffle=True,
            )
        elif sampler_type == "triplet":
            sampler = TripletBatchSampler(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            raise ValueError(f"sampler_type không hợp lệ: {sampler_type}")

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
        )
        print(f"[{sampler_type.upper()}Sampler] Batches/epoch: {len(sampler)}, "
              f"batch_size: {batch_size}")
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
        batch_size=32,
        num_workers=2,
        use_triplet=True,
        sampler_type="pk",
    )

    print(f"\nTrain dataset: {len(train_loader.dataset)} slices")
    print(f"Batches/epoch: {len(train_loader)}")

    for imgs, labels in train_loader:
        print(f"\nBatch shape: {imgs.shape}")
        unique, counts = labels.unique(return_counts=True)
        print(f"Labels trong batch: {dict(zip(unique.tolist(), counts.tolist()))}")
        break