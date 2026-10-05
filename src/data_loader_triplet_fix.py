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

        # ===== THÊM: mapping patient_id (str) → int =====
        self.patient_ids_str = [s["patient_id"] for s in self.samples]
        unique_pids = sorted(set(self.patient_ids_str))
        self.patient_to_int = {pid: i for i, pid in enumerate(unique_pids)}
        self.patient_ids = [self.patient_to_int[pid] for pid in self.patient_ids_str]

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
                            "patient_id": patient_id,      # ← THÊM
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[index]
        patient_dir = sample["patient_dir"]
        slice_name = sample["slice_name"]
        label = sample["label"]
        patient_id_int = self.patient_to_int[sample["patient_id"]]   # ← THÊM

        # 1. Đọc và xếp chồng kênh
        img = self._load_and_stack(patient_dir, slice_name)  # (C, H, W)

        # 2. Augmentation (chỉ train)
        if self.augmentation is not None:
            img = self.augmentation(img)
        else:
            img = TF.resize(img, [96, 96])

        # 3. Chuẩn hóa cường độ
        img = self._intensity_normalize(img)

        # Trả về thêm patient_id_int
        return img, label, patient_id_int


class TripletBatchSampler(Sampler):
    """
    BatchSampler cho Triplet Loss (patient-aware).
    Mỗi batch:
      - num_per_class mẫu class 0, num_per_class mẫu class 1
      - Mỗi mẫu đến từ 1 BỆNH NHÂN KHÁC NHAU
        → đảm bảo positive luôn khác bệnh nhân với anchor
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_per_class = batch_size // 2

        # Nhóm indices theo (label, patient_id)
        self.class_patient_indices = {}
        for idx, (label, pid) in enumerate(zip(dataset.labels, dataset.patient_ids)):
            key = (label, pid)
            self.class_patient_indices.setdefault(key, []).append(idx)

        # Nhóm patient_id theo class
        self.class_patients = {0: [], 1: []}
        for (label, pid) in self.class_patient_indices.keys():
            self.class_patients[label].append(pid)

        # num_batches giới hạn bởi class có ít patient nhất
        n0 = len(self.class_patients[0]) // self.num_per_class
        n1 = len(self.class_patients[1]) // self.num_per_class
        self.num_batches = min(n0, n1)

    def __iter__(self):
        patients0 = self.class_patients[0].copy()
        patients1 = self.class_patients[1].copy()
        if self.shuffle:
            random.shuffle(patients0)
            random.shuffle(patients1)

        # Cắt vừa đủ num_batches * num_per_class
        n_used = self.num_batches * self.num_per_class
        patients0 = patients0[:n_used]
        patients1 = patients1[:n_used]

        batches = []
        for i in range(self.num_batches):
            start = i * self.num_per_class
            sel0 = patients0[start:start + self.num_per_class]
            sel1 = patients1[start:start + self.num_per_class]

            batch = []
            # Mỗi bệnh nhân đóng góp ĐÚNG 1 slice vào batch
            for pid in sel0:
                idxs = self.class_patient_indices[(0, pid)]
                batch.append(random.choice(idxs))
            for pid in sel1:
                idxs = self.class_patient_indices[(1, pid)]
                batch.append(random.choice(idxs))

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
    Nếu use_triplet=True, train_loader sẽ dùng TripletBatchSampler (patient-aware).
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
        use_triplet=True,
    )

    print("\n=== Kiểm tra patient diversity trong batch ===")
    for i, (imgs, labels, patient_ids) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  imgs shape: {imgs.shape}")
        print(f"  labels: {labels.tolist()}")
        print(f"  patient_ids: {patient_ids.tolist()}")
        print(f"  Số bệnh nhân khác nhau: {len(patient_ids.unique())}")
        print(f"  Tổng số mẫu: {len(patient_ids)}")
        if i >= 2:
            break