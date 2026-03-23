import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

class BreastDCE_Dataset(Dataset):
    def __init__(self, root_path, split="train"):
        self.root = os.path.join(root_path, split)
        self.split = split

        self.samples = []
        self.labels = []

        classes = sorted([c for c in os.listdir(self.root) 
                          if os.path.isdir(os.path.join(self.root, c))])

        for label, cls in enumerate(classes):
            cls_path = os.path.join(self.root, cls)

            for case in os.listdir(cls_path):
                case_path = os.path.join(cls_path, case)
                if not os.path.isdir(case_path):
                    continue
                for f in os.listdir(case_path):
                    if f.endswith(".npy"):
                        self.samples.append(os.path.join(case_path, f))
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]

        # Load (H,W,9)
        data = np.load(path).astype(np.float32)
        data = data.transpose(2, 0, 1)  # (9,H,W)
        data = data - data[0:1, :, :]   

        # Normalize per-case
        mean = data.mean()
        std = data.std() + 1e-6
        data = (data - mean) / std

        data = torch.from_numpy(data)  # (9,H,W)
        data = data.unsqueeze(0)       # (1,9,H,W)
        data = F.interpolate(data, size=(256,256), mode='bilinear', align_corners=False)
        data = data.squeeze(0)         # (9,256,256)

        # Data augmentation
        if self.split == "train":
            i = random.randint(0, 256-224)
            j = random.randint(0, 256-224)
            data = data[:, i:i+224, j:j+224]
            if random.random() > 0.5:
                data = torch.flip(data, dims=[2])
            if random.random() > 0.5:
                data = torch.flip(data, dims=[1])
        else:
            data = data[:, 16:240, 16:240]

        # Clamp & scale to [0,1]
        data = torch.clamp(data, -3, 3)
        data = (data + 3) / 6

        return data, label

def build_dataloader(root_path, split, batch_size, num_workers=2):
    dataset = BreastDCE_Dataset(root_path, split)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=(split=="train"),
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=(split=="train"))
    return loader

def get_dataloaders(root_path, batch_size, num_workers=2):
    train_loader = build_dataloader(root_path, "train", batch_size, num_workers)
    val_loader   = build_dataloader(root_path, "val", batch_size, num_workers)
    test_loader  = build_dataloader(root_path, "test", batch_size, num_workers)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    root = "/kaggle/input/breastdm/cls/img9Se"
    train_loader, val_loader, test_loader = get_dataloaders(root, batch_size=4)
    print("Train size:", len(train_loader.dataset))
    for x, y in train_loader:
        print("Batch shape:", x.shape)
        print("Min/Max:", x.min().item(), x.max().item())
        print("Labels:", y)
        break