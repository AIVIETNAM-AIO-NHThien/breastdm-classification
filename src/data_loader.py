import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from torchvision import transforms
import torchvision.transforms.functional as F

# ---------------- CONSTANTS -----------------
NORM_MEAN = [0.485]
NORM_STD = [0.229]

# ---------------- DATASET -----------------
class BreastDMExp1Dataset(Dataset):
    def __init__(self, root_path, split="train"):
        """
        root_path: thư mục cls chứa img9Se
        split: 'train', 'val', 'test'
        """
        self.split = split
        self.root_path = os.path.join(root_path, "img9Se", split)

        self.samples = []
        self.labels = []

        # load class names
        classes = sorted([
            c for c in os.listdir(self.root_path)
            if os.path.isdir(os.path.join(self.root_path, c))
        ])

        # scan all .npy files
        for label, cls in enumerate(classes):
            cls_path = os.path.join(self.root_path, cls)
            for case in os.listdir(cls_path):
                case_path = os.path.join(cls_path, case)
                if not os.path.isdir(case_path):
                    continue
                for file in os.listdir(case_path):
                    if file.endswith(".npy"):
                        self.samples.append(os.path.join(case_path, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]

        # ---------- LOAD AND FIX SHAPE ----------
        data = np.load(path)           # shape: (H,W,9)
        data = data.transpose(2,0,1)  # (9,H,W)
        data = data[:, None, :, :]     # (9,1,H,W)

        # Convert each slice to PIL
        imgs = [F.to_pil_image(data[i]) for i in range(data.shape[0])]

        processed = []

        if self.split == "train":
            # Resize to 256x256 first
            imgs = [F.resize(img, (256,256)) for img in imgs]

            # Random crop 224x224, same params for all slices
            i,j,h,w = transforms.RandomCrop.get_params(imgs[0], (224,224))
            do_hflip = random.random() > 0.5
            do_vflip = random.random() > 0.5

            for img in imgs:
                img = F.crop(img, i,j,h,w)
                if do_hflip:
                    img = F.hflip(img)
                if do_vflip:
                    img = F.vflip(img)
                img = F.to_tensor(img)
                img = F.normalize(img, NORM_MEAN, NORM_STD)
                processed.append(img)
        else:
            # val/test: resize only
            for img in imgs:
                img = F.resize(img,(224,224))
                img = F.to_tensor(img)
                img = F.normalize(img, NORM_MEAN, NORM_STD)
                processed.append(img)

        data = torch.stack(processed)  # (9,1,H,W)
        return data, label

# ---------------- DATALOADER -----------------
def build_dataloader(root_path, split, batch_size):
    dataset = BreastDMExp1Dataset(root_path, split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split=="train"),
        num_workers=2,
        pin_memory=True,
        drop_last=(split=="train")
    )
    return loader

def get_dataloaders(root_path, batch_size):
    train_loader = build_dataloader(root_path, "train", batch_size)
    val_loader = build_dataloader(root_path, "val", batch_size)
    test_loader = build_dataloader(root_path, "test", batch_size)
    return train_loader, val_loader, test_loader

# ---------------- TEST -----------------
if __name__ == "__main__":
    root = "/kaggle/input/breastdm/cls"
    train_loader, val_loader, test_loader = get_dataloaders(root, 4)

    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    print("Test size:", len(test_loader.dataset))

    for data, label in train_loader:
        print("Batch shape:", data.shape)  # (B,9,1,H,W)
        print("Labels:", label)
        break