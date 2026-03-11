import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import torchvision.transforms as T

NORM_MEAN = [0.485]
NORM_STD = [0.229]

# ---------------- DATASET -----------------
class BreastDMExp1Dataset(Dataset):
    def __init__(self, root_path, split="train"):
        self.split = split
        self.root_path = os.path.join(root_path, "img9Se", split)
        self.samples = []
        self.labels = []

        classes = sorted([c for c in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path,c))])
        for label, cls in enumerate(classes):
            cls_path = os.path.join(self.root_path, cls)
            for case in os.listdir(cls_path):
                case_path = os.path.join(cls_path, case)
                if not os.path.isdir(case_path): continue
                for file in os.listdir(case_path):
                    if file.endswith(".npy"):
                        self.samples.append(os.path.join(case_path,file))
                        self.labels.append(label)

        # Define transforms once
        self.resize = T.Resize((256,256))
        self.crop = T.RandomCrop(224)
        self.hflip = T.RandomHorizontalFlip()
        self.vflip = T.RandomVerticalFlip()
        self.to_tensor = T.Lambda(lambda x: x)  # placeholder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]

        # Load numpy -> (H,W,9)
        data = np.load(path).astype(np.float32)
        data = data.transpose(2,0,1)[:, None, :, :]  # (9,1,H,W)
        data = torch.from_numpy(data)               # tensor

        if self.split=="train":
            # Resize
            data = torch.nn.functional.interpolate(data, size=(256,256), mode='bilinear', align_corners=False)
            # Random crop
            i = random.randint(0, data.shape[2]-224)
            j = random.randint(0, data.shape[3]-224)
            data = data[:,:,i:i+224,j:j+224]
            # Random flip
            if random.random() > 0.5:
                data = torch.flip(data, dims=[3])  # horizontal
            if random.random() > 0.5:
                data = torch.flip(data, dims=[2])  # vertical
        else:
            # Resize val/test
            data = torch.nn.functional.interpolate(data, size=(224,224), mode='bilinear', align_corners=False)

        # Normalize
        data = (data - NORM_MEAN[0])/NORM_STD[0]

        return data, label

# ---------------- DATALOADER -----------------
def build_dataloader(root_path, split, batch_size):
    dataset = BreastDMExp1Dataset(root_path, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=2, pin_memory=True, drop_last=(split=="train"))
    return loader

def get_dataloaders(root_path, batch_size):
    train_loader = build_dataloader(root_path,"train",batch_size)
    val_loader = build_dataloader(root_path,"val",batch_size)
    test_loader = build_dataloader(root_path,"test",batch_size)
    return train_loader, val_loader, test_loader

# ---------------- TEST -----------------
if __name__ == "__main__":
    root = "/kaggle/input/breastdm/cls"
    train_loader, val_loader, test_loader = get_dataloaders(root, 4)
    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    print("Test size:", len(test_loader.dataset))

    for data,label in train_loader:
        print("Batch shape:", data.shape)  # (B,9,1,H,W)
        print("Labels:", label)
        break