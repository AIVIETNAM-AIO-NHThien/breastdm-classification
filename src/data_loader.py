
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

IMG_SIZE = 224


# -------- Transforms --------
def get_transforms(is_train=True):

    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ])


# -------- Dataset --------
class CachedNPYDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.samples = []
        self.labels = []
        self.transform = transform

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:

            cls_path = os.path.join(root_dir, cls)

            for root, _, files in os.walk(cls_path):

                for f in files:

                    if f.lower().endswith(".npy"):

                        self.samples.append(os.path.join(root, f))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path = self.samples[idx]
        label = self.labels[idx]

        img = np.load(path)

        img = torch.tensor(img).float()

        # scale pixel
        img = img / 255.0

        # nếu grayscale
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
            img = img.repeat(3, 1, 1)

        # nếu HWC -> CHW
        elif len(img.shape) == 3 and img.shape[0] != 3:
            img = img.permute(2, 0, 1)

        if self.transform:
            img = self.transform(img)

        return img, label


# -------- Data Loaders --------
def load_training(root_path, phase='train', batch_size=32, num_workers=4):

    transform = get_transforms(True)

    dataset = CachedNPYDataset(
        os.path.join(root_path, phase),
        transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def load_testing(root_path, phase='val', batch_size=32, num_workers=4):

    transform = get_transforms(False)

    dataset = CachedNPYDataset(
        os.path.join(root_path, phase),
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader, dataset.labels
if __name__ == "__main__":

    DATA_PATH = "/kaggle/input/breastdm/cls/img9Se"

    train_loader = load_training(DATA_PATH, "train", batch_size=8)
    val_loader, labels = load_testing(DATA_PATH, "val", batch_size=8)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))

    for images, targets in train_loader:
        print("Image shape:", images.shape)
        print("Labels:", targets)
        break