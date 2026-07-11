from torchvision import datasets, transforms
import torch
import os

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize([256,256]),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)   # BỎ inplace=True
    ])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return train_loader

def load_validation(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)   # BỎ inplace=True
    ])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir), transform=transform)
    names = list(map(lambda x: os.path.basename(x[0]), list(data.imgs)))
    label = list(map(lambda x: x[1], list(data.imgs)))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)
    return loader, names, label

def load_testing(root_path, dir, batch_size, kwargs):
    return load_validation(root_path, dir, batch_size, kwargs)