import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import timm

import data_loader
import Models
import Fusion_model
# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='vit_full')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--data_path', type=str, default='../input/roi-classification')
parser.add_argument('--save_path', type=str, default='./checkpoints')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

# ---------------- CUDA ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Load Data ----------------
train_loader = data_loader.load_training(args.data_path, 'train', args.batch_size)
val_loader, val_labels = data_loader.load_testing(args.data_path, 'val', args.batch_size)

# ---------------- Build Model ----------------
def build_model(name):
    name = name.lower()
    if name == 'resnet18': return Models.ResNet18(args.num_classes)
    if name == 'resnet50': return Models.ResNet50(args.num_classes)
    if name == 'resnet101': return Models.ResNet101(args.num_classes)
    if name == 'densenet169': return Models.DenseNet169(args.num_classes)
    if name == 'densenet201': return Models.DenseNet201(args.num_classes)
    if name == 'vgg16': return Models.VGG16(args.num_classes)
    if name == 'senet50': return Models.SENet50(args.num_classes)
    if name == 'resnext101': return Models.ResNeXt101(args.num_classes)
    if name == "fusion_model": return Fusion_model.FusionModel(args.num_classes)
    if name in ['vit', 'vit_full']:
        # Load pretrained ViT-B/16 từ timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=args.num_classes)
        return model

    raise ValueError(f"Unsupported model: {name}")

model = build_model(args.model)

# ---------------- Freeze / Fine-tune ----------------
if args.model.lower() in ['vit', 'vit_full']:
    print("🔹 Freezing pretrained ViT-B/16 blocks 0-7, fine-tuning 8-11 + head")
    # Freeze all
    for p in model.parameters():
        p.requires_grad = False

    # Fine-tune last 4 transformer blocks
    for blk in range(8, 12):
        for p in model.blocks[blk].parameters():
            p.requires_grad = True

    # Fine-tune head
    for p in model.head.parameters():
        p.requires_grad = True

# ---------------- Multi-GPU ----------------
model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

# ---------------- Loss ----------------
criterion = nn.CrossEntropyLoss()

# ---------------- Optimizer & Scheduler ----------------
if args.model.lower() in ['vit', 'vit_full']:
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
else:
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ---------------- Training Loop ----------------
os.makedirs(args.save_path, exist_ok=True)
best_auc = 0.0

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    scheduler.step()
    print(f"[Epoch {epoch}] Train Loss: {train_loss / len(train_loader):.4f}")

    # ---------------- Validation ----------------
    model.eval()
    preds, probs = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            softmax_out = torch.softmax(outputs, dim=1)

            preds.extend(torch.argmax(softmax_out, dim=1).cpu().numpy())
            probs.extend(softmax_out[:, 1].cpu().numpy())

    acc = accuracy_score(val_labels, preds)
    auc = roc_auc_score(val_labels, probs)
    print(f"[Epoch {epoch}] Val Acc: {acc:.4f} | AUC: {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        save_path = os.path.join(args.save_path, f"{args.model}_best.pth")
        torch.save(
            model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            save_path
        )
        print("✔ Best model saved")

print("✅ Training completed.")
