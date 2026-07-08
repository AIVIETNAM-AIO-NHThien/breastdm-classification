import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import sys

# Import custom data loader và model
from loader_npy import create_dataloaders_npy
from fusionModels import FusionM

# -------------------------------
# Cấu hình dòng lệnh
# -------------------------------
parser = argparse.ArgumentParser(description='LG-CAFN training on BreastDM (.npy dataset)')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--model', type=str, default='fusion', choices=['fusion'], help='model type')
parser.add_argument('--gpu', type=str, default='0', help='GPU id(s)')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')
parser.add_argument('--data-root', type=str, required=True,
                    help='root directory of .npy dataset (e.g., /kaggle/input/breast-dm/cls/img9Se)')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.01, help='L2 regularization')
parser.add_argument('--load-vit', action='store_true', default=True, help='load pretrained ViT weights')
parser.add_argument('--vit-path', type=str, default='./model/vit_base_patch16_224_in21k.pth',
                    help='path to ViT pretrained weights')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save model checkpoints')
parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--warmup-epochs', type=int, default=5, help='number of warmup epochs')
parser.add_argument('--grad-clip', type=float, default=1.0, help='gradient clipping max norm')
parser.add_argument('--seed', type=int, default=42, help='random seed')

args = parser.parse_args()

# -------------------------------
# Set seed
# -------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# -------------------------------
# Thiết bị GPU
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -------------------------------
# Xác định số kênh dựa trên tên thư mục
# -------------------------------
if 'img9Se' in args.data_root:
    in_channels = 9
elif 'img17Se' in args.data_root:
    in_channels = 17
else:
    # Fallback: thử đọc một file .npy để suy ra số kênh
    # (có thể không cần nếu bạn luôn truyền đúng đường dẫn)
    in_channels = 9
    print("Warning: could not determine number of channels from path, defaulting to 9.")

# -------------------------------
# Tạo DataLoader
# -------------------------------
train_loader, val_loader, test_loader = create_dataloaders_npy(
    root_dir=args.data_root,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples:   {len(val_loader.dataset)}")
print(f"Test samples:  {len(test_loader.dataset)}")

# -------------------------------
# Khởi tạo mô hình
# -------------------------------
model = FusionM(num_classes=args.num_class,
                in_c=in_channels,
                load_vit=args.load_vit)
if args.load_vit:
    model.path = args.vit_path

model = model.to(device)

if len(args.gpu.split(',')) > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpu.split(',')))))

# -------------------------------
# Loss function, Optimizer, Scheduler
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# -------------------------------
# Hàm huấn luyện (có warmup + gradient clipping)
# -------------------------------
def train_one_epoch(epoch, model, loader, optimizer, criterion, device, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # Warmup learning rate
    if epoch <= args.warmup_epochs:
        lr = args.lr * (epoch / args.warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        total_loss += loss.item() * data.size(0)
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / total
    acc = 100. * correct / total
    print(f'Train Epoch: {epoch} - Average loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')
    return avg_loss, acc

# -------------------------------
# Hàm validation / test
# -------------------------------
def evaluate(model, loader, criterion, device, target_name='Val'):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            prob = torch.softmax(output, dim=1)[:, 1]
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    avg_loss = total_loss / total
    acc = 100. * correct / total

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    auc = roc_auc_score(all_labels, all_probs)
    if target_name == 'Test':
        print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'], digits=4))

    print(f'{target_name} set: Average loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%, AUC: {auc:.4f}')
    return avg_loss, acc, auc

# -------------------------------
# Vòng lặp chính
# -------------------------------
best_val_acc = 0.0
best_epoch = -1
os.makedirs(args.save_dir, exist_ok=True)

for epoch in range(1, args.epochs + 1):
    print(f'\n===== Epoch {epoch}/{args.epochs} =====')
    train_loss, train_acc = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device, args)
    val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device, 'Val')

    # Scheduler step sau warmup
    if epoch > args.warmup_epochs:
        scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        save_path = os.path.join(args.save_dir, f'best_model_experiment_npy.pth')
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        print(f'Checkpoint saved to {save_path} (val acc: {val_acc:.2f}%)')

print(f'\nTraining finished. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}')

# -------------------------------
# Đánh giá trên tập test với model tốt nhất
# -------------------------------
print('\nLoading best model for test evaluation...')
best_model_path = os.path.join(args.save_dir, f'best_model_experiment_npy.pth')
if os.path.exists(best_model_path):
    model_test = FusionM(num_classes=args.num_class, in_c=in_channels, load_vit=False)
    model_test.load_state_dict(torch.load(best_model_path, map_location=device))
    model_test = model_test.to(device)
    if len(args.gpu.split(',')) > 1:
        model_test = torch.nn.DataParallel(model_test)
    test_loss, test_acc, test_auc = evaluate(model_test, test_loader, criterion, device, 'Test')
else:
    print('Best model not found, evaluating current model.')
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device, 'Test')