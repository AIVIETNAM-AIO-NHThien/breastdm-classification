import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import sys
import random
import math

from data_loader_original import create_dataloaders
from Fusion_inflate import FusionM


# -------------------------------
# Early Stopping (giống tác giả)
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=20, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# -------------------------------
# Seed
# -------------------------------
def set_seed(seed=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(4)

# -------------------------------
# Cấu hình dòng lệnh
# -------------------------------
parser = argparse.ArgumentParser(description='LG-CAFN training on BreastDM')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--model', type=str, default='fusion', choices=['fusion'], help='model type')
parser.add_argument('--gpu', type=str, default='0', help='GPU id(s)')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')
parser.add_argument('--experiment', type=str, default='Exp-1', choices=['Exp-1', 'Exp-2'],
                    help='Experiment type: Exp-1 (9 channels) or Exp-2 (17 channels)')
parser.add_argument('--data-root', type=str, required=True, help='root directory containing train/val/test folders')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.01, help='L2 regularization')
parser.add_argument('--load-vit', action='store_true', default=False, help='load pretrained ViT weights')
parser.add_argument('--vit-path', type=str, default='./model/vit_base_patch16_224_in21k.pth',
                    help='path to ViT pretrained weights')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save model checkpoints')
parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')

args = parser.parse_args()

# -------------------------------
# Thiết bị GPU
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -------------------------------
# Số kênh
# -------------------------------
if args.experiment == 'Exp-1':
    in_channels = 9
elif args.experiment == 'Exp-2':
    in_channels = 17
else:
    raise ValueError('Unknown experiment')

# -------------------------------
# DataLoader
# -------------------------------
train_loader, val_loader, test_loader = create_dataloaders(
    root_dir=args.data_root,
    experiment=args.experiment,
    batch_size=args.batch_size,
    num_workers=args.num_workers
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples:   {len(val_loader.dataset)}")
print(f"Test samples:  {len(test_loader.dataset)}")

# -------------------------------
# Model
# -------------------------------
model = FusionM(
    num_classes=args.num_class,
    in_c=in_channels,
    load_vit=args.load_vit,
    vit_path=args.vit_path if args.load_vit else None
)
model = model.to(device)

if len(args.gpu.split(',')) > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpu.split(',')))))

# -------------------------------
# Loss, Optimizer (SGD không scheduler – tự cập nhật lr)
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

# -------------------------------
# Hàm tính Sensitivity và Specificity dùng Youden index (giống tác giả)
# -------------------------------
def calc_sens_spec_youden(all_labels, all_probs):
    """Tính Sensitivity, Specificity tại ngưỡng tối ưu theo Youden index."""
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    J = tpr - fpr
    idx = np.argmax(J)
    opt_thresh = thresholds[idx]
    sens = tpr[idx]
    spec = 1 - fpr[idx]
    # Tạo pred dựa trên ngưỡng tối ưu để tính confusion matrix
    preds_opt = (all_probs >= opt_thresh).astype(int)
    cm_opt = confusion_matrix(all_labels, preds_opt)
    return sens, spec, opt_thresh, cm_opt

# -------------------------------
# Train một epoch
# -------------------------------
def train_one_epoch(epoch, model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
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
# Hàm đánh giá (dùng Youden index để tính Sens/Spec)
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

            prob = torch.softmax(output, dim=1)[:, 1]   # xác suất lớp malignant
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    avg_loss = total_loss / total
    acc = 100. * correct / total

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    auc = roc_auc_score(all_labels, all_probs)

    # Tính Sensitivity / Specificity bằng Youden index (giống tác giả)
    sens_youden, spec_youden, opt_thresh, cm_youden = calc_sens_spec_youden(all_labels, all_probs)

    if target_name == 'Test':
        print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant'], digits=4))

    print(f'{target_name} set: Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, AUC: {auc:.4f}, '
          f'Sensitivity: {sens_youden:.4f}, Specificity: {spec_youden:.4f}')
    print(f'Optimal threshold (Youden): {opt_thresh:.4f}')
    print('Confusion Matrix (at Youden threshold):')
    print(cm_youden)

    return avg_loss, acc, auc, sens_youden, spec_youden

# -------------------------------
# Vòng lặp chính
# -------------------------------
best_auc = 0.0
best_epoch = -1
os.makedirs(args.save_dir, exist_ok=True)

early_stopping = EarlyStopping(patience=20, verbose=True)

for epoch in range(1, args.epochs + 1):
    print(f'\n===== Epoch {epoch}/{args.epochs} =====')

    # Cập nhật learning rate theo công thức của tác giả
    current_lr = max(args.lr * (0.1 ** (epoch // 10)), 1e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    print(f'Learning rate: {current_lr:.6f}')

    train_loss, train_acc = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_auc, val_sens, val_spec = evaluate(model, val_loader, criterion, device, 'Val')

    # Early stopping dựa trên validation loss
    early_stopping(val_loss, model)
    if early_stopping.early_stop:
        print('Early stopping triggered.')
        break

    # Lưu model nếu AUC validation tốt nhất (giống tác giả lưu theo AUC)
    if val_auc > best_auc:
        best_auc = val_auc
        best_epoch = epoch
        save_path = os.path.join(args.save_dir, f'best_model_epoch_{epoch}_auc_{val_auc:.4f}.pth')
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        print(f'Checkpoint saved (val AUC: {val_auc:.4f})')

print(f'\nTraining finished. Best validation AUC: {best_auc:.4f} at epoch {best_epoch}')

# -------------------------------
# Đánh giá trên test với model tốt nhất
# -------------------------------
print('\nLoading best model for test evaluation...')
# Tìm file model có AUC cao nhất đã lưu (có thể load trực tiếp best_epoch)
best_model_path = os.path.join(args.save_dir, f'best_model_epoch_{best_epoch}_auc_{best_auc:.4f}.pth')
if os.path.exists(best_model_path):
    model_test = FusionM(num_classes=args.num_class, in_c=in_channels, load_vit=False)
    model_test.load_state_dict(torch.load(best_model_path, map_location=device))
    model_test = model_test.to(device)
    if len(args.gpu.split(',')) > 1:
        model_test = torch.nn.DataParallel(model_test)
    test_loss, test_acc, test_auc, test_sens, test_spec = evaluate(model_test, test_loader, criterion, device, 'Test')
    print(f'\n✅ Final Test Results:')
    print(f'   Accuracy:  {test_acc:.2f}%')
    print(f'   AUC:       {test_auc:.4f}')
    print(f'   Sensitivity: {test_sens:.4f}')
    print(f'   Specificity: {test_spec:.4f}')
else:
    print('Best model not found, evaluating current model.')
    test_loss, test_acc, test_auc, test_sens, test_spec = evaluate(model, test_loader, criterion, device, 'Test')