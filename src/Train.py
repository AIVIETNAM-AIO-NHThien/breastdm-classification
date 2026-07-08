import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import sys

# Import custom data loader and model
from data_loader_original import create_dataloaders   # hoặc đổi tên thành get_breast_loaders nếu bạn đã định nghĩa
from fusionModels import FusionM                          # file Fusion.py đã được chỉnh sửa

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
parser.add_argument('--load-vit', action='store_true', default=True, help='load pretrained ViT weights')
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
# Xác định số kênh dựa trên experiment
# -------------------------------
if args.experiment == 'Exp-1':
    in_channels = 9
elif args.experiment == 'Exp-2':
    in_channels = 17
else:
    raise ValueError('Unknown experiment')

# -------------------------------
# Tạo DataLoader
# -------------------------------
# create_dataloaders trả về (train_loader, val_loader, test_loader)
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
# Khởi tạo mô hình
# -------------------------------
model = FusionM(num_classes=args.num_class,
                in_c=in_channels,
                load_vit=args.load_vit)
# Nếu load_vit, model sẽ tự load pretrained trong __init__ nếu file tồn tại.
# Bạn có thể cập nhật đường dẫn trong FusionM hoặc truyền vào.
# Ở đây ta có thể set lại model.path nếu cần.
if args.load_vit:
    model.path = args.vit_path   # cập nhật đường dẫn file pretrained
    # Gọi lại hàm load nếu chưa load (nếu FusionM đã load trong __init__ thì không cần)
    # Nhưng để chắc chắn, ta có thể gọi _load_pretrained_vit() lần nữa.
    # Tuy nhiên FusionM hiện tại đã load trong __init__ nếu load_vit=True.
    # Để an toàn, nếu bạn đã set load_vit=False ở trên mà vẫn muốn load, cần chỉnh logic.
    # Ở đây ta dùng load_vit=True và đảm bảo file tồn tại.

model = model.to(device)

# Sử dụng DataParallel nếu có nhiều GPU (tùy chọn)
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

# Scheduler: giảm lr khi val loss không cải thiện sau 10 epoch
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# -------------------------------
# Hàm huấn luyện
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

            # Lưu cho AUC
            prob = torch.softmax(output, dim=1)[:, 1]   # xác suất lớp malignant
            all_probs.append(prob.cpu().numpy())
            all_preds.append(pred.cpu().numpy())
            all_labels.append(target.cpu().numpy())

    avg_loss = total_loss / total
    acc = 100. * correct / total

    # Tính AUC và các chỉ số khác
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    auc = roc_auc_score(all_labels, all_probs)
    # In classification report (dành cho test chi tiết hơn, có thể chỉ in cho val)
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
    train_loss, train_acc = train_one_epoch(epoch, model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device, 'Val')

    # Scheduler step dựa trên validation loss
    scheduler.step(val_loss)

    # Lưu checkpoint tốt nhất dựa trên accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        # Lưu model state dict
        save_path = os.path.join(args.save_dir, f'best_model_experiment_{args.experiment}.pth')
        # Nếu dùng DataParallel, cần lưu model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        print(f'Checkpoint saved to {save_path} (val acc: {val_acc:.2f}%)')

    # Có thể thêm early stopping nếu muốn, nhưng yêu cầu không bắt buộc
    # early_stopping(val_loss, model) ...

print(f'\nTraining finished. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}')

# -------------------------------
# Đánh giá trên tập test với model tốt nhất
# -------------------------------
print('\nLoading best model for test evaluation...')
best_model_path = os.path.join(args.save_dir, f'best_model_experiment_{args.experiment}.pth')
if os.path.exists(best_model_path):
    # Khởi tạo lại model và load trọng số
    model_test = FusionM(num_classes=args.num_class, in_c=in_channels, load_vit=False)
    model_test.load_state_dict(torch.load(best_model_path, map_location=device))
    model_test = model_test.to(device)
    if len(args.gpu.split(',')) > 1:
        model_test = torch.nn.DataParallel(model_test)
    test_loss, test_acc, test_auc = evaluate(model_test, test_loader, criterion, device, 'Test')
else:
    print('Best model not found, evaluating current model.')
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion, device, 'Test')