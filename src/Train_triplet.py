import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Import data loader và model
from data_loader_triplet import create_dataloaders   # đã được cập nhật với TripletBatchSampler
from Fusion_triplet import FusionM                    # model đã có embedding head

# -------------------------------
# Seed
# -------------------------------
def set_seed(seed=8):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(8)

# -------------------------------
# Cấu hình dòng lệnh
# -------------------------------
parser = argparse.ArgumentParser(description='LG-CAFN training on BreastDM (with optional triplet loss)')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--model', type=str, default='fusion', choices=['fusion'], help='model type')
parser.add_argument('--gpu', type=str, default='0', help='GPU id(s)')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')
parser.add_argument('--experiment', type=str, default='Exp-1', choices=['Exp-1', 'Exp-2'],
                    help='Experiment type')
parser.add_argument('--data-root', type=str, required=True, help='root directory containing train/val/test folders')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.01, help='L2 regularization')
parser.add_argument('--load-vit', action='store_true', default=True, help='load pretrained ViT weights')
parser.add_argument('--vit-path', type=str, default='./model/vit_base_patch16_224_in21k.pth',
                    help='path to ViT pretrained weights')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save model checkpoints')
parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')

# Tham số cho triplet loss
parser.add_argument('--use-triplet', action='store_true', default=False,
                    help='Use triplet loss instead of cross-entropy')
parser.add_argument('--triplet-margin', type=float, default=1.0, help='margin for triplet loss')
parser.add_argument('--triplet-weight', type=float, default=1.0,
                    help='Weight of triplet loss when combined with CE (if both used)')
parser.add_argument('--embedding-dim', type=int, default=128,
                    help='Dimension of embedding for triplet loss')

args = parser.parse_args()

# -------------------------------
# Thiết bị GPU
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# -------------------------------
# Xác định số kênh
# -------------------------------
if args.experiment == 'Exp-1':
    in_channels = 9
elif args.experiment == 'Exp-2':
    in_channels = 17
else:
    raise ValueError('Unknown experiment')

# -------------------------------
# Tạo DataLoader (với triplet sampler nếu cần)
# -------------------------------
train_loader, val_loader, test_loader = create_dataloaders(
    root_dir=args.data_root,
    experiment=args.experiment,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    use_triplet=args.use_triplet   # truyền vào để dùng TripletBatchSampler
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples:   {len(val_loader.dataset)}")
print(f"Test samples:  {len(test_loader.dataset)}")

# -------------------------------
# Khởi tạo mô hình
# -------------------------------
model = FusionM(num_classes=args.num_class,
                in_c=in_channels,
                load_vit=args.load_vit,
                embedding_dim=args.embedding_dim)
if args.load_vit:
    model.path = args.vit_path

model = model.to(device)

if len(args.gpu.split(',')) > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpu.split(',')))))

# -------------------------------
# Loss function
# -------------------------------
criterion_ce = nn.CrossEntropyLoss()
criterion_triplet = nn.TripletMarginLoss(margin=args.triplet_margin, p=2.0)

# -------------------------------
# Optimizer & Scheduler
# -------------------------------
optimizer = optim.SGD(model.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# -------------------------------
# Hàm huấn luyện
# -------------------------------
def train_one_epoch(epoch, model, loader, optimizer, criterion_ce, criterion_triplet, device, args):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if args.use_triplet:
            # Lấy embedding
            embeddings = model(data, return_embedding=True)   # (B, embedding_dim)

            # Tạo triplet ngay trong batch
            # Yêu cầu batch có ít nhất 2 mẫu mỗi lớp (đảm bảo bởi TripletBatchSampler)
            loss = 0.0
            # Lấy mask cho từng lớp
            for cls in range(args.num_class):
                mask = (target == cls)
                if mask.sum() < 2:
                    continue  # không đủ mẫu để tạo triplet cho lớp này
                # Anchor: tất cả mẫu của lớp cls
                anchor_emb = embeddings[mask]
                # Positive: chọn ngẫu nhiên các mẫu khác trong cùng lớp (có thể lấy toàn bộ rồi tính)
                # Với mỗi anchor, chọn một positive khác anchor
                # Ở đây ta dùng cách đơn giản: với mỗi anchor, chọn một positive bất kỳ khác trong batch (cùng lớp)
                # và một negative từ lớp còn lại.
                other_cls = 1 - cls
                neg_mask = (target == other_cls)
                if neg_mask.sum() == 0:
                    continue
                neg_emb = embeddings[neg_mask]

                # Tính all triplet loss cho lớp này: mỗi anchor với mỗi positive (khác anchor) và mỗi negative
                # Để đơn giản, ta lấy mỗi anchor, một positive ngẫu nhiên, một negative ngẫu nhiên
                for i in range(anchor_emb.size(0)):
                    anchor = anchor_emb[i].unsqueeze(0)
                    # Chọn một positive khác anchor (nếu có >1 mẫu)
                    pos_indices = torch.where(mask)[0]
                    # loại bỏ chính nó
                    pos_candidates = [idx for idx in pos_indices if idx != torch.where(mask)[0][i]]
                    if len(pos_candidates) == 0:
                        continue
                    pos_idx = random.choice(pos_candidates)
                    positive = embeddings[pos_idx].unsqueeze(0)
                    # Chọn một negative ngẫu nhiên
                    neg_idx = random.choice(torch.where(neg_mask)[0])
                    negative = embeddings[neg_idx].unsqueeze(0)
                    loss += criterion_triplet(anchor, positive, negative)
            loss = loss / (mask.sum() + 1e-8)   # trung bình
        else:
            # Cross-entropy
            output = model(data)
            loss = criterion_ce(output, target)
            # Tính accuracy cho CE
            _, pred = output.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(loader.dataset)
    # Khi dùng triplet, không có accuracy để tính, ta có thể bỏ qua hoặc in loss
    if args.use_triplet:
        print(f'Train Epoch: {epoch} - Average Triplet Loss: {avg_loss:.4f}')
        return avg_loss, None   # không có accuracy
    else:
        acc = 100. * correct / total
        print(f'Train Epoch: {epoch} - Average loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')
        return avg_loss, acc

# -------------------------------
# Hàm validation / test (luôn dùng classifier)
# -------------------------------
def evaluate(model, loader, criterion_ce, device, target_name='Val'):
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
            output = model(data)   # dùng logits
            loss = criterion_ce(output, target)

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
    if args.use_triplet:
        train_loss, _ = train_one_epoch(epoch, model, train_loader, optimizer,
                                        criterion_ce, criterion_triplet, device, args)
    else:
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, optimizer,
                                                criterion_ce, criterion_triplet, device, args)

    val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion_ce, device, 'Val')
    scheduler.step()

    # Lưu checkpoint tốt nhất dựa trên accuracy (nếu dùng triplet vẫn tính accuracy qua evaluate)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        save_path = os.path.join(args.save_dir, f'best_model_triplet_{args.experiment}.pth')
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        print(f'Checkpoint saved to {save_path} (val acc: {val_acc:.2f}%)')

print(f'\nTraining finished. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}')

# -------------------------------
# Đánh giá trên tập test
# -------------------------------
print('\nLoading best model for test evaluation...')
best_model_path = os.path.join(args.save_dir, f'best_model_triplet_{args.experiment}.pth')
if os.path.exists(best_model_path):
    model_test = FusionM(num_classes=args.num_class, in_c=in_channels,
                         load_vit=False, embedding_dim=args.embedding_dim)
    model_test.load_state_dict(torch.load(best_model_path, map_location=device))
    model_test = model_test.to(device)
    if len(args.gpu.split(',')) > 1:
        model_test = torch.nn.DataParallel(model_test)
    test_loss, test_acc, test_auc = evaluate(model_test, test_loader, criterion_ce, device, 'Test')
else:
    print('Best model not found, evaluating current model.')
    test_loss, test_acc, test_auc = evaluate(model, test_loader, criterion_ce, device, 'Test')