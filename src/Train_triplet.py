import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- THÊM DÒNG NÀY
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# Import data loader và model
from data_loader_triplet import create_dataloaders
from Fusion_triplet import FusionM

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
parser = argparse.ArgumentParser(description='LG-CAFN training on BreastDM (CE + Batch Hard Triplet)')
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
                    help='Enable triplet loss with batch hard mining (combined with CE)')
parser.add_argument('--triplet-margin', type=float, default=1.0, help='margin for triplet loss')
parser.add_argument('--triplet-weight', type=float, default=1.0,
                    help='Weight of triplet loss')
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
# Tạo DataLoader
# -------------------------------
train_loader, val_loader, test_loader = create_dataloaders(
    root_dir=args.data_root,
    experiment=args.experiment,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    use_triplet=args.use_triplet
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
# Loss functions
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
# Batch Hard Triplet Loss (đã sửa device)
# -------------------------------
def batch_semihard_triplet_loss(embeddings, labels, margin=1.0):
    """
    Semi-hard triplet loss.
    Với mỗi anchor, chọn positive xa nhất (hardest positive).
    Sau đó chọn negative thỏa mãn: d(a,p) < d(a,n) < d(a,p) + margin.
    Nếu không có negative nào thỏa mãn, bỏ qua anchor đó.
    """
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)   # (B, B)
    loss = 0.0
    num_triplets = 0
    device = embeddings.device

    for i in range(len(labels)):
        anchor_label = labels[i]
        pos_mask = (labels == anchor_label) & (torch.arange(len(labels), device=device) != i)
        neg_mask = (labels != anchor_label)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        # Hardest positive (xa nhất)
        hardest_pos_dist = pairwise_dist[i][pos_mask].max()

        # Lấy tất cả khoảng cách tới negative
        neg_dists = pairwise_dist[i][neg_mask]

        # Semi-hard condition: d(a,p) < d(a,n) < d(a,p) + margin
        semi_hard_mask = (neg_dists > hardest_pos_dist) & (neg_dists < hardest_pos_dist + margin)

        if semi_hard_mask.sum() == 0:
            continue   # không có negative semi-hard, bỏ qua anchor này

        # Trong các semi-hard negative, chọn cái gần nhất (hardest semi-hard)
        hardest_semihard_dist = neg_dists[semi_hard_mask].min()

        loss += F.relu(hardest_pos_dist - hardest_semihard_dist + margin)
        num_triplets += 1

    if num_triplets > 0:
        loss = loss / num_triplets
    return loss
# -------------------------------
# Hàm huấn luyện một epoch
# -------------------------------
def train_one_epoch(epoch, model, loader, optimizer, criterion_ce, criterion_triplet, device, args):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_triplet = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Luôn lấy logits (cho CE và đánh giá)
        logits = model(data)
        loss_ce = criterion_ce(logits, target)
        _, pred = logits.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        loss = loss_ce

        if args.use_triplet:
            embeddings = model(data, return_embedding=True)
            loss_triplet = batch_semihard_triplet_loss(embeddings, target, args.triplet_margin)
            loss = loss_ce + args.triplet_weight * loss_triplet
            total_triplet += loss_triplet.item() * data.size(0)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        total_ce += loss_ce.item() * data.size(0)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / total
    avg_ce = total_ce / total
    avg_triplet = total_triplet / total if args.use_triplet else 0.0
    acc = 100. * correct / total
    print(f'Train Epoch: {epoch} - Avg loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, Triplet: {avg_triplet:.4f}, Accuracy: {acc:.2f}%')
    return avg_loss, acc

# -------------------------------
# Hàm validation / test
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
            output = model(data)
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
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, optimizer,
                                                criterion_ce, criterion_triplet, device, args)
    else:
        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, optimizer,
                                                criterion_ce, criterion_triplet, device, args)

    val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion_ce, device, 'Val')
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        save_path = os.path.join(args.save_dir, f'best_model_triplet_{args.experiment}.pth')
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
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