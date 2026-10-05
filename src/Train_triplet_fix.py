import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.svm import SVC

# Import data loader và model
from data_loader_triplet import create_dataloaders
from Fusion_triplet_new import FusionM


# -------------------------------
# Hàm tính Sensitivity và Specificity dùng Youden index
# -------------------------------
def calc_sens_spec_youden(all_labels, all_probs):
    """Tính Sensitivity, Specificity tại ngưỡng tối ưu theo Youden index."""
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    J = tpr - fpr
    idx = np.argmax(J)
    opt_thresh = thresholds[idx]
    sens = tpr[idx]
    spec = 1 - fpr[idx]
    preds_opt = (all_probs >= opt_thresh).astype(int)
    cm_opt = confusion_matrix(all_labels, preds_opt)
    return sens, spec, opt_thresh, cm_opt


# -------------------------------
# Cấu hình dòng lệnh
# -------------------------------
parser = argparse.ArgumentParser(description='LG-CAFN training on BreastDM (Only Triplet Loss)')
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
parser.add_argument('--weight-decay', type=float, default=0.0005, help='L2 regularization')
parser.add_argument('--load-vit', action='store_true', default=True, help='load pretrained ViT weights')
parser.add_argument('--vit-path', type=str, default='./model/vit_base_patch16_224_in21k.pth',
                    help='path to ViT pretrained weights')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save model checkpoints')
parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')

# Triplet Loss
parser.add_argument('--triplet-margin', type=float, nargs='+', default=[1.0],
                    help='margin for triplet loss')
parser.add_argument('--embedding-dim', type=int, default=128,
                    help='Dimension of embedding for triplet loss')
parser.add_argument('--svm-C', type=float, default=0.1, help='SVM regularization parameter')

parser.add_argument('--seed', type=int, default=8, help='random seed')
args = parser.parse_args()


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
# Xác định số kênh
# -------------------------------
if args.experiment == 'Exp-1':
    in_channels = 9
elif args.experiment == 'Exp-2':
    in_channels = 17
else:
    raise ValueError('Unknown experiment')

# -------------------------------
# Tạo DataLoader (dùng TripletBatchSampler patient-aware)
# -------------------------------
train_loader, val_loader, test_loader = create_dataloaders(
    root_dir=args.data_root,
    experiment=args.experiment,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    use_triplet=True
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples:   {len(val_loader.dataset)}")
print(f"Test samples:  {len(test_loader.dataset)}")
print(f"Train batches/epoch: {len(train_loader)}")


# -------------------------------
# Semi-hard Triplet Loss (patient-aware)
# -------------------------------
def batch_semihard_triplet_loss(embeddings, labels, patient_ids, margin):
    """
    Triplet loss với điều kiện:
      - Positive: CÙNG class VÀ KHÁC bệnh nhân
      - Negative: KHÁC class
      - Negative chọn theo semi-hard
    """
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    loss = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
    num_triplets = 0
    device = embeddings.device
    arange_all = torch.arange(len(labels), device=device)

    for i in range(len(labels)):
        anchor_label = labels[i]
        anchor_pid = patient_ids[i]

        # Positive: cùng class VÀ khác bệnh nhân
        pos_mask = (
            (labels == anchor_label) &
            (arange_all != i) &
            (patient_ids != anchor_pid)
        )
        # Negative: khác class
        neg_mask = (labels != anchor_label)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            continue

        hardest_pos_dist = pairwise_dist[i][pos_mask].max()
        neg_dists = pairwise_dist[i][neg_mask]
        semi_hard_mask = (neg_dists > hardest_pos_dist) & (neg_dists < hardest_pos_dist + margin)

        if semi_hard_mask.sum() == 0:
            continue

        hardest_semihard_dist = neg_dists[semi_hard_mask].min()
        loss += F.relu(hardest_pos_dist - hardest_semihard_dist + margin)
        num_triplets += 1

    if num_triplets > 0:
        loss = loss / num_triplets
    else:
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return loss


# -------------------------------
# Hàm đánh giá bằng SVM trên embedding (validation)
# -------------------------------
def evaluate_embedding_svm(model, train_loader, val_loader, device, kernel='rbf', C=args.svm_C):
    model.eval()
    train_embs, train_labels = [], []
    val_embs, val_labels = [], []

    with torch.no_grad():
        for data, target, _ in train_loader:      # ← 3 thứ
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            train_embs.append(emb)
            train_labels.append(target.numpy())
        for data, target, _ in val_loader:        # ← 3 thứ
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            val_embs.append(emb)
            val_labels.append(target.numpy())

    X_train = np.concatenate(train_embs)
    y_train = np.concatenate(train_labels)
    X_val = np.concatenate(val_embs)
    y_val = np.concatenate(val_labels)

    clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    print(f'SVM (kernel={kernel}, C={C}) on Val: Acc: {acc*100:.2f}%, AUC: {auc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}')
    return acc * 100, auc, sens, spec


# -------------------------------
# Hàm huấn luyện một epoch (chỉ Triplet Loss, patient-aware)
# -------------------------------
def train_one_epoch(epoch, model, loader, optimizer, device, margin):
    model.train()
    total_loss = 0.0
    total_triplet = 0.0

    for batch_idx, (data, target, patient_ids) in enumerate(loader):   # ← 3 thứ
        data = data.to(device)
        target = target.to(device)
        patient_ids = patient_ids.to(device)                            # ← THÊM

        optimizer.zero_grad()

        # Forward → embedding
        embeddings = model(data, return_embedding=True)

        # Triplet Loss (patient-aware)
        loss = batch_semihard_triplet_loss(embeddings, target, patient_ids, margin)

        total_triplet += loss.item() * data.size(0)

        # Backward
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(loader.dataset)
    avg_triplet = total_triplet / len(loader.dataset)

    print(f'Train Epoch: {epoch} - Avg loss: {avg_loss:.4f}, Triplet: {avg_triplet:.4f}')
    return avg_loss


# -------------------------------
# Hàm đánh giá cuối cùng bằng SVM cho test
# -------------------------------
def evaluate_final_svm(model, train_loader, test_loader, device, kernel='rbf', C=args.svm_C):
    model.eval()
    train_embs, train_labels = [], []
    test_embs, test_labels = [], []

    with torch.no_grad():
        for data, target, _ in train_loader:      # ← 3 thứ
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            train_embs.append(emb)
            train_labels.append(target.numpy())
        for data, target, _ in test_loader:       # ← 3 thứ
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            test_embs.append(emb)
            test_labels.append(target.numpy())

    X_train = np.concatenate(train_embs)
    y_train = np.concatenate(train_labels)
    X_test = np.concatenate(test_embs)
    y_test = np.concatenate(test_labels)

    clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    print(f'Test set (SVM): Accuracy: {acc:.2f}%, AUC: {auc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}')
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'], digits=4))
    print('Confusion Matrix:')
    print(cm)
    return acc, auc


# -------------------------------
# Hàm huấn luyện cho một margin cụ thể
# -------------------------------
def train_with_margin(margin, args, train_loader, val_loader, test_loader, in_channels, device):
    print(f"\n{'='*60}")
    print(f"   BẮT ĐẦU HUẤN LUYỆN VỚI MARGIN = {margin} (Chỉ Triplet Loss, patient-aware)")
    print('='*60)

    # Tạo thư mục lưu riêng cho margin này
    save_dir = os.path.join(args.save_dir, f"margin_{margin}")
    os.makedirs(save_dir, exist_ok=True)

    # Khởi tạo model mới
    model = FusionM(num_classes=args.num_class,
                    in_c=in_channels,
                    load_vit=args.load_vit,
                    embedding_dim=args.embedding_dim)
    if args.load_vit:
        model.path = args.vit_path
    model = model.to(device)
    if len(args.gpu.split(',')) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpu.split(',')))))

    best_val_auc = 0.0
    best_epoch = -1

    # ===== VÒNG LẶP EPOCH =====
    for epoch in range(1, args.epochs + 1):
        print(f'\n===== Epoch {epoch}/{args.epochs} =====')

        # 1. Tính learning rate
        current_lr = max(args.lr * (0.5 ** (epoch // 20)), 1e-5)
        print(f'Learning rate: {current_lr:.6f}')

        # 2. Tạo optimizer (chỉ lấy params có requires_grad=True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(
            trainable_params,
            lr=current_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        # 3. Train một epoch (chỉ Triplet Loss)
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, device, margin)

        # 4. Đánh giá validation bằng SVM
        val_acc, val_auc, val_sens, val_spec = evaluate_embedding_svm(
            model, train_loader, val_loader, device, kernel='rbf', C=args.svm_C
        )
        print(f'Val set (SVM): Accuracy: {val_acc:.2f}%, AUC: {val_auc:.4f}, Sens: {val_sens:.4f}, Spec: {val_spec:.4f}')

        # 5. Lưu best model theo val AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            save_path = os.path.join(save_dir, f'best_model_triplet_only_{args.experiment}.pth')
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f'Checkpoint saved to {save_path} (val AUC: {val_auc:.4f})')

    print(f'\nTraining finished for margin={margin}. Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}')

    # ----- Đánh giá test với best model -----
    print('\nLoading best model for test evaluation...')
    best_model_path = os.path.join(save_dir, f'best_model_triplet_only_{args.experiment}.pth')

    model_test = FusionM(num_classes=args.num_class, in_c=in_channels,
                         load_vit=False, embedding_dim=args.embedding_dim)
    model_test.load_state_dict(torch.load(best_model_path, map_location=device))
    model_test = model_test.to(device)
    if len(args.gpu.split(',')) > 1:
        model_test = torch.nn.DataParallel(model_test)

    evaluate_final_svm(model_test, train_loader, test_loader, device, kernel='rbf', C=args.svm_C)

    # Lưu kết quả vào file log
    log_file = os.path.join(save_dir, 'results.txt')
    with open(log_file, 'w') as f:
        f.write(f"Margin: {margin}\n")
        f.write(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}\n")
    print(f'Results saved to {log_file}')


def main():
    margin_list = args.triplet_margin
    print(f"Will run with margins: {margin_list}")

    for margin in margin_list:
        train_with_margin(margin, args, train_loader, val_loader, test_loader, in_channels, device)


if __name__ == "__main__":
    main()