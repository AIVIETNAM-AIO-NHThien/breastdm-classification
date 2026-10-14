import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
import json
import joblib                                    # ⚠️ THÊM: để lưu SVM
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.svm import SVC

# Import data loader và model
from data_loader_triplet import create_dataloaders
from Fusion_triplet_new import FusionM


# ============================================================
# HELPER: Tính metrics (dùng chung)
# ============================================================
def compute_metrics(y_true, y_pred, probs=None):
    """Tính Acc, Sens, Spec, AUC cho 1 cấp độ đánh giá."""
    acc = accuracy_score(y_true, y_pred) * 100
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        TN, FP, FN, TP = cm.ravel()
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    else:
        sens = spec = 0.0
    
    auc = 0.0
    if probs is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, probs)
        except ValueError:
            auc = 0.0
    
    return {
        'acc': float(acc),
        'sens': float(sens),
        'spec': float(spec),
        'auc': float(auc),
        'cm': cm
    }


# ============================================================
# HELPER: Tính patient-level metrics — CHỈ MAJORITY VOTING
# ============================================================
def compute_patient_level_metrics_majority(y_pred, y_proba, y_true, patient_ids):
    """
    Tổng hợp predictions từ slice-level lên patient-level.
    CHỈ dùng MAJORITY VOTING.
    
    Args:
        y_pred: (N,) slice-level predictions
        y_proba: (N,) slice-level probabilities cho class Malignant
        y_true: (N,) slice-level true labels
        patient_ids: (N,) patient ID cho mỗi slice
    
    Returns:
        metrics: dict cho majority voting
        pids: list patient IDs
        y_patient: (P,) nhãn patient-level
    """
    # Nhóm theo patient
    patient_votes = defaultdict(list)
    patient_true = {}
    
    for pred, true, pid in zip(y_pred, y_true, patient_ids):
        patient_votes[pid].append(int(pred))
        patient_true[pid] = int(true)
    
    pids = list(patient_votes.keys())
    y_patient = np.array([patient_true[pid] for pid in pids])
    
    # ===== Majority voting =====
    probs_maj = np.array([np.mean(patient_votes[pid]) for pid in pids])
    preds_maj = (probs_maj >= 0.5).astype(int)
    
    return {
        'majority': compute_metrics(y_patient, preds_maj, probs_maj)
    }, pids, y_patient


# ============================================================
# Hàm tính Sensitivity và Specificity dùng Youden index
# ============================================================
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


# ============================================================
# Cấu hình dòng lệnh
# ============================================================
parser = argparse.ArgumentParser(description='LG-CAFN training on BreastDM (CE + Triplet or Only Triplet)')
parser.add_argument('--batch-size', type=int, default=16, help='batch size')
parser.add_argument('--model', type=str, default='fusion', choices=['fusion'], help='model type')
parser.add_argument('--gpu', type=str, default='0', help='GPU id(s)')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')
parser.add_argument('--experiment', type=str, default='Exp-1', choices=['Exp-1', 'Exp-2'],
                    help='Experiment type')
parser.add_argument('--data-root', type=str, required=True, help='root directory containing train/val/test folders')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.05, help='L2 regularization')
parser.add_argument('--load-vit', action='store_true', default=True, help='load pretrained ViT weights')
parser.add_argument('--vit-path', type=str, default='./model/vit_base_patch16_224_in21k.pth',
                    help='path to ViT pretrained weights')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save model checkpoints')
parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')

# Tham số cho triplet loss
parser.add_argument('--triplet-margin', type=float, nargs='+', default=[1.0],
                    help='margin for triplet loss, can provide multiple values')
parser.add_argument('--use-triplet', action='store_true', default=False,
                    help='Enable triplet loss (combined with CE if --only-triplet not set)')
parser.add_argument('--only-triplet', action='store_true', default=False,
                    help='Use ONLY triplet loss (no cross-entropy)')
parser.add_argument('--triplet-weight', type=float, default=1.0,
                    help='Weight of triplet loss (when combined with CE)')
parser.add_argument('--embedding-dim', type=int, default=128,
                    help='Dimension of embedding for triplet loss')
parser.add_argument('--eval-embedding', action='store_true', default=False,
                    help='Evaluate val accuracy/AUC using SVM on embeddings when only triplet')
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
# Tạo DataLoader
# -------------------------------
train_loader, val_loader, test_loader = create_dataloaders(
    root_dir=args.data_root,
    experiment=args.experiment,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    use_triplet=(args.use_triplet or args.only_triplet)
)

print(f"Train samples: {len(train_loader.dataset)}")
print(f"Val samples:   {len(val_loader.dataset)}")
print(f"Test samples:  {len(test_loader.dataset)}")


# -------------------------------
# Semi-hard triplet loss
# -------------------------------
def batch_semihard_triplet_loss(embeddings, labels, margin):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    loss = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
    num_triplets = 0
    device_ = embeddings.device

    for i in range(len(labels)):
        anchor_label = labels[i]
        pos_mask = (labels == anchor_label) & (torch.arange(len(labels), device=device_) != i)
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
# Hàm đánh giá CE (dùng Youden)
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
# Hàm đánh giá SVM trên embedding (val)
# -------------------------------
def evaluate_embedding_svm(model, train_loader, val_loader, device, kernel='rbf', C=args.svm_C):
    """
    Đánh giá SVM trên val set.
    """
    model.eval()
    train_embs, train_labels = [], []
    val_embs, val_labels = [], []
    val_patient_ids = []

    with torch.no_grad():
        for data, target in train_loader:
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            train_embs.append(emb)
            train_labels.append(target.numpy())

        slice_offset = 0
        for data, target in val_loader:
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            val_embs.append(emb)
            val_labels.append(target.numpy())

            bs = data.size(0)
            for i in range(bs):
                idx = slice_offset + i
                if idx < len(val_loader.dataset.samples):
                    pid = val_loader.dataset.samples[idx]['patient_dir']
                    val_patient_ids.append(pid)
            slice_offset += bs

    X_train = np.concatenate(train_embs)
    y_train = np.concatenate(train_labels)
    X_val = np.concatenate(val_embs)
    y_val = np.concatenate(val_labels)
    val_patient_ids = np.array(val_patient_ids)

    clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    # ===== SLICE-LEVEL =====
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    print(f'SVM (kernel={kernel}, C={C}) on Val [SLICE]: '
          f'Acc: {acc * 100:.2f}%, AUC: {auc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}')

    # ===== PATIENT-LEVEL (MAJORITY ONLY) =====
    if len(np.unique(val_patient_ids)) > 1:
        pl_metrics, pids, y_patient = compute_patient_level_metrics_majority(
            y_pred, y_proba, y_val, val_patient_ids
        )
        print(f'  → [PATIENT-LEVEL] Số patients: {len(pids)}')
        m = pl_metrics['majority']
        print(f'     majority  : Acc={m["acc"]:.2f}%, AUC={m["auc"]:.4f}, '
              f'Sens={m["sens"]:.4f}, Spec={m["spec"]:.4f}')

    return acc * 100, auc, sens, spec


# -------------------------------
# Hàm huấn luyện một epoch
# -------------------------------
def train_one_epoch(epoch, model, loader, optimizer, criterion_ce, criterion_triplet, device, args, margin):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_triplet = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        if args.only_triplet:
            embeddings = model(data, return_embedding=True)
            loss = batch_semihard_triplet_loss(embeddings, target, margin)
            total_triplet += loss.item() * data.size(0)
        else:
            logits = model(data)
            loss_ce = criterion_ce(logits, target)
            _, pred = logits.max(1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            loss = loss_ce
            total_ce += loss_ce.item() * data.size(0)

            if args.use_triplet:
                embeddings = model(data, return_embedding=True)
                loss_triplet = batch_semihard_triplet_loss(embeddings, target, margin)
                loss = loss_ce + args.triplet_weight * loss_triplet
                total_triplet += loss_triplet.item() * data.size(0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.size(0)

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loader.dataset)} '
                  f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = total_loss / len(loader.dataset)
    avg_ce = total_ce / len(loader.dataset) if not args.only_triplet else 0.0
    avg_triplet = total_triplet / len(loader.dataset)

    if args.only_triplet:
        print(f'Train Epoch: {epoch} - Avg loss: {avg_loss:.4f}, Triplet: {avg_triplet:.4f}')
        return avg_loss, None
    else:
        acc = 100. * correct / total
        print(f'Train Epoch: {epoch} - Avg loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, '
              f'Triplet: {avg_triplet:.4f}, Accuracy: {acc:.2f}%')
        return avg_loss, acc


# -------------------------------
# Hàm đánh giá cuối cùng bằng SVM cho test — LƯU SVM
# -------------------------------
def evaluate_final_svm(model, train_loader, test_loader, device,
                       kernel='rbf', C=args.svm_C, save_svm_path=None):
    """
    Đánh giá cuối cùng trên test set.
    ⚠️ ĐÃ SỬA: Lưu SVM vào file .pkl.
    ⚠️ CHỈ dùng majority voting.
    """
    model.eval()
    train_embs, train_labels = [], []
    test_embs, test_labels = [], []
    test_patient_ids = []

    with torch.no_grad():
        # ===== Trích embedding train =====
        for data, target in train_loader:
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            train_embs.append(emb)
            train_labels.append(target.numpy())

        # ===== Trích embedding test + thu thập patient IDs =====
        slice_offset = 0
        for data, target in test_loader:
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            test_embs.append(emb)
            test_labels.append(target.numpy())

            bs = data.size(0)
            for i in range(bs):
                idx = slice_offset + i
                if idx < len(test_loader.dataset.samples):
                    pid = test_loader.dataset.samples[idx]['patient_dir']
                    test_patient_ids.append(pid)
            slice_offset += bs

    X_train = np.concatenate(train_embs)
    y_train = np.concatenate(train_labels)
    X_test = np.concatenate(test_embs)
    y_test = np.concatenate(test_labels)
    test_patient_ids = np.array(test_patient_ids)

    print(f"\n[SVM-fit] Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"[Patient] Số slices: {len(test_patient_ids)}, "
          f"Số patients: {len(np.unique(test_patient_ids))}")

    clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    clf.fit(X_train, y_train)

    # ===== ⚠️ LƯU SVM =====
    if save_svm_path is not None:
        joblib.dump(clf, save_svm_path)
        print(f'✅ SVM saved to {save_svm_path}')
        print(f'   Support vectors: {len(clf.support_vectors_)}')
        print(f'   File size: {os.path.getsize(save_svm_path) / 1024:.2f} KB')

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # ============ SLICE-LEVEL ============
    print("\n" + "=" * 60)
    print("SLICE-LEVEL METRICS")
    print("=" * 60)
    slice_metrics = compute_metrics(y_test, y_pred, y_proba)
    print(f"Accuracy:    {slice_metrics['acc']:.2f}%")
    print(f"AUC:         {slice_metrics['auc']:.4f}")
    print(f"Sensitivity: {slice_metrics['sens']:.4f}")
    print(f"Specificity: {slice_metrics['spec']:.4f}")
    print("Confusion Matrix:")
    print(slice_metrics['cm'])
    print()
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'], digits=4))

    # ============ PATIENT-LEVEL (MAJORITY ONLY) ============
    print("\n" + "=" * 60)
    print("PATIENT-LEVEL METRICS (MAJORITY VOTING)")
    print("=" * 60)

    patient_metrics, pids, y_patient = compute_patient_level_metrics_majority(
        y_pred, y_proba, y_test, test_patient_ids
    )

    print(f"Số patients: {len(pids)}")
    print(f"Nhãn: {np.bincount(y_patient)} (Benign, Malignant)")
    print()

    m = patient_metrics['majority']
    print(f"--- MAJORITY ---")
    print(f"  Accuracy:    {m['acc']:.2f}%")
    print(f"  AUC:         {m['auc']:.4f}")
    print(f"  Sensitivity: {m['sens']:.4f}")
    print(f"  Specificity: {m['spec']:.4f}")
    print()

    # ============ SO SÁNH SLICE vs PATIENT ============
    print("=" * 60)
    print("SO SÁNH SLICE-LEVEL VÀ PATIENT-LEVEL")
    print("=" * 60)
    print(f"{'Cấp độ':<20} {'Acc':>10} {'AUC':>10} {'Sens':>10} {'Spec':>10}")
    print("-" * 60)
    print(f"{'Slice-level':<20} {slice_metrics['acc']:>9.2f}% {slice_metrics['auc']:>10.4f} "
          f"{slice_metrics['sens']:>10.4f} {slice_metrics['spec']:>10.4f}")
    print(f"{'Patient-majority':<20} {m['acc']:>9.2f}% {m['auc']:>10.4f} "
          f"{m['sens']:>10.4f} {m['spec']:>10.4f}")

    # Trả về dict
    return {
        'slice_level': {
            'acc': slice_metrics['acc'],
            'auc': slice_metrics['auc'],
            'sens': slice_metrics['sens'],
            'spec': slice_metrics['spec'],
        },
        'patient_level': {
            'majority': {
                'acc': m['acc'],
                'auc': m['auc'],
                'sens': m['sens'],
                'spec': m['spec'],
            }
        }
    }


# -------------------------------
# Hàm huấn luyện cho một margin cụ thể
# -------------------------------
def train_with_margin(margin, args, train_loader, val_loader, test_loader, in_channels, device):
    print(f"\n{'=' * 60}")
    print(f"   BẮT ĐẦU HUẤN LUYỆN VỚI MARGIN = {margin}")
    print('=' * 60)

    save_dir = os.path.join(args.save_dir, f"margin_{margin}")
    os.makedirs(save_dir, exist_ok=True)

    # Khởi tạo model
    model = FusionM(num_classes=args.num_class,
                    in_c=in_channels,
                    load_vit=args.load_vit,
                    embedding_dim=args.embedding_dim)
    if args.load_vit:
        model.path = args.vit_path
    model = model.to(device)
    if len(args.gpu.split(',')) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpu.split(',')))))

    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    best_val_auc = 0.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        print(f'\n===== Epoch {epoch}/{args.epochs} =====')

        # Cập nhật learning rate
        current_lr = max(args.lr * (0.5 ** (epoch // 20)), 1e-5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        print(f'Learning rate: {current_lr:.6f}')

        train_loss, train_acc = train_one_epoch(epoch, model, train_loader, optimizer,
                                                criterion_ce, None, device, args, margin)

        if args.only_triplet:
            if args.eval_embedding:
                val_acc, val_auc, val_sens, val_spec = evaluate_embedding_svm(
                    model, train_loader, val_loader, device, kernel='rbf', C=args.svm_C
                )
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    save_path = os.path.join(save_dir, f'best_model_triplet_only_{args.experiment}.pth')
                    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, save_path)
                    print(f'Checkpoint saved to {save_path} (val AUC: {val_auc:.4f})')
            else:
                if 'best_val_loss' not in locals():
                    best_val_loss = float('inf')
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    save_path = os.path.join(save_dir, f'best_model_triplet_only_{args.experiment}.pth')
                    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                    torch.save(state_dict, save_path)
                    print(f'Checkpoint saved (train loss: {train_loss:.4f})')
        else:
            val_loss, val_acc, val_auc, val_sens, val_spec = evaluate(model, val_loader, criterion_ce, device, 'Val')
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                save_path = os.path.join(save_dir, f'best_model_ce_{args.experiment}.pth')
                state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
                torch.save(state_dict, save_path)
                print(f'Checkpoint saved to {save_path} (val AUC: {val_auc:.4f})')

    print(f'\nTraining finished for margin={margin}. Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}')

    # ----- Đánh giá test -----
    print('\nLoading best model for test evaluation...')
    test_results = None

    if args.only_triplet:
        best_model_path = os.path.join(save_dir, f'best_model_triplet_only_{args.experiment}.pth')
        model_test = FusionM(num_classes=args.num_class, in_c=in_channels,
                             load_vit=False, embedding_dim=args.embedding_dim)
        model_test.load_state_dict(torch.load(best_model_path, map_location=device))
        model_test = model_test.to(device)
        if len(args.gpu.split(',')) > 1:
            model_test = torch.nn.DataParallel(model_test)

        # ⚠️ ĐƯỜNG DẪN LƯU SVM
        svm_save_path = os.path.join(save_dir, f'svm_rbf_triplet_{args.experiment}.pkl')

        test_results = evaluate_final_svm(
            model_test, train_loader, test_loader, device,
            kernel='rbf', C=args.svm_C,
            save_svm_path=svm_save_path                # ← Truyền đường dẫn
        )
    else:
        best_model_path = os.path.join(save_dir, f'best_model_ce_{args.experiment}.pth')
        if os.path.exists(best_model_path):
            model_test = FusionM(num_classes=args.num_class, in_c=in_channels,
                                 load_vit=False, embedding_dim=args.embedding_dim)
            model_test.load_state_dict(torch.load(best_model_path, map_location=device))
            model_test = model_test.to(device)
            if len(args.gpu.split(',')) > 1:
                model_test = torch.nn.DataParallel(model_test)
            evaluate(model_test, test_loader, criterion_ce, device, 'Test')
        else:
            print('Best model not found, evaluating current model.')
            evaluate(model, test_loader, criterion_ce, device, 'Test')

    # ===== Lưu kết quả vào file log =====
    log_file = os.path.join(save_dir, 'results.txt')
    with open(log_file, 'w') as f:
        f.write(f"Margin: {margin}\n")
        f.write(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}\n")
        if test_results is not None:
            f.write("\n=== SLICE-LEVEL ===\n")
            sl = test_results['slice_level']
            f.write(f"Accuracy:    {sl['acc']:.2f}%\n")
            f.write(f"AUC:         {sl['auc']:.4f}\n")
            f.write(f"Sensitivity: {sl['sens']:.4f}\n")
            f.write(f"Specificity: {sl['spec']:.4f}\n")
            f.write("\n=== PATIENT-LEVEL (MAJORITY) ===\n")
            pl = test_results['patient_level']['majority']
            f.write(f"  Accuracy:    {pl['acc']:.2f}%\n")
            f.write(f"  AUC:         {pl['auc']:.4f}\n")
            f.write(f"  Sensitivity: {pl['sens']:.4f}\n")
            f.write(f"  Specificity: {pl['spec']:.4f}\n")
    print(f'Results saved to {log_file}')

    # ===== Lưu JSON =====
    if test_results is not None:
        json_file = os.path.join(save_dir, 'test_results.json')
        with open(json_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f'JSON results saved to {json_file}')


def main():
    margin_list = args.triplet_margin
    print(f"Will run with margins: {margin_list}")

    for margin in margin_list:
        train_with_margin(margin, args, train_loader, val_loader, test_loader, in_channels, device)


if __name__ == "__main__":
    main()