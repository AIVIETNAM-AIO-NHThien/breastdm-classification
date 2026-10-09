import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.svm import SVC

# Import data loader và model
from data_loader_triplet import create_dataloaders, BreastDMDataset, TripletBatchSampler
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

# ===== THÊM: Tùy chọn cho SVM evaluation =====
parser.add_argument('--svm-augment', action='store_true', default=False,
                    help='Nếu BẬT: fit SVM trên ảnh augment (giống code cũ). '
                         'Mặc định TẮT = fit trên ảnh gốc (fix augment mismatch).')

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
# Tạo DataLoader (dùng cho training model)
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


# ============================================================
# HÀM MỚI: Tạo DataLoader cho SVM với augment=False
# Giữ nguyên TripletBatchSampler → cùng subset như code cũ
# ============================================================
def get_svm_train_loader(args):
    """
    Tạo DataLoader cho SVM evaluation với augment=False.
    
    - Nếu args.svm_augment=True: trả về train_loader cũ (augment=True) 
      → giống code cũ.
    - Nếu args.svm_augment=False (mặc định): tạo DataLoader mới với 
      augment=False, cùng TripletBatchSampler → cùng subset nhưng ảnh gốc.
    """
    if args.svm_augment:
        # Dùng train_loader cũ — augment=True (giống code gốc)
        print(f"[SVM] Using train_loader (augment=True, subset)")
        return train_loader
    
    # Tạo dataset mới với augment=False
    train_dataset_svm = BreastDMDataset(
        root_dir=args.data_root,
        split='train',
        experiment=args.experiment,
        augment=False,                 # ⚠️ ĐIỂM SỬA CHÍNH
    )
    
    # Dùng TripletBatchSampler → giữ nguyên subset như code cũ
    sampler = TripletBatchSampler(
        train_dataset_svm,
        batch_size=args.batch_size,
        shuffle=False,                 # Không shuffle khi eval
    )
    
    loader = DataLoader(
        train_dataset_svm,
        batch_sampler=sampler,
        num_workers=args.num_workers,
    )
    
    print(f"[SVM] Using new loader (augment=False, subset): "
          f"{len(sampler)} batches × {args.batch_size}")
    return loader


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
# Hàm đánh giá SVM trên embedding (ĐÃ FIX AUGMENT)
# -------------------------------
def evaluate_embedding_svm(model, args, val_loader, device, kernel='rbf', C=None):
    """
    Đánh giá bằng SVM.
    
    ĐÃ FIX: SVM fit trên ảnh GỐC (augment=False) khi args.svm_augment=False.
    Vẫn giữ TripletBatchSampler → cùng subset ~800 slices.
    """
    if C is None:
        C = args.svm_C
    
    model.eval()
    
    # Lấy DataLoader cho SVM (có thể là augment=False)
    train_loader_svm = get_svm_train_loader(args)
    
    train_embs, train_labels = [], []
    val_embs, val_labels = [], []

    with torch.no_grad():
        for data, target in train_loader_svm:
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            train_embs.append(emb)
            train_labels.append(target.numpy())
        
        for data, target in val_loader:
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            val_embs.append(emb)
            val_labels.append(target.numpy())

    X_train = np.concatenate(train_embs)
    y_train = np.concatenate(train_labels)
    X_val = np.concatenate(val_embs)
    y_val = np.concatenate(val_labels)

    print(f"[SVM-fit] Train: {X_train.shape}, Val: {X_val.shape}")
    
    clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    print(f'SVM (kernel={kernel}, C={C}) on Val: Acc: {acc*100:.2f}%, AUC: {auc:.4f}, '
          f'Sens: {sens:.4f}, Spec: {spec:.4f}')
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
# Hàm đánh giá cuối cùng bằng SVM cho test (ĐÃ FIX AUGMENT)
# -------------------------------
def evaluate_final_svm(model, args, test_loader, device, kernel='rbf', C=None):
    """
    Đánh giá cuối cùng trên test set.
    
    ĐÃ FIX: SVM fit trên ảnh GỐC (augment=False) khi args.svm_augment=False.
    """
    if C is None:
        C = args.svm_C
    
    model.eval()
    
    train_loader_svm = get_svm_train_loader(args)
    
    train_embs, train_labels = [], []
    test_embs, test_labels = [], []

    with torch.no_grad():
        for data, target in train_loader_svm:
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            train_embs.append(emb)
            train_labels.append(target.numpy())
        
        for data, target in test_loader:
            emb = model(data.to(device), return_embedding=True).cpu().numpy()
            test_embs.append(emb)
            test_labels.append(target.numpy())

    X_train = np.concatenate(train_embs)
    y_train = np.concatenate(train_labels)
    X_test = np.concatenate(test_embs)
    y_test = np.concatenate(test_labels)

    print(f"[SVM-fit] Train: {X_train.shape}, Test: {X_test.shape}")
    
    clf = SVC(kernel=kernel, C=C, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    print(f'Test set (SVM): Accuracy: {acc:.2f}%, AUC: {auc:.4f}, '
          f'Sens: {sens:.4f}, Spec: {spec:.4f}')
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant'], digits=4))
    print('Confusion Matrix:')
    print(cm)
    return acc, auc


# -------------------------------
# Hàm huấn luyện cho một margin cụ thể
# -------------------------------
def train_with_margin(margin, args, train_loader, val_loader, test_loader, in_channels, device):
    print(f"\n{'='*60}")
    print(f"   BẮT ĐẦU HUẤN LUYỆN VỚI MARGIN = {margin}")
    print(f"   SVM augment: {args.svm_augment} "
          f"({'Ảnh augment (giống code cũ)' if args.svm_augment else 'Ảnh gốc (đã fix)'})")
    print('='*60)

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

    # Loss và Optimizer
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
                    model, args, val_loader, device, kernel='rbf'
                )
                print(f'Val set (SVM): Accuracy: {val_acc:.2f}%, AUC: {val_auc:.4f}, '
                      f'Sens: {val_sens:.4f}, Spec: {val_spec:.4f}')
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
    if args.only_triplet:
        best_model_path = os.path.join(save_dir, f'best_model_triplet_only_{args.experiment}.pth')
        model_test = FusionM(num_classes=args.num_class, in_c=in_channels,
                             load_vit=False, embedding_dim=args.embedding_dim)
        model_test.load_state_dict(torch.load(best_model_path, map_location=device))
        model_test = model_test.to(device)
        if len(args.gpu.split(',')) > 1:
            model_test = torch.nn.DataParallel(model_test)
        evaluate_final_svm(model_test, args, test_loader, device, kernel='rbf')
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

    # Lưu kết quả
    log_file = os.path.join(save_dir, 'results.txt')
    with open(log_file, 'w') as f:
        f.write(f"Margin: {margin}\n")
        f.write(f"SVM augment: {args.svm_augment}\n")
        f.write(f"Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}\n")
    print(f'Results saved to {log_file}')


# -------------------------------
# HÀM CHÍNH
# -------------------------------
def main():
    margin_list = args.triplet_margin
    print(f"Will run with margins: {margin_list}")

    for margin in margin_list:
        train_with_margin(margin, args, train_loader, val_loader, test_loader, in_channels, device)


if __name__ == "__main__":
    main()