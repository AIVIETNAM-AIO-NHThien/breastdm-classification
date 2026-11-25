import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import joblib
import shutil
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.svm import SVC

# Import data loader và model
from data_loader_triplet_fix_new import create_dataloaders
from Fusion_triplet_new import FusionM

# Hàm tính Sensitivity và Specificity 
def calc_sens_spec_youden(all_labels, all_probs):
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
parser = argparse.ArgumentParser(description='LG-CAFN Triplet Only Training')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--gpu', type=str, default='0', help='GPU id(s)')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')
parser.add_argument('--experiment', type=str, default='Exp-1', choices=['Exp-1', 'Exp-2'], help='Experiment type')
parser.add_argument('--data-root', type=str, required=True, help='root directory containing train/val/test folders')
parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.05, help='L2 regularization')
parser.add_argument('--load-vit', action='store_true', default=True, help='load pretrained ViT weights')
parser.add_argument('--vit-path', type=str, default='./model/vit_base_patch16_224_in21k.pth', help='path to ViT pretrained weights')
parser.add_argument('--save-dir', type=str, default='checkpoints', help='directory to save model checkpoints')
parser.add_argument('--num-workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--triplet-margin', type=float, nargs='+', default=[1.0], help='margin for triplet loss')
parser.add_argument('--embedding-dim', type=int, default=128, help='Dimension of embedding for triplet loss')
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

# Tạo DataLoader (Luôn dùng Triplet)
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

# Semi-hard triplet loss 
def batch_semihard_triplet_loss(embeddings, labels, margin):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
    loss = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
    num_triplets = 0
    device = embeddings.device

    for i in range(len(labels)):
        anchor_label = labels[i]
        pos_mask = (labels == anchor_label) & (torch.arange(len(labels), device=device) != i)
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

# Hàm huấn luyện một epoch (Chỉ dùng Triplet Loss)
def train_one_epoch(epoch, model, loader, optimizer, device, margin):
    model.train()
    total_loss = 0.0
    total_triplet = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        embeddings = model(data, return_embedding=True)
        loss = batch_semihard_triplet_loss(embeddings, target, margin)
        total_triplet += loss.item() * data.size(0)

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

# Hàm load Checkpoint và đánh giá bằng SVM
def evaluate_with_saved_checkpoints(model_path, svm_path, val_loader, in_channels, args, device, target_name='Val'):
    # 1. Load Model
    model_test = FusionM(num_classes=args.num_class, in_c=in_channels, load_vit=False, embedding_dim=args.embedding_dim)
    model_test.load_state_dict(torch.load(model_path, map_location=device))
    model_test = model_test.to(device)
    if len(args.gpu.split(',')) > 1:
        model_test = torch.nn.DataParallel(model_test)
    model_test.eval()
    
    # 2. Load SVM
    clf = joblib.load(svm_path)
    
    # 3. Trích xuất đặc trưng từ tập Val
    val_embs, val_labels = [], []
    with torch.no_grad():
        for data, target in val_loader:
            emb = model_test(data.to(device), return_embedding=True).cpu().numpy()
            val_embs.append(emb)
            val_labels.append(target.numpy())
            
    X_val = np.concatenate(val_embs)
    y_val = np.concatenate(val_labels)
    
    # 4. Dự đoán
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1]
    
    # 5. Tính toán metrics
    acc = accuracy_score(y_val, y_pred) * 100
    auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    TN, FP = cm[0,0], cm[0,1]
    FN, TP = cm[1,0], cm[1,1]
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    print(f'{target_name} set (SVM): Acc: {acc:.2f}%, AUC: {auc:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}')
    print('Confusion Matrix:')
    print(cm)
    if target_name == 'Test':
        print(classification_report(y_val, y_pred, target_names=['Benign', 'Malignant'], digits=4))
        
    return acc, auc, sens, spec

# Hàm huấn luyện cho một margin 
def train_with_margin(margin, args, train_loader, val_loader, test_loader, in_channels, device):
    print(f"\n{'='*60}")
    print(f"   BẮT ĐẦU HUẤN LUYỆN VỚI MARGIN = {margin}")
    print('='*60)

    save_dir = os.path.join(args.save_dir, f"margin_{margin}")
    os.makedirs(save_dir, exist_ok=True)

    # Khởi tạo model mới (Đã truyền vit_path chuẩn xác)
    model = FusionM(num_classes=args.num_class,
                    in_c=in_channels,
                    load_vit=args.load_vit,
                    embedding_dim=args.embedding_dim,
                    vit_path=args.vit_path)
    
    model = model.to(device)
    if len(args.gpu.split(',')) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(len(args.gpu.split(',')))))

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

        # BƯỚC 1: Train một epoch
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, device, margin)

        # BƯỚC 2: Lưu Checkpoint của FusionM
        model_save_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
        state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
        torch.save(state_dict, model_save_path)
        print(f'Model Checkpoint saved to {model_save_path}')

        # BƯỚC 3: Trích xuất embedding tập Train và Fit SVM
        print("Extracting train embeddings for SVM...")
        model.eval()
        train_embs, train_labels = [], []
        with torch.no_grad():
            for data, target in train_loader:
                emb = model(data.to(device), return_embedding=True).cpu().numpy()
                train_embs.append(emb)
                train_labels.append(target.numpy())
        
        X_train = np.concatenate(train_embs)
        y_train = np.concatenate(train_labels)
        
        clf = SVC(kernel='rbf', C=args.svm_C, probability=True, random_state=42)
        clf.fit(X_train, y_train)
        
        # BƯỚC 4: Lưu Checkpoint của SVM
        svm_save_path = os.path.join(save_dir, f'svm_epoch_{epoch}.pkl')
        joblib.dump(clf, svm_save_path)
        print(f'SVM Checkpoint saved to {svm_save_path}')

        # BƯỚC 5: Load cả 2 Checkpoint vừa lưu và Evaluate trên tập Val
        print("Loading checkpoints for Val evaluation...")
        val_acc, val_auc, val_sens, val_spec = evaluate_with_saved_checkpoints(
            model_save_path, svm_save_path, val_loader, in_channels, args, device, target_name='Val'
        )

        # BƯỚC 6: Lưu lại best model nếu Val AUC tốt nhất
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            
            best_model_path = os.path.join(save_dir, f'best_model_triplet_only_{args.experiment}.pth')
            best_svm_path = os.path.join(save_dir, f'best_svm_{args.experiment}.pkl')
            
            shutil.copy(model_save_path, best_model_path)
            shutil.copy(svm_save_path, best_svm_path)
            
            print(f'*** Best Model & SVM saved (Val AUC: {val_auc:.4f}) ***')

    print(f'\nTraining finished for margin={margin}. Best validation AUC: {best_val_auc:.4f} at epoch {best_epoch}')

    # ----- Đánh giá Test với Best Checkpoints -----
    print('\nLoading best model & SVM for test evaluation...')
    best_model_path = os.path.join(save_dir, f'best_model_triplet_only_{args.experiment}.pth')
    best_svm_path = os.path.join(save_dir, f'best_svm_{args.experiment}.pkl')
    
    if os.path.exists(best_model_path) and os.path.exists(best_svm_path):
        test_acc, test_auc, test_sens, test_spec = evaluate_with_saved_checkpoints(
            best_model_path, best_svm_path, test_loader, in_channels, args, device, target_name='Test'
        )
        print(f'\n✅ Final Test Results (SVM):')
        print(f'   Accuracy:  {test_acc:.2f}%')
        print(f'   AUC:       {test_auc:.4f}')
        print(f'   Sensitivity: {test_sens:.4f}')
        print(f'   Specificity: {test_spec:.4f}')
    else:
        print('Best checkpoints not found.')

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