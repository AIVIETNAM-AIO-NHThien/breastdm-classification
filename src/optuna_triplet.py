"""
Optuna hyperparameter optimization for Fusion Model + Triplet Loss + SVM
Usage: conda run -n py39 python optuna_triplet.py
"""

import optuna
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from data_loader_triplet import create_dataloaders
from Fusion_triplet_new import FusionM


# ============================================================
# SEED
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# TRIplet LOSS (copy từ Train_triplet.py)
# ============================================================
def batch_semihard_triplet_loss(embeddings, labels, margin):
    """
    Semi-hard triplet loss.
    embeddings: (B, embedding_dim)
    labels: (B,)
    margin: float
    """
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
        loss += torch.relu(hardest_pos_dist - hardest_semihard_dist + margin)
        num_triplets += 1

    if num_triplets > 0:
        loss = loss / num_triplets
    else:
        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    return loss


# ============================================================
# SVM EVALUATION (copy từ Train_triplet.py)
# ============================================================
def evaluate_embedding_svm(model, train_loader, val_loader, device, C=0.1):
    """Trích xuất embedding, huấn luyện SVM, trả về validation AUC."""
    model.eval()
    train_embs, train_labels = [], []
    val_embs, val_labels = [], []

    with torch.no_grad():
        for data, target in train_loader:
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

    clf = SVC(kernel='rbf', C=C, probability=True, random_state=42)
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    return auc


# ============================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================
def objective(trial):
    """
    Mỗi trial: Optuna chọn 1 bộ siêu tham số, chạy huấn luyện và đánh giá.
    Trả về validation AUC để tối đa hóa.
    """
    # ===== 1. KHÔNG GIAN TÌM KIẾM (thu hẹp để chạy nhanh) =====
    lr = trial.suggest_float('lr', 1e-4, 0.01, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 0.05, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32])  # Cố định 32 cho nhanh
    triplet_margin = trial.suggest_float('triplet_margin', 0.3, 1.0, step=0.1)
    embedding_dim = trial.suggest_categorical('embedding_dim', [128])  # Cố định 128
    svm_C = trial.suggest_float('svm_C', 0.05, 1.0, log=True)
    epochs = trial.suggest_int('epochs', 10, 20, step=5)

    print(f"\n🔍 Trial {trial.number}:")
    print(f"   lr={lr:.6f}, wd={weight_decay:.6f}, batch={batch_size}, margin={triplet_margin}")
    print(f"   emb_dim={embedding_dim}, svm_C={svm_C:.4f}, epochs={epochs}")

    # ===== 2. SEED + DATALOADER =====
    set_seed(42 + trial.number)  # Mỗi trial dùng seed khác nhau
    train_loader, val_loader, _ = create_dataloaders(
        root_dir='/kaggle/input/roi-classification',
        experiment='Exp-1',
        batch_size=batch_size,
        num_workers=4,
        use_triplet=True
    )

    # ===== 3. MODEL =====
    model = FusionM(
        num_classes=2,
        in_c=9,
        load_vit=True,
        embedding_dim=embedding_dim
    )
    model.path = './model/vit_base_patch16_224_in21k.pth'
    model = model.cuda()

    # ===== 4. OPTIMIZER =====
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    # ===== 5. HUẤN LUYỆN =====
    best_val_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            embeddings = model(data, return_embedding=True)
            loss = batch_semihard_triplet_loss(embeddings, target, triplet_margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Đánh giá SVM trên validation set
        val_auc = evaluate_embedding_svm(model, train_loader, val_loader, device='cuda', C=svm_C)
        print(f"   Epoch {epoch}/{epochs} - Val AUC: {val_auc:.4f}")

        # Báo cáo cho Optuna (pruning)
        trial.report(val_auc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_auc > best_val_auc:
            best_val_auc = val_auc

    return best_val_auc


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Tạo study (lưu vào SQLite database)
    study = optuna.create_study(
        direction='maximize',          # Tối đa hóa validation AUC
        study_name='triplet_fusion_optuna_v2',
        storage='sqlite:///triplet_optuna.db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    print("="*60)
    print("🚀 BẮT ĐẦU TỐI ƯU HÓA VỚI OPTUNA")
    print("="*60)
    print(f"   Study name: triplet_fusion_optuna_v2")
    print(f"   Database: triplet_optuna.db")
    print(f"   Sampler: TPE (seed=42)")
    print(f"   Direction: maximize validation AUC")
    print("="*60)

    # ===== CHẠY OPTUNA =====
    study.optimize(
        objective,
        n_trials=25,           # 25 lần thử
        timeout=14400,         # 4 giờ (14400 giây)
        n_jobs=1               # 1 GPU
    )

    # ===== KẾT QUẢ =====
    print("\n" + "="*60)
    print("🏆 KẾT QUẢ TỐI ƯU")
    print("="*60)
    best_trial = study.best_trial
    print(f"   Best validation AUC: {best_trial.value:.4f}")
    print("\n   Siêu tham số tối ưu:")
    for key, value in best_trial.params.items():
        print(f"      {key}: {value}")
    print("="*60)

    # ===== LƯU KẾT QUẢ =====
    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv('optuna_results_triplet.csv', index=False)
    print("\n✅ Đã lưu kết quả vào optuna_results_triplet.csv")

    # ===== IN RA LỆNH CHẠY TRAIN ĐẦY ĐỦ =====
    print("\n" + "="*60)
    print("📋 LỆNH CHẠY TRAINING ĐẦY ĐỦ VỚI THAM SỐ TỐI ƯU")
    print("="*60)
    best = study.best_params
    print(f"""
conda run -n py39 python Train_triplet.py \\
    --data-root /kaggle/input/roi-classification \\
    --experiment Exp-1 \\
    --batch-size {best['batch_size']} \\
    --epochs 100 \\
    --lr {best['lr']:.6f} \\
    --weight-decay {best['weight_decay']:.6f} \\
    --only-triplet \\
    --eval-embedding \\
    --svm-C {best['svm_C']:.4f} \\
    --triplet-margin {best['triplet_margin']} \\
    --embedding-dim {best['embedding_dim']} \\
    --load-vit \\
    --vit-path ./model/vit_base_patch16_224_in21k.pth \\
    --save-dir /kaggle/working/checkpoints
""")

    print("="*60)
    print("✅ HOÀN THÀNH! Chúc bạn thành công! 🚀")