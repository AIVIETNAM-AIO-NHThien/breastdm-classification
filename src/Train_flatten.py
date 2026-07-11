from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import random
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt
from shutil import rmtree, copyfile

import data_loader_flatten
import Fusion_flatten
from tools import EarlyStopping

# -------------------- CẤU HÌNH --------------------
DATA_PATH = "/kaggle/working/dataset_formatted"
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01
SEED = 8
PATIENCE = 20

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"   # thay "0" thành "0,1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Seed
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

# -------------------- DỮ LIỆU --------------------
num_classes = 2

train_loader = data_loader_flatten.load_training(DATA_PATH, 'train', BATCH_SIZE, kwargs)
val_loader, val_names, val_labels = data_loader_flatten.load_validation(DATA_PATH, 'val', BATCH_SIZE, kwargs)
test_loader, test_names, test_labels = data_loader_flatten.load_testing(DATA_PATH, 'test', BATCH_SIZE, kwargs)

len_train = len(train_loader.dataset)
len_val = len(val_loader.dataset)
len_test = len(test_loader.dataset)

print(f"Train: {len_train}, Val: {len_val}, Test: {len_test}")

# -------------------- MODEL --------------------
model = Fusion_flatten.FusionM(num_classes=num_classes, load_vit=True)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.to(device)
print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# -------------------- HÀM ĐÁNH GIÁ --------------------
def youden(tpr, fpr, thresholds):
    J = tpr - fpr
    idx = np.argmax(J)
    return idx, thresholds[idx]

def evaluate(model, loader, dataset_size, name=''):
    model.eval()
    test_loss = 0
    correct = 0
    possbilitys = None
    pred_all = []
    labels_all = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum')
            test_loss += loss.item()
            pred = output.data.max(1)[1]
            pred_all.extend(pred.cpu().numpy())
            labels_all.extend(target.cpu().numpy())
            prob = F.softmax(output, dim=1).cpu().numpy()
            possbility = prob
            if possbilitys is None:
                possbilitys = possbility
            else:
                possbilitys = np.append(possbilitys, possbility, axis=0)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    pred_all = np.array(pred_all)
    labels_all = np.array(labels_all)
    
    cm = metrics.confusion_matrix(labels_all, pred_all)
    label_onehot = np.eye(num_classes)[labels_all.astype(np.int32)]
    fpr, tpr, thresholds = roc_curve(label_onehot.ravel(), possbilitys.ravel())
    idx, _ = youden(tpr, fpr, thresholds)
    auc_value = auc(fpr, tpr)
    
    acc = 100. * correct / dataset_size
    avg_loss = test_loss / dataset_size
    
    print(f"{name} | Acc: {acc:.2f}% | AUC: {auc_value:.4f} | Sens: {tpr[idx]:.4f} | Spec: {1-fpr[idx]:.4f}")
    print(cm)
    return acc, avg_loss, auc_value

# -------------------- TRAIN --------------------
def train_epoch(epoch, model, optimizer):
    model.train()
    correct = 0
    total_loss = 0
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
    train_acc = 100. * correct / len_train
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%")
    return avg_loss

# -------------------- MAIN --------------------
if __name__ == "__main__":
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    
    best_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_epoch(epoch, model, optimizer)
        scheduler.step()
        
        with torch.no_grad():
            val_acc, val_loss, val_auc = evaluate(model, val_loader, len_val, name='Val')
        
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_fusion_model.pth")
            print(f"  -> Saved best model with Val AUC = {best_auc:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    print("\n=== Training Complete ===")
    print(f"Best Val AUC: {best_auc:.4f}")
    
    # Đánh giá trên test với model tốt nhất
    model.load_state_dict(torch.load("best_fusion_model.pth"))
    print("\n=== Final Test Evaluation ===")
    test_acc, test_loss, test_auc = evaluate(model, test_loader, len_test, name='Test')