from __future__ import print_function
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
import data_loader
import Models
from sklearn import metrics
import numpy as np
from shutil import rmtree, copyfile
import random
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt
import fusionModels
from tools import EarlyStopping

# ========== HÀM youden ==========
def youden(tpr, fpr, thresholds):
    J = tpr - fpr
    idx = np.argmax(J)
    return idx, thresholds[idx]
# ================================

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--num_class', type=int, default=2)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--task_name', type=str, default='breast-cancer-dataset')
parser.add_argument('--path', type=str, default=r'E:\Data')
parser.add_argument('--auto_split', type=str, default='0')

arg = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu

batch_size = arg.batch_size
epochs = 100
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
l2_decay = 0.01
random_seed = int(arg.random_seed)
path = arg.path

cuda = not no_cuda and torch.cuda.is_available()
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

def split_data():
    """Chia dữ liệu từ thư mục B, M thành train (70%), val (10%), test (20%)"""
    for name in ['train', 'val', 'test']:
        dir_path = os.path.join(path, name)
        if os.path.exists(dir_path):
            rmtree(dir_path)
        os.makedirs(dir_path)
    
    items = [i for i in os.listdir(path) if i not in ['train', 'val', 'test']]
    for class_name in items:
        class_path = os.path.join(path, class_name)
        if not os.path.isdir(class_path):
            continue
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.seed(random_seed)
        random.shuffle(files)
        total = len(files)
        train_cnt = int(total * 0.7)
        val_cnt = int(total * 0.1)
        
        train_files = files[:train_cnt]
        val_files = files[train_cnt:train_cnt+val_cnt]
        test_files = files[train_cnt+val_cnt:]
        
        for subset, subset_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            dest_dir = os.path.join(path, subset, class_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            for f in subset_files:
                copyfile(os.path.join(class_path, f), os.path.join(dest_dir, f))
    print('Data split into train/val/test with ratio 70/10/20')

if arg.auto_split == '1':
    split_data()

num_classes = len(os.listdir(os.path.join(path, 'train')))
train_loader = data_loader.load_training(path, 'train', batch_size, kwargs)
val_loader, _, _ = data_loader.load_testing(path, 'val', batch_size, kwargs)
test_loader, test_names, test_labels = data_loader.load_testing(path, 'test', batch_size, kwargs)

len_train = len(train_loader.dataset)
len_val = len(val_loader.dataset)
len_test = len(test_loader.dataset)

def save_dict(model, epoch, is_best=False):
    dict_state = model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict()
    if not os.path.exists('model/{}'.format(arg.task_name)):
        os.makedirs('model/{}'.format(arg.task_name))
    if is_best:
        torch.save(dict_state, 'model/{}/{}_best.pth'.format(arg.task_name, arg.model))
    else:
        torch.save(dict_state, 'model/{}/{}_{}.pth'.format(arg.task_name, arg.model, epoch))

def train(epoch, model):
    LEARNING_RATE = max(lr * (0.1 ** (epoch // 10)), 1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
    model.train()
    correct = 0
    for data, label in train_loader:
        data, label = data.float().cuda(), label.long().cuda()
        pred = model(data)
        optimizer.zero_grad()
        loss = F.nll_loss(F.log_softmax(pred, dim=1), label)
        loss.backward()
        optimizer.step()
        pred = pred.data.max(1)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        # In loss mỗi batch (tùy chọn, có thể bớt để tránh spam)
        # print('Train Epoch: {} loss: {:.4f} lr: {:.6f}'.format(epoch, loss.item(), LEARNING_RATE))
    print('Train Epoch: {} accuracy: {:.2f}% lr: {:.6f}'.format(epoch, 100. * correct / len_train, LEARNING_RATE))

def evaluate(loader, name):
    model.eval()
    total_loss = 0
    correct = 0
    prob_all = None
    target_all = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.float().cuda(), target.long().cuda()
            output = model(data)
            total_loss += F.nll_loss(F.log_softmax(output, dim=1), target, reduction='sum').item()
            pred = output.data.max(1)[1]
            target_all.extend(target.cpu().numpy())
            prob = F.softmax(output, dim=1).cpu().data.numpy()
            if prob_all is None:
                prob_all = prob
            else:
                prob_all = np.append(prob_all, prob, axis=0)
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    total_loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    
    target_onehot = np.eye(num_classes)[np.array(target_all).astype(np.int32).tolist()]
    fpr, tpr, thresholds = roc_curve(target_onehot.ravel(), prob_all.ravel())
    idx, _ = youden(tpr, fpr, thresholds)
    auc_value = auc(fpr, tpr)
    specificity = 1 - fpr[idx]
    sensitivity = tpr[idx]
    
    print('{} set: Loss: {:.4f}, Accuracy: {:.2f}%, AUC: {:.4f}, Specificity: {:.4f}, Sensitivity: {:.4f}'.format(
        name, total_loss, acc, auc_value, specificity, sensitivity))
    return acc, total_loss, auc_value

if __name__ == '__main__':
    if arg.model == 'fusion':
        model = fusionModels.FusionM(num_classes=num_classes, load_vit=True)
    elif arg.model == 'senet50':
        model = Models.Senet50(num_classes=num_classes)
    elif arg.model == 'resnet50':
        model = Models.Resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Model {arg.model} not supported")
    
    model = torch.nn.DataParallel(model, device_ids=list(range(len(arg.gpu.split(',')))))
    model.cuda()
    
    early_stopping = EarlyStopping(patience=20, verbose=True)
    best_val_auc = 0
    
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        val_acc, val_loss, val_auc = evaluate(val_loader, 'Validation')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_dict(model, epoch, is_best=True)
            print('*** Best model saved (val AUC: {:.4f}) ***'.format(best_val_auc))
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping triggered')
            break
    
    print('\n========== Final evaluation on test set ==========')
    best_model_path = 'model/{}/{}_best.pth'.format(arg.task_name, arg.model)
    if os.path.exists(best_model_path):
        state_dict = torch.load(best_model_path)
        model.module.load_state_dict(state_dict)
        print('Loaded best model from {}'.format(best_model_path))
    else:
        print('Warning: best model not found, using last model')
    
    test_acc, test_loss, test_auc = evaluate(test_loader, 'Test')
    print('\n===== Exp-1 Final Results =====')
    print('Test Accuracy: {:.2f}%'.format(test_acc))
    print('Test AUC: {:.4f}'.format(test_auc))