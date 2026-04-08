from __future__ import print_function
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import math
import data_loader
import Models
import time
from torch.utils import model_zoo
from sklearn import metrics

import numpy as np
import pandas as pd
import os
from shutil import rmtree, copytree, copyfile
import random
from sklearn.metrics import roc_auc_score, auc, roc_curve
import matplotlib.pyplot as plt

import fusionModels
from tools import EarlyStopping

# ========== THÊM HÀM youden ==========
def youden(tpr, fpr, thresholds):
    J = tpr - fpr
    idx = np.argmax(J)
    return idx, thresholds[idx]
# =====================================

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--model', type=str, default='resnet50', help='coco.data file path')
parser.add_argument('--gpu', type=str, default='0', help='coco.data file path')
parser.add_argument('--num_class', type=int, default=2, help='coco.data file path')
parser.add_argument('--random_seed', type=int, default=1, help='coco.data file path')
parser.add_argument('--split_train_ratio', type=float, default=0.8, help='coco.data file path')
parser.add_argument('--task_name', type=str, default='breast-cancer-dataset', help='coco.data file path')
parser.add_argument('--path', type=str, default=r'E:\Data', help='coco.data file path')
parser.add_argument('--auto_split', type=str, default='0', help='coco.data file path')

arg = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
# Training settings
batch_size = arg.batch_size
epochs = 100
lr = 0.01
momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 0.01
random_seed = int(arg.random_seed)
split_train_ratio = arg.split_train_ratio
path = arg.path

# Sửa source_name và target_name cho phù hợp
source_name = "train"
target_name = "test"   # sẽ load từ thư mục test sau khi chia 70/10/20

cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

def split_data():
    # Xóa các thư mục cũ nếu có
    for name in ['train', 'val', 'test']:
        dir_path = os.path.join(path, name)
        if os.path.exists(dir_path):
            rmtree(dir_path)
        os.makedirs(dir_path)
    
    # Lấy danh sách các thư mục con (B, M, ...) - mỗi thư mục là một class
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
        # test_cnt còn lại = total - train_cnt - val_cnt
        
        train_files = files[:train_cnt]
        val_files = files[train_cnt:train_cnt+val_cnt]
        test_files = files[train_cnt+val_cnt:]
        
        # Copy vào các thư mục đích
        for subset, subset_files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            dest_dir = os.path.join(path, subset, class_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            for f in subset_files:
                copyfile(os.path.join(class_path, f), os.path.join(dest_dir, f))
    print('Complete data split into train/val/test with ratio 70/10/20')

if arg.auto_split == '1':
    split_data()
else:
    pass

# Sau khi split, đếm số class dựa trên thư mục train (hoặc test)
num_classes = len(os.listdir(os.path.join(path, source_name)))  # thư mục train chứa các class (B, M)

source_loader = data_loader.load_training(path, source_name, batch_size, kwargs)
# Load validation từ thư mục test (theo đúng ý đồ của tác giả, test đóng vai trò validation)
target_val_loader, names, label = data_loader.load_testing(path, target_name, batch_size, kwargs)

len_source_dataset = len(source_loader.dataset)
len_target_dataset = len(target_val_loader.dataset)

def save_dict(model):
    dict = model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict()
    if not os.path.exists('model/{}'.format(arg.task_name)):
        os.makedirs('model/{}'.format(arg.task_name))
    torch.save(dict, 'model/{}/{}.pth'.format(arg.task_name, arg.model))

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    classes = ['benign', 'malignant']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 12), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig(savename, format='png')
    plt.show()

def train(epoch, model):
    LEARNING_RATE = max(lr * (0.1 ** (epoch // 10)), 1e-5)

    optimizer = torch.optim.SGD([
        {'params': model.parameters()}
    ], lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

    model.train()
    correct = 0
    for data, label in source_loader:
        data = data.float().cuda()
        label = label.long().cuda()
        pred = model(data)

        optimizer.zero_grad()
        loss = F.nll_loss(F.log_softmax(pred, dim=1), label)
        loss.backward()
        optimizer.step()
        pred = pred.data.max(1)[1]
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        print('Train Epoch: {} loss :{} learning_rate:{}\n'.format(epoch, loss.item(), LEARNING_RATE))
    print(f'train accuracy: {100. * correct / len_source_dataset}%')

def val(model):
    model.eval()
    test_loss = 0
    correct = 0
    possbilitys = None
    pred_all = []
    all_targets = []  # lưu target để tính confusion matrix
    for data, target in target_val_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        s_output = model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, reduction='sum').item()
        pred = s_output.data.max(1)[1]
        pred_all.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        possbility = F.softmax(s_output, dim=1).cpu().data.numpy()
        if possbilitys is None:
            possbilitys = possbility
        else:
            possbilitys = np.append(possbilitys, possbility, axis=0)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    pred_all = [i for item in pred_all for i in item]
    all_targets = [i for item in all_targets for i in item]
    cm = metrics.confusion_matrix(all_targets, pred_all)
    label_onehot = np.eye(num_classes)[np.array(all_targets).astype(np.int32).tolist()]
    fpr, tpr, thresholds = roc_curve(label_onehot.ravel(), possbilitys.ravel())
    index, optimal_threshold = youden(tpr, fpr, thresholds)
    auc_value = auc(fpr, tpr)
    test_loss /= len_target_dataset

    print('Specific:{} sensitivity:{} Auc:{}'.format(1 - fpr[index], tpr[index], auc_value))
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))

    return 100. * correct / len_target_dataset, test_loss, auc_value

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    possbilitys = None
    label_names = ['benign', 'malignant']
    pred_all = []
    all_targets = []
    for data, target in target_val_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        s_output = model(data)
        test_loss += F.nll_loss(F.log_softmax(s_output, dim=1), target, reduction='sum').item()
        pred = s_output.data.max(1)[1]
        pred_all.append(pred.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        possbility = F.softmax(s_output, dim=1).cpu().data.numpy()
        if possbilitys is None:
            possbilitys = possbility
        else:
            possbilitys = np.append(possbilitys, possbility, axis=0)
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    pred_all = [i for item in pred_all for i in item]
    all_targets = [i for item in all_targets for i in item]
    print(metrics.classification_report(all_targets, pred_all, labels=range(2), target_names=label_names, digits=4))
    cm = metrics.confusion_matrix(all_targets, pred_all, labels=range(2))
    print(cm)
    label_onehot = np.eye(num_classes)[np.array(all_targets).astype(np.int32).tolist()]
    fpr, tpr, thresholds = roc_curve(label_onehot.ravel(), possbilitys.ravel())
    index, optimal_threshold = youden(tpr, fpr, thresholds)
    auc_value = auc(fpr, tpr)
    test_loss /= len_target_dataset
    print('Specific:{} sensitivity:{} Auc:{}'.format(1 - fpr[index], tpr[index], auc_value))
    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        target_name, test_loss, correct, len_target_dataset,
        100. * correct / len_target_dataset))
    return 100. * correct / len_target_dataset

if __name__ == '__main__':
    if arg.model == 'resnet101':
        model = Models.Resnet101(num_classes=num_classes)
    if arg.model == 'resnext101':
        model = Models.Resnext101(num_classes=num_classes)
    if arg.model == 'densenet201':
        model = Models.Densnet201(num_classes=num_classes)
    if arg.model == 'resnet50':
        model = Models.Resnet50(num_classes=num_classes)
    if arg.model == 'densenet169':
        model = Models.Densenet169(num_classes=num_classes)
    if arg.model == 'vgg16':
        model = Models.vgg16(num_classes=num_classes)
    if arg.model == 'senet101':
        model = Models.Senet101(num_classes=num_classes)
    if arg.model == 'resnet18':
        model = Models.Resnet18(num_classes=num_classes)
    if arg.model == 'mynet':
        model = Models.MyNet(num_classes=num_classes)
    if arg.model == 'senet50':
        model = Models.Senet50(num_classes=num_classes)
    if arg.model == 'resnet152':
        model = Models.Resnet152(num_classes=num_classes)
    if arg.model == 'fusion':
        model = fusionModels.FusionM(num_classes=num_classes, load_vit=True)

    print(model)

    model = torch.nn.DataParallel(model, device_ids=list(range(len(arg.gpu.split(',')))))
    model.cuda()
    best_acc = 0
    early_stopping = EarlyStopping(patience=20, verbose=True)
    for epoch in range(1, epochs + 1):
        train(epoch, model)
        with torch.no_grad():
            e_acc, test_loss, Auc = val(model)
        dict = model.module.state_dict() if type(model) is nn.parallel.DistributedDataParallel else model.state_dict()
        if not os.path.exists(os.path.join('model', arg.task_name)):
            os.makedirs(os.path.join('model', arg.task_name))
        early_stopping(test_loss, model)
        if Auc > best_acc:
            best_acc = Auc
            torch.save(dict, os.path.join('model', arg.task_name, arg.model + str(epoch) + '.pth'),
                       _use_new_zipfile_serialization=False)
            print('save success')
        if early_stopping.early_stop:
            print('Early stopping')
            break