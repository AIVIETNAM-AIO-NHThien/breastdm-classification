import pretrainedmodels.models
import torch.nn as nn
import torch
import torch.nn.functional as F
import pretrainedmodels.models as premodels 

class Senet101(nn.Module):
    def __init__(self, num_classes=2):
        super(Senet101, self).__init__()
        model_se = premodels.se_resnet101(pretrained='imagenet')
        self.layer0 = model_se.layer0
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2
        self.layer3 = model_se.layer3
        self.layer4 = model_se.layer4
        self.avgpool = model_se.avg_pool
        self.dropout = model_se.dropout
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Senet50(nn.Module):
    def __init__(self,num_classes=2):
        super(Senet50, self).__init__()
        model_se = premodels.se_resnet50(pretrained='imagenet')
        self.layer0 = model_se.layer0
        self.layer1 = model_se.layer1
        self.layer2 = model_se.layer2
        self.layer3 = model_se.layer3
        self.layer4 = model_se.layer4
        self.avgpool = model_se.avg_pool
        self.dropout = model_se.dropout
        self.fc = nn.Linear(2048, num_classes)

    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x