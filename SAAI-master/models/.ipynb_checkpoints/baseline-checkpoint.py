import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn import Parameter
import numpy as np

from models.resnet import resnet50, resnet18
from utils.calc_acc import calc_acc



from layers import TripletLoss, module
from layers import CSLoss
from layers import CenterLoss
from layers import cbam
from layers import NonLocalBlockND
from layers import DualBNNeck
from layers.module.part_pooling import TransformerPool, SAFL

from layers.module.my_module import SEAttention,ECAAttention,CoTAttention,S2Attention,SemanticPartAttention
#ResNet50+Bnneck+SAAA
class Baseline(nn.Module):
    def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False,
                 modality_attention=0, mutual_learning=False, **kwargs):
        super(Baseline, self).__init__()

        self.drop_last_stride = drop_last_stride
        self.pattern_attention = pattern_attention
        self.modality_attention = modality_attention
        self.mutual_learning = mutual_learning

        if backbone == "resnet50":
            self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride,
                                     modality_attention=modality_attention)
            D = 2048
        elif backbone == "resnet18":
            self.backbone = resnet18(pretrained=True, drop_last_stride=drop_last_stride,
                                     modality_attention=modality_attention)
            D = 512

        self.base_dim = D
        self.dim = D
        self.k_size = kwargs.get('k_size', 8)
        self.part_num = kwargs.get('num_parts', 7)
        self.dp = kwargs.get('dp', "l2")
        self.dp_w = kwargs.get('dp_w', 0.5)
        self.cs_w = kwargs.get('cs_w', 1.0)
        self.margin1 = kwargs.get('margin1', 0.01)
        self.margin2 = kwargs.get('margin2', 0.7)

        self.attn_pool = SAFL(part_num=self.part_num)
        
        self.SAAA = SemanticPartAttention()
        
        self.bn_neck = DualBNNeck(self.base_dim + self.dim * self.part_num)

        self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data
        self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data

        self.KL_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.weight_KL = kwargs.get('weight_KL', 2.0)
        self.update_rate = kwargs.get('update_rate', 0.2)
        self.update_rate_ = self.update_rate

        self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.cs_loss_fn = CSLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2)

        #self.SEAttention = SEAttention(2048)
        #self.CoTAttention = CoTAttention(2048)
        self.S2Attention = S2Attention(channels=2048)
    def forward(self, inputs, labels=None, **kwargs):
        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)
        # CNN
        # input image [80, 3, 288, 144]

        global_feat = self.backbone(inputs)  # ResNet50


        b, c, w, h = global_feat.shape
        #   input ->  conv1 ->  max pool -> conv2 -> conv3 ->  conv4 ->  conv5
        # [80, 3, 288, 144]->[80, 64, 144, 72]->[80, 64, 72, 36]->[80, 256, 72, 36]->[80, 512, 36, 18]->[80, 1024, 18, 9]->[80, 2048, 18, 9]
        # After ResNet50 [80, 3, 288, 144]->[80, 2048, 18, 9]
        #part_feat, attn = self.attn_pool(global_feat)  # SAFL
        
        part_feat, attn, part_diversity_loss = self.SAAA(global_feat)
        
        # After ResNet50, SAFL [80, 2048, 18, 9]->[80, 2048, 162]->[80, 162, 2048]->[80, 7, 2048]->[80,14336]
        global_feat = self.S2Attention(global_feat)
        #global_feat = self.SEAttention(global_feat)
        #global_feat_2 = self.CoTAttention(global_feat)
       # global_feat = global_feat_1 + global_feat_2
        
        global_feat = global_feat.mean(dim=(2, 3))  # Global average pooling
        # After ResNet50  SAFL Global average pooling：[80, 2048, 18, 9]->[80, 2048]
        # cat [80,14336]+[80, 2048]->[80, 16384]
        feats = torch.cat([part_feat, global_feat], dim=1)

        if self.training:  # 计算DP Loss
            masks = attn.view(b, self.part_num, w * h)
            if self.dp == "cos":  # 余弦相似度
                loss_dp = torch.bmm(masks, masks.permute(0, 2, 1))
                loss_dp = torch.triu(loss_dp, diagonal=1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
                loss_dp += -masks.mean() + 1
            elif self.dp == "l2":  # 欧式距离（L2范数）
                loss_dp = 0
                for i in range(self.part_num):
                    for j in range(i + 1, self.part_num):
                        loss_dp += ((((masks[:, i] - masks[:, j]) ** 2).sum(dim=1) / (18 * 9)) ** 0.5).sum()
                loss_dp = - loss_dp / (b * self.part_num * (self.part_num - 1) / 2)
                loss_dp *= self.dp_w
        if not self.training:
            feats = self.bn_neck(feats, sub)
            return feats
        else:
            return self.train_forward(feats, labels, loss_dp, sub, **kwargs)

    def train_forward(self, feat, labels, loss_dp, sub, **kwargs):
        metric = {}

        loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels)  # CS Loss  交叉熵损失
        feat = self.bn_neck(feat, sub)

        logits = self.classifier(feat)
        loss_id = self.ce_loss_fn(logits.float(), labels)  # ID loss
        tmp = self.ce_loss_fn(logits.float(), labels)
        metric.update({'ce': tmp.data})  # CE LOSS  中心损失的交叉熵损失

        cam_ids = kwargs.get('cam_ids')
        sub = (cam_ids == 3) + (cam_ids == 6)

        logits_v = self.visible_classifier(feat[sub == 0])  # 可见光分类器
        loss_id += self.ce_loss_fn(logits_v.float(), labels[sub == 0])
        logits_i = self.infrared_classifier(feat[sub == 1])  # 红外分类器
        loss_id += self.ce_loss_fn(logits_i.float(), labels[sub == 1])

        logits_m = torch.cat([logits_v, logits_i], 0).float()
        with torch.no_grad():  # 不进行梯度计算
            self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
                                                    + self.infrared_classifier.weight.data * self.update_rate
            self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
                                                   + self.visible_classifier.weight.data * self.update_rate

            logits_v_ = self.infrared_classifier_(feat[sub == 0])
            logits_i_ = self.visible_classifier_(feat[sub == 1])
            logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()

        loss_id += self.ce_loss_fn(logits_m, logits_m_.softmax(dim=1))

        metric.update({'id': loss_id.data})  # Identification loss
        metric.update({'cs': loss_cs.data})  # Cross-entropy loss
        metric.update({'dp': loss_dp.data})  # Diversity loss

        loss = loss_id + loss_cs * self.cs_w + loss_dp * self.dp_w  # Total loss

        return loss, metric


# import torch.nn as nn
# from torch.nn import functional as F
# from models.resnet import resnet50, resnet18

#ResNet50+BnNeck
# class Baseline(nn.Module):
#     def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False,
#                   modality_attention=0, mutual_learning=False, **kwargs):
#         super(Baseline, self).__init__()
#         if backbone == "resnet50":
#             self.backbone = resnet50(pretrained=True, drop_last_stride= drop_last_stride)
#             D = 2048
#         elif backbone == "resnet18":
#             self.backbone = resnet18(pretrained=True, drop_last_stride=drop_last_stride)
#             D = 512
#         self.drop_last_stride = drop_last_stride
#         self.pattern_attention = pattern_attention
#         self.modality_attention = modality_attention
#         self.mutual_learning = mutual_learning
#         self.k_size = kwargs.get('k_size', 8)
#         self.cs_w = kwargs.get('cs_w', 1.0)
#         self.margin1 = kwargs.get('margin1', 0.01)
#         self.margin2 = kwargs.get('margin2', 0.7)
#         self.weight_KL = kwargs.get('weight_KL', 2.0)# KL散度损失权重
#         self.update_rate = kwargs.get('update_rate', 0.2)# 更新率
#         self.update_rate_ = self.update_rate# 更新率
#         self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
#         self.cs_loss_fn = CSLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2)
        
#         self.bn_neck = DualBNNeck(D)
#         self.visible_classifier = nn.Linear(D, num_classes, bias=False)
#         self.infrared_classifier = nn.Linear(D, num_classes, bias=False)
#         self.visible_classifier_ = nn.Linear(D, num_classes, bias=False)
#         self.visible_classifier_.weight.requires_grad_(False)
#         self.visible_classifier_.weight.data = self.visible_classifier.weight.data

#         self.infrared_classifier_ = nn.Linear(D, num_classes, bias=False)
#         self.infrared_classifier_.weight.requires_grad_(False)
#         self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data
    
#         self.classifier = nn.Linear(D, num_classes, bias=False)
       
        
#         self.classifier = nn.Linear(D, num_classes, bias=False)
#         self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

#     def forward(self, inputs, labels=None, **kwargs):
#         cam_ids = kwargs.get('cam_ids')
#         sub = (cam_ids == 3) + (cam_ids == 6)
#         global_feat = self.backbone(inputs)
#         # CNN
#           # ResNet50_feat = global_
#         global_feat = global_feat.mean(dim=(2, 3))  # Global average pooling

#         if not self.training:
#             feats = self.bn_neck(global_feat, sub)
#             return feats
#         else:
#             return self.train_forward(global_feat, labels, sub, **kwargs)

#     def train_forward(self, feat, labels, sub, **kwargs):
#         metric = {}

#         loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels)  # CS Loss
#         feat = self.bn_neck(feat, sub)

#         logits = self.classifier(feat)
#         loss_id = self.ce_loss_fn(logits.float(), labels)  # ID loss
#         tmp = self.ce_loss_fn(logits.float(), labels)
#         metric.update({'ce': tmp.data})

#         cam_ids = kwargs.get('cam_ids')
#         sub = (cam_ids == 3) + (cam_ids == 6)

#         logits_v = self.visible_classifier(feat[sub == 0])  # 可见光分类器
#         loss_id += self.ce_loss_fn(logits_v.float(), labels[sub == 0])

#         logits_i = self.infrared_classifier(feat[sub == 1])  # 红外分类器
#         loss_id += self.ce_loss_fn(logits_i.float(), labels[sub == 1])

#         logits_m = torch.cat([logits_v, logits_i], 0).float()

#         with torch.no_grad():
#             self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
#                                                     + self.infrared_classifier.weight.data * self.update_rate
#             self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
#                                                    + self.visible_classifier.weight.data * self.update_rate

#             logits_v_ = self.infrared_classifier_(feat[sub == 0])
#             logits_i_ = self.visible_classifier_(feat[sub == 1])
#             logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()

#         loss_id += self.ce_loss_fn(logits_m, logits_m_.softmax(dim=1))

#         metric.update({'id': loss_id.data})
#         metric.update({'cs': loss_cs.data})

#         loss = loss_id + loss_cs * self.cs_w  # 去掉 DP 损失

#         return loss, metric

import torch.nn as nn
from torch.nn import functional as F
from models.resnet import resnet50, resnet18
#只有ResNet50
# class Baseline(nn.Module):
#     def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False,
#                  modality_attention=0, mutual_learning=False, **kwargs):
#         super(Baseline, self).__init__()
#         if backbone == "resnet50":
#             self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride)
#             D = 2048
#         elif backbone == "resnet18":
#             self.backbone = resnet18(pretrained=True, drop_last_stride=drop_last_stride)
#             D = 512
#         self.drop_last_stride = drop_last_stride
#         self.pattern_attention = pattern_attention
#         self.modality_attention = modality_attention
#         self.mutual_learning = mutual_learning
#         self.k_size = kwargs.get('k_size', 8)
#         self.cs_w = kwargs.get('cs_w', 1.0)
#         self.margin1 = kwargs.get('margin1', 0.01)
#         self.margin2 = kwargs.get('margin2', 0.7)
#         self.weight_KL = kwargs.get('weight_KL', 2.0)
#         self.update_rate = kwargs.get('update_rate', 0.2)
#         self.update_rate_ = self.update_rate
#         self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
#         self.cs_loss_fn = CSLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2)
#         self.visible_classifier = nn.Linear(D, num_classes, bias=False)
#         self.infrared_classifier = nn.Linear(D, num_classes, bias=False)
#         self.visible_classifier_ = nn.Linear(D, num_classes, bias=False)
#         self.visible_classifier_.weight.requires_grad_(False)
#         self.visible_classifier_.weight.data = self.visible_classifier.weight.data
#         self.infrared_classifier_ = nn.Linear(D, num_classes, bias=False)
#         self.infrared_classifier_.weight.requires_grad_(False)
#         self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data
#         self.classifier = nn.Linear(D, num_classes, bias=False)
#         self.bn = nn.BatchNorm1d(D)
        
#     def forward(self, inputs, labels=None, **kwargs):
#         cam_ids = kwargs.get('cam_ids')
#         sub = (cam_ids // 3) + (cam_ids // 6)
#         global_feat = self.backbone(inputs)
#         global_feat = global_feat.mean(dim=(2, 3))  # Global average pooling
        
        
#         if not self.training:
#             global_feat = self.bn(global_feat)
#             return global_feat
#         else:
#             return self.train_forward(global_feat, labels, sub, **kwargs)

#     def train_forward(self, feat, labels, sub, **kwargs):
#         metric = {}
#         loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels)  # CS Loss
#         feat = self.bn(feat)
#         logits = self.classifier(feat)
#         loss_id = self.ce_loss_fn(logits.float(), labels)  # ID loss
#         tmp = self.ce_loss_fn(logits.float(), labels)
#         metric.update({'ce': tmp.data})
#         cam_ids = kwargs.get('cam_ids')
#         sub = (cam_ids // 3) + (cam_ids // 6)
#         logits_v = self.visible_classifier(feat[sub == 0])  # Visible light classifier
#         loss_id += self.ce_loss_fn(logits_v.float(), labels[sub == 0])
#         logits_i = self.infrared_classifier(feat[sub == 1])  # Infrared classifier
#         loss_id += self.ce_loss_fn(logits_i.float(), labels[sub == 1])
#         logits_m = torch.cat([logits_v, logits_i], 0).float()
#         with torch.no_grad():
#             self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
#                                                     + self.infrared_classifier.weight.data * self.update_rate
#             self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
#                                                    + self.visible_classifier.weight.data * self.update_rate
#             logits_v_ = self.infrared_classifier_(feat[sub == 0])
#             logits_i_ = self.visible_classifier_(feat[sub == 1])
#             logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()
#         loss_id += self.ce_loss_fn(logits_m, logits_m_.softmax(dim=1))
#         metric.update({'id': loss_id.data})
#         metric.update({'cs': loss_cs.data})
#         loss = loss_id + loss_cs * self.cs_w
#         return loss, metric
    
    
# #ResNet50+SAAA  
# class Baseline(nn.Module):
#     def __init__(self, num_classes=None, backbone="resnet50", drop_last_stride=False, pattern_attention=False,
#                  modality_attention=0, mutual_learning=False, **kwargs):
#         super(Baseline, self).__init__()
#         self.drop_last_stride = drop_last_stride
#         self.pattern_attention = pattern_attention
#         self.modality_attention = modality_attention
#         self.mutual_learning = mutual_learning
        
#         # Backbone
#         if backbone == "resnet50":
#             self.backbone = resnet50(pretrained=True, drop_last_stride=drop_last_stride,modality_attention=modality_attention)
#             D = 2048
#         elif backbone == "resnet18":
#             self.backbone = resnet18(pretrained=True, drop_last_stride=drop_last_stride,modality_attention=modality_attention)
#             D = 512
#         self.base_dim = D
#         self.dim = D
#         # 配置参数
#         self.k_size = kwargs.get('k_size', 8)
#         self.part_num = kwargs.get('num_parts', 7)
#         self.dp = kwargs.get('dp', "l2")
#         self.dp_w = kwargs.get('dp_w', 0.5)
#         self.cs_w = kwargs.get('cs_w', 1.0)
#         self.margin1 = kwargs.get('margin1', 0.01)
#         self.margin2 = kwargs.get('margin2', 0.7)
        
#         # 注意力模块
#         self.attn_pool = SAFL(part_num=self.part_num)
#         self.SAAA = SemanticPartAttention()
#         self.S2Attention = S2Attention(channels=2048)
        
#         # 添加简单的BN层替代DualBNNeck
#         self.bn = nn.BatchNorm1d(self.base_dim + self.dim * self.part_num)
        
#         # 分类器
#         feat_dim = self.base_dim + self.dim * self.part_num
#         self.visible_classifier = nn.Linear(feat_dim, num_classes, bias=False)
#         self.infrared_classifier = nn.Linear(feat_dim, num_classes, bias=False)
#         self.visible_classifier_ = nn.Linear(feat_dim, num_classes, bias=False)
#         self.infrared_classifier_ = nn.Linear(feat_dim, num_classes, bias=False)
#         self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        
#         # 初始化动态分类器
#         self.visible_classifier_.weight.requires_grad_(False)
#         self.visible_classifier_.weight.data = self.visible_classifier.weight.data
#         self.infrared_classifier_.weight.requires_grad_(False)
#         self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data
#         # 损失函数
#         self.KL_loss_fn = nn.KLDivLoss(reduction='batchmean')
#         self.ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
#         self.cs_loss_fn = CSLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2)
        
#         # 其他参数
#         self.weight_KL = kwargs.get('weight_KL', 2.0)
#         self.update_rate = kwargs.get('update_rate', 0.2)
#         self.update_rate_ = self.update_rate

#     def forward(self, inputs, labels=None, **kwargs):
#         cam_ids = kwargs.get('cam_ids')
#         sub = (cam_ids // 3) + (cam_ids // 6)
        
#         # 特征提取
#         global_feat = self.backbone(inputs)
#         b, c, w, h = global_feat.shape
        
#         # SAAA模块
#         part_feat, attn, part_diversity_loss = self.SAAA(global_feat)
        
#         # S2Attention
#         #global_feat = self.S2Attention(global_feat)
#         global_feat = global_feat.mean(dim=(2, 3))# 特征融合
#         feats = torch.cat([part_feat, global_feat], dim=1)
        
#         # 计算DP Loss
#         if self.training:
#             masks = attn.view(b, self.part_num, w * h)
#             loss_dp = 0
#             if self.dp == "l2":
#                 for i in range(self.part_num):
#                     for j in range(i + 1, self.part_num):
#                         loss_dp += ((((masks[:, i] - masks[:, j]) ** 2).sum(dim=1) / (18 * 9)) * 0.5).sum()
#                 loss_dp = - loss_dp / (b * self.part_num * (self.part_num - 1) / 2)
#                 loss_dp *= self.dp_w
        
#         # BN归一化
        
        
#         if not self.training:
#             feats = self.bn(feats)
#             return feats
#         else:
#             return self.train_forward(feats, labels, loss_dp, sub, **kwargs)

#     def train_forward(self, feat, labels, loss_dp, sub, **kwargs):
#         metric = {}

#         # 计算各种损失
#         loss_cs, _, _ = self.cs_loss_fn(feat.float(), labels)
#         feat = self.bn(feat)
#         logits = self.classifier(feat)
#         loss_id = self.ce_loss_fn(logits.float(), labels)

#         tmp = self.ce_loss_fn(logits.float(), labels)
#         metric.update({'ce': tmp.data})

#         # 可见光和红外分类
#         logits_v = self.visible_classifier(feat[sub == 0])
#         loss_id += self.ce_loss_fn(logits_v.float(), labels[sub == 0])

#         logits_i = self.infrared_classifier(feat[sub == 1])
#         loss_id += self.ce_loss_fn(logits_i.float(), labels[sub == 1])
#         logits_m = torch.cat([logits_v, logits_i], 0).float()

#         # 动态更新分类器
#         with torch.no_grad():
#             self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
#                                                 + self.infrared_classifier.weight.data * self.update_rate
#             self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
#                 + self.visible_classifier.weight.data * self.update_rate
#             logits_v_ = self.infrared_classifier_(feat[sub == 0])
#             logits_i_ = self.visible_classifier_(feat[sub == 1])
#             logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()

#         loss_id += self.ce_loss_fn(logits_m, logits_m_.softmax(dim=1))

#         # 更新度量
#         metric.update({'id': loss_id.data})
#         metric.update({'cs': loss_cs.data})
#         metric.update({'dp': loss_dp.data})

#         # 总损失
#         loss = loss_id + loss_cs * self.cs_w + loss_dp * self.dp_w
#         return loss, metric
