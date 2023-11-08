# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from functions import ReverseLayerF



class DG_Network(nn.Module):
    def __init__(self, classes, channels, F1=4, D=2, domains=3):  # , hidden_dim=400
        super(DG_Network, self).__init__()
        self.dropout = 0.25  # default:0.25

        self.block1_1 = nn.Sequential(  # 1*22*512
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(1, F1, kernel_size=(1, 8), bias=False),  # 4*22*512
            nn.BatchNorm2d(F1)
        )

        self.block1_2 = nn.Sequential(  # 1*22*512
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(1, F1, kernel_size=(1, 16), bias=False),  # 4*22*512
            nn.BatchNorm2d(F1)
        )

        self.block1_3 = nn.Sequential(  # 1*22*512
            nn.ZeroPad2d((15, 16, 0, 0)),
            nn.Conv2d(1, F1, kernel_size=(1, 32), bias=False),  # 4*22*512
            nn.BatchNorm2d(F1)
        )

        self.block1_4 = nn.Sequential(  # 1*22*512
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, F1, kernel_size=(1, 64), bias=False),  # 4*22*512
            nn.BatchNorm2d(F1)
        )

        self.block2 = nn.Sequential(  # 16*22*512
            # DepthwiseConv2D
            nn.Conv2d(F1 * 4, F1 * 4 * D, kernel_size=(channels, 1), groups=F1 * 4, bias=False),
            # groups=F1 for depthWiseConv  # 32*1*512
            nn.BatchNorm2d(F1 * 4 * D),
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 4)),  # 32*1*128
            nn.Dropout(self.dropout),
        )

        self.block3_1 = nn.Sequential(  # 32*1*128
            # SeparableConv2D
            nn.ZeroPad2d((0, 1, 0, 0)),
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, kernel_size=(1, 2), groups=F1 * 4 * D, bias=False),
            # groups=F1 for depthWiseConv  # 32*1*128
            nn.BatchNorm2d(F1 * 4 * D),
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn   # 32*1*128
            nn.BatchNorm2d(F1 * 4 * D),
        )

        self.block3_2 = nn.Sequential(
            # SeparableConv2D
            nn.ZeroPad2d((1, 2, 0, 0)),
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, kernel_size=(1, 4), groups=F1 * 4 * D, bias=False),
            # groups=F1 for depthWiseConv  # 32*1*128
            nn.BatchNorm2d(F1 * 4 * D),
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn   # 32*1*128
            nn.BatchNorm2d(F1 * 4 * D),
        )

        self.block3_3 = nn.Sequential(
            # SeparableConv2D
            nn.ZeroPad2d((3, 4, 0, 0)),
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, kernel_size=(1, 8), groups=F1 * 4 * D, bias=False),
            # groups=F1 for depthWiseConv  # 32*1*128
            nn.BatchNorm2d(F1 * 4 * D),
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn   # 32*1*128
            nn.BatchNorm2d(F1 * 4 * D),
        )

        self.block3_4 = nn.Sequential(
            # SeparableConv2D
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, kernel_size=(1, 16), groups=F1 * 4 * D, bias=False),
            # groups=F1 for depthWiseConv  # 32*1*128
            nn.BatchNorm2d(F1 * 4 * D),
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, kernel_size=(1, 1), groups=1, bias=False),  # point-wise cnn   # 32*1*128
            nn.BatchNorm2d(F1 * 4 * D),
        )

        self.block4 = nn.Sequential(
            # nn.ELU(inplace=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 8)),  # 32*1*16
            nn.Dropout(self.dropout)
        )

        self.special_features1 = nn.Sequential(  # 32*1*16
            nn.Linear(3968, 400),
            # nn.Dropout(self.dropout)
        )

        self.special_features2 = nn.Sequential(  # 32*1*16
            nn.Linear(3968, 400),
            # nn.Dropout(self.dropout)
        )

        self.special_features3 = nn.Sequential(  # 32*1*16
            nn.Linear(3968, 400),
            # nn.Dropout(self.dropout)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(3968, domains),
        )

        self.classifier = nn.Sequential(
            nn.Linear(400, classes)
        )


    def forward(self, data_train1, data_train2, data_train3):  #, alpha
        data1 = data_train1.to(torch.float32)
        data2 = data_train2.to(torch.float32)
        data3 = data_train3.to(torch.float32)
        data = torch.cat((data1, data2, data3), dim=0)

        # extracting general features
        feat_1 = self.block1_1(data)   # (4,22,512)
        feat_2 = self.block1_2(data)
        feat_3 = self.block1_3(data)
        feat_4 = self.block1_4(data)
        feat = torch.cat((feat_1, feat_2, feat_3, feat_4), dim=1)  # 在第一个维度拼接(4*4,22,512)

        # extracting special features
        feature = self.block2(feat)               # (32, 1, 128)

        feature_1 = self.block3_1(feature)
        feature_2 = self.block3_2(feature)
        feature_3 = self.block3_3(feature)
        feature_4 = self.block3_4(feature)
        features = torch.cat((feature_1, feature_2, feature_3, feature_4), dim=1)  # 在第二个维度拼接(32*4,1,128)

        features = self.block4(features)          # (128, 1, 16)

        features = torch.flatten(features, 1)     # (2048)


        # extracting special features
        feat1 = self.special_features1(features)  # 全部过一遍
        feat2 = self.special_features2(features)
        feat3 = self.special_features3(features)
        Feat_s = [feat1, feat2, feat3]

        # feat for domain classifier, dom for computing domain specific loss
        # reverse_feature = ReverseLayerF.apply(feat, alpha)
        # feat_ = self.domain_classifier(reverse_feature)
        feat_ = self.domain_classifier(features)
        weight = nn.functional.softmax(feat_, dim=1)

        feat123 = torch.stack((feat1, feat2, feat3), dim=1)   # 在第二个维度扩维拼接（48,3,100）
        weighted = weight.unsqueeze(0).permute(1, 0, 2)
        weighted_feature = torch.bmm(weighted, feat123)       # 按权重叠加特征
        weighted_feature = torch.flatten(weighted_feature, 1)
        feature = self.classifier(weighted_feature)
        out = nn.functional.softmax(feature, dim=1)

        return out, weight, Feat_s, weighted_feature   #


    def predict(self, data):
        data = data.to(torch.float32)

        feat_1 = self.block1_1(data)   # (4,22,512)
        feat_2 = self.block1_2(data)
        feat_3 = self.block1_3(data)
        feat_4 = self.block1_4(data)
        feat = torch.cat((feat_1, feat_2, feat_3, feat_4), dim=1)  # 在第一个维度拼接(4*4,22,512)

        # extracting special features
        feature = self.block2(feat)           # (16, 32, 1, 128)
        feature_1 = self.block3_1(feature)
        feature_2 = self.block3_2(feature)
        feature_3 = self.block3_3(feature)
        feature_4 = self.block3_4(feature)

        features = torch.cat((feature_1, feature_2, feature_3, feature_4), dim=1)  # 在第二个维度拼接(16,128,1,128)
        features = self.block4(features)       # (16, 128, 1, 16)
        features = torch.flatten(features, 1)  # (16, 2048)

        # extracting special features
        feat1 = self.special_features1(features)  # 全部过一遍
        feat2 = self.special_features2(features)
        feat3 = self.special_features3(features)

        # feat for domain classifier, dom for computing domain specific loss
        feat_ = self.domain_classifier(features)
        weight = nn.functional.softmax(feat_, dim=1)

        feat123 = torch.stack((feat1, feat2, feat3), dim=1)  # 在第二个维度扩维拼接（48,3,100）
        weighted = weight.unsqueeze(0).permute(1, 0, 2)
        weighted_feature = torch.bmm(weighted, feat123)  # 按权重叠加特征
        weighted_feature = torch.flatten(weighted_feature, 1)
        feature = self.classifier(weighted_feature)
        out = nn.functional.softmax(feature, dim=1)

        return out, weight, weighted_feature

