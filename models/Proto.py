import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as torch_models
import numpy as np
from .backbones import Conv_4, ResNet
from .backbones.CSCAM import CSCAM  # 显式导入 CSCAM 类
import argparse
import numpy as np


def pdist(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return dist


class Proto(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False):

        super().__init__()
        self.change_channel = 256
        num_channel = 64  # 默认 Conv_4 通道
        if resnet:
            self.emb_dim = self.change_channel  # 修改：ResNet 经过 change 后的通道
            self.change = nn.Sequential(
                nn.Conv2d(640, self.change_channel, 3, padding=1, bias=False),  # 修复：输入通道为 640
                nn.BatchNorm2d(self.change_channel),
                nn.ReLU(inplace=True)
            )
            self.CSCAM = CSCAM(sequence_length=25, embedding_dim=self.change_channel)
            self.feature_extractor = ResNet.resnet12()
            spatial_size = 1  # ResNet 经过 avg_pool2d 后空间为 1x1
        else:
            self.emb_dim = num_channel  # Conv_4 通道
            self.change = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_channel),
                nn.ReLU(inplace=True)
            )
            self.CSCAM = CSCAM(sequence_length=25, embedding_dim=num_channel)
            self.feature_extractor = Conv_4.BackBone(num_channel)
            spatial_size = 25  # Conv_4 无池化，5x5=25
        self.dim = self.emb_dim * spatial_size  # 修改：动态计算总嵌入维度 (ResNet: 256*1=256; Conv_4: 64*25=1600)

        self.shots = shots
        self.way = way
        self.resnet = resnet
        # temperature scaling, correspond to gamma in the paper
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def get_feature_vector(self, inp):
        feature_map = self.feature_extractor(inp)

        # 处理 Conv_4 返回的 tuple，使用最后一个特征图
        if isinstance(feature_map, tuple):
            feature_map = feature_map[-1]

        return feature_map

    def get_neg_l2_dist(self, inp, way, shot, query_shot):
        feature_map = self.get_feature_vector(inp)
        feature_map = self.change(feature_map)

        support = feature_map[:way * shot]
        query = feature_map[way * shot:]

        support, query = self.CSCAM(support, query)

        if self.resnet:
            support = F.avg_pool2d(input=support, kernel_size=support.size(-1))
            query = F.avg_pool2d(input=query, kernel_size=query.size(-1))
            support = support.contiguous().view(way, shot, -1)
            query = query.contiguous().view(way * query_shot, -1)
        else:
            support = support.contiguous().view(way, shot, -1)
            query = query.contiguous().view(way * query_shot, -1)
        centroid = torch.mean(support, 1)  # way,dim
        neg_l2_dist = pdist(query, centroid).neg().view(way * query_shot, way)  # way*query_shot,way

        return neg_l2_dist

    def meta_test(self, inp, way, shot, query_shot):
        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=way,
                                           shot=shot,
                                           query_shot=query_shot)
        _, max_index = torch.max(neg_l2_dist, 1)
        return max_index, neg_l2_dist  # 修改：返回两个值以匹配 eval.py 的解包

    def forward(self, inp):
        neg_l2_dist = self.get_neg_l2_dist(inp=inp,
                                           way=self.way,
                                           shot=self.shots[0],
                                           query_shot=self.shots[1])

        logits = neg_l2_dist / self.dim * self.scale
        log_prediction = F.log_softmax(logits, dim=1)

        return log_prediction