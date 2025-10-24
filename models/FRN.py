import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union
from .backbones.Conv_4 import BackBone  # 显式导入 BackBone 类
from .backbones.ResNet import resnet12  # 假设 ResNet 模块有 resnet12 函数，显式导入
from .backbones.CSCAM import CSCAM  # 显式导入 CSCAM 类


class FRN(nn.Module):

    def __init__(self, way=None, shots=None, resnet=False, is_pretraining=False,
                 num_cat=None, self_attention_model=None, cross_attention_model=None):
        super().__init__()

        self.change_channel = 256
        self.resnet = resnet

        # ✅ 提前定义通道数，防止属性错误
        if resnet:
            num_channel = 640
        else:
            num_channel = 64
        self.d = num_channel

        # --- 主干网络 ---
        if resnet:
            self.change = nn.Sequential(
                nn.Conv2d(self.d, self.change_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.change_channel),
                nn.ReLU(inplace=True)
            )
            self.feature_extractor = resnet12()  # 使用导入的函数
        else:
            self.change = nn.Sequential(
                nn.Conv2d(self.d, self.d, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.d),
                nn.ReLU(inplace=True)
            )
            self.feature_extractor = BackBone(num_channel)  # 使用导入的类

        # --- 基本参数 ---
        self.shots = shots
        self.way = way
        self.self_attention_model = self_attention_model
        self.cross_attention_model = cross_attention_model

        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        self.w1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.s1 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
        self.s2 = nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)

        self.r = nn.Parameter(torch.zeros(2), requires_grad=not is_pretraining)

        # --- 预训练 ---
        if is_pretraining:
            self.num_cat = num_cat
            self.cat_mat = nn.Parameter(
                torch.randn(self.num_cat, 25, self.d), requires_grad=True
            )
        if not is_pretraining:
            emb_dim = self.change_channel if self.resnet else self.d
            self.CSCAM = CSCAM(sequence_length=25, embedding_dim=emb_dim)
        else:
            self.CSCAM = None

    # --- 特征提取 ---
    def get_feature_map(self, inp):
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)

        if isinstance(feature_map, tuple):
            feature_map = feature_map[-1]  # 使用最后一个特征图 x4 (5x5)

        if self.resnet:
            feature_map = feature_map / np.sqrt(640)

        return feature_map.contiguous().view(batch_size, self.d, -1).permute(0, 2, 1).contiguous()

    # --- 重建距离 ---
    def get_recon_dist(self, query, support, alpha, beta, Woodbury=True):
        reg = support.size(1) / support.size(2)
        lam = reg * alpha.exp() + 1e-6
        rho = beta.exp()
        st = support.permute(0, 2, 1)

        if Woodbury:
            sts = st.matmul(support)
            m_inv = (sts + torch.eye(sts.size(-1), device=sts.device).unsqueeze(0).mul(lam)).inverse()
            hat = m_inv.matmul(sts)
        else:
            sst = support.matmul(st)
            m_inv = (sst + torch.eye(sst.size(-1), device=sst.device).unsqueeze(0).mul(lam)).inverse()
            hat = st.matmul(m_inv).matmul(support)

        Q_bar = query.matmul(hat).mul(rho)
        dist = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0)
        return dist

    # --- 负 L2 距离 ---
    def get_neg_l2_dist(self, inp, way, shot, query_shot, return_support=False):
        inp_device = inp.device
        self.to(inp_device)

        alpha = self.r[0]
        beta = self.r[1]

        feature_map = self.get_feature_map(inp)
        B, HW, C = feature_map.shape
        resolution = HW
        side = int(np.sqrt(resolution))

        feature_map = feature_map.transpose(-1, -2).contiguous().view(
            (way * (shot + query_shot)), self.d, side, side)
        feature_map = self.change(feature_map)

        # ✅ 动态创建 CSCAM
        if self.CSCAM is None or getattr(self.CSCAM, "sequence_length", None) != resolution:
            emb_dim = self.change_channel if self.resnet else self.d
            self.CSCAM = CSCAM(sequence_length=resolution, embedding_dim=emb_dim).to(inp_device)

        # 划分 support/query
        support = feature_map[:way * shot]
        query = feature_map[way * shot:]

        # ✅ 直接传入 4 维 [N, C, H, W] 到 CSCAM（移除展平和恢复）
        B_s, C_s, H, W = support.shape
        B_q, C_q, _, _ = query.shape
        support, query = self.CSCAM(support, query)

        # --- 计算距离 ---
        if self.resnet:
            support = support.contiguous().view(
                way * shot, self.change_channel, resolution).transpose(-1, -2).contiguous().view(
                way, shot * resolution, self.change_channel)
            query = query.contiguous().view(
                way * query_shot, self.change_channel, resolution).transpose(-1, -2).contiguous().view(
                way * query_shot * resolution, self.change_channel)
            recon_dist = self.get_recon_dist(query=query, support=support,
                                             alpha=alpha, beta=beta, Woodbury=False)
        else:
            support = support.contiguous().view(
                way * shot, self.d, resolution).transpose(-1, -2).contiguous().view(
                way, shot * resolution, self.d)
            query = query.contiguous().view(
                way * query_shot, self.d, resolution).transpose(-1, -2).contiguous().view(
                way * query_shot * resolution, self.d)
            recon_dist = self.get_recon_dist(query=query, support=support,
                                             alpha=alpha, beta=beta, Woodbury=True)

        neg_l2_dist = recon_dist.neg().contiguous().view(
            way * query_shot, resolution, way).mean(1)

        return (neg_l2_dist, support) if return_support else neg_l2_dist

    # --- 元测试 ---
    def meta_test(self, inp, way, shot, query_shot):
        neg_l2_dist = self.get_neg_l2_dist(inp=inp, way=way, shot=shot, query_shot=query_shot)
        _, max_index = torch.max(neg_l2_dist, 1)
        return max_index, neg_l2_dist  # 修改：返回两个值以匹配 eval.py 的解包

    # --- 预训练 ---
    def forward_pretrain(self, inp):
        inp_device = inp.device
        self.to(inp_device)

        feature_map = self.get_feature_map(inp)
        batch_size, resolution, _ = feature_map.shape
        alpha = self.r[0]
        beta = self.r[1]

        if getattr(self, "cat_mat", None) is None or self.cat_mat.size(1) != resolution:
            self.cat_mat = nn.Parameter(torch.randn(self.num_cat, resolution, self.d, device=inp_device),
                                        requires_grad=True)

        feature_map = feature_map.contiguous().view(batch_size * resolution, self.d)
        recon_dist = self.get_recon_dist(query=feature_map, support=self.cat_mat,
                                         alpha=alpha, beta=beta)
        neg_l2_dist = recon_dist.neg().contiguous().view(
            batch_size, resolution, self.num_cat).mean(1)
        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction

    # --- 前向传播 ---
    def forward(self, inp):
        inp_device = inp.device
        self.to(inp_device)

        neg_l2_dist, support = self.get_neg_l2_dist(inp=inp,
                                                    way=self.way,
                                                    shot=self.shots[0],
                                                    query_shot=self.shots[1],
                                                    return_support=True)
        logits = neg_l2_dist * self.scale
        log_prediction = F.log_softmax(logits, dim=1)
        return log_prediction, support