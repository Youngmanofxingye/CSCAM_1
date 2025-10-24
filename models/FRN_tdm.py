import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .backbones.Conv_4 import BackBone
from .backbones.ResNet import resnet12
from .backbones.CSCAM import CSCAM
from models.TDM import TDM
import math


class FRN(nn.Module):

    def __init__(self, args=None, way=None, shots=None, resnet=False, is_pretraining=False,
                 num_cat=None, self_attention_model=None, cross_attention_model=None):
        super().__init__()

        # 兼容旧调用：如果无 args，从 way/shots/resnet 构造 Namespace
        import argparse
        if args is None:
            args = argparse.Namespace(
                train_way=way,
                train_shot=shots[0] if shots else 1,
                train_query_shot=shots[1] if shots else 15,
                resnet=resnet,
                TDM=False
            )
        self.args = args

        self.change_channel = 256
        self.resnet = resnet

        # 提前定义通道数
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
            self.feature_extractor = resnet12()
        else:
            self.change = nn.Sequential(
                nn.Conv2d(self.d, self.d, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.d),
                nn.ReLU(inplace=True)
            )
            self.feature_extractor = BackBone(num_channel)

        # --- 基本参数 ---
        self.shots = shots if shots is not None else [self.args.train_shot, self.args.train_query_shot]
        self.way = way if way is not None else self.args.train_way
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

        # TDM 初始化（仅当 TDM=True）
        if getattr(self.args, 'TDM', False):
            emb_dim = self.change_channel if self.resnet else self.d
            self.TDM = TDM(self.args, in_channels=emb_dim)
        else:
            self.TDM = None

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    # --- 特征提取 ---
    def get_feature_map(self, inp):
        batch_size = inp.size(0)
        feature_map = self.feature_extractor(inp)

        if isinstance(feature_map, tuple):
            feature_map = feature_map[-1]

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

        # 动态创建 CSCAM
        if self.CSCAM is None or getattr(self.CSCAM, "sequence_length", None) != resolution:
            emb_dim = self.change_channel if self.resnet else self.d
            self.CSCAM = CSCAM(sequence_length=resolution, embedding_dim=emb_dim).to(inp_device)

        # 划分 support/query
        support = feature_map[:way * shot]
        query = feature_map[way * shot:]

        # CSCAM 处理
        B_s, C_s, H, W = support.shape
        B_q, C_q, _, _ = query.shape
        support, query = self.CSCAM(support, query)

        # TDM 权重计算（在reshape之前）
        tdm_weight = None
        if self.TDM is not None:
            # 将support和query reshape为TDM期望的4D格式
            # support: [way*shot, C, H, W] -> [way, shot, H*W, C]
            support_4d = support.view(way, shot, C_s, H * W).permute(0, 1, 3, 2).contiguous()
            # query: [way*query_shot, C, H, W] -> [way, query_shot, H*W, C]
            query_4d = query.view(way, query_shot, C_q, H * W).permute(0, 1, 3, 2).contiguous()

            # 获取TDM权重: [way, query_shot, C]
            tdm_weight = self.TDM(support_4d, query_4d)

        # 原始FRN的reshape流程
        if self.resnet:
            support = support.contiguous().view(
                way * shot, self.change_channel, resolution).transpose(-1, -2).contiguous().view(
                way, shot * resolution, self.change_channel)
            query = query.contiguous().view(
                way * query_shot, self.change_channel, resolution).transpose(-1, -2).contiguous().view(
                way * query_shot * resolution, self.change_channel)

            # 应用TDM权重（如果存在）
            if tdm_weight is not None:
                # 计算通道重要性的全局平均
                channel_importance = tdm_weight.mean(dim=(0, 1))  # [C]
                # 归一化到合理范围 [0.5, 1.5]
                channel_importance = 0.5 + 0.5 * channel_importance
                # 应用到support和query
                support = support * channel_importance.unsqueeze(0).unsqueeze(0)
                query = query * channel_importance.unsqueeze(0)

            recon_dist = self.get_recon_dist(query=query, support=support,
                                             alpha=alpha, beta=beta, Woodbury=False)
        else:
            support = support.contiguous().view(
                way * shot, self.d, resolution).transpose(-1, -2).contiguous().view(
                way, shot * resolution, self.d)
            query = query.contiguous().view(
                way * query_shot, self.d, resolution).transpose(-1, -2).contiguous().view(
                way * query_shot * resolution, self.d)

            # 应用TDM权重（如果存在）
            if tdm_weight is not None:
                # 计算通道重要性的全局平均
                channel_importance = tdm_weight.mean(dim=(0, 1))  # [C]
                # 归一化到合理范围 [0.5, 1.5]
                channel_importance = 0.5 + 0.5 * channel_importance
                # 应用到support和query
                support = support * channel_importance.unsqueeze(0).unsqueeze(0)
                query = query * channel_importance.unsqueeze(0)

            recon_dist = self.get_recon_dist(query=query, support=support,
                                             alpha=alpha, beta=beta, Woodbury=True)

        neg_l2_dist = recon_dist.neg().contiguous().view(
            way * query_shot, resolution, way).mean(1)

        return (neg_l2_dist, support) if return_support else neg_l2_dist

    # --- 元测试 ---
    def meta_test(self, inp, way, shot, query_shot):
        neg_l2_dist = self.get_neg_l2_dist(inp=inp, way=way, shot=shot, query_shot=query_shot)
        _, max_index = torch.max(neg_l2_dist, 1)
        return max_index, neg_l2_dist

    # --- 预训练 ---
    def forward_pretrain(self, inp):
        inp_device = inp.device
        self.to(inp_device)

        feature_map = self.get_feature_map(inp)
        batch_size, resolution, _ = feature_map.shape
        alpha = self.r[0]
        beta = self.r[1]

        if getattr(self, "cat_mat", None) is None or self.cat_mat.size(1) != resolution:
            self.cat_mat = nn.Parameter(
                torch.randn(self.num_cat, resolution, self.d, device=inp_device),
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