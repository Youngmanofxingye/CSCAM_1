import torch
import torch.nn as nn
import torch.nn.functional as F


class SandGlassBlock(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_c,
                                 out_features=in_c * 2,
                                 bias=False)
        self.bn1 = nn.BatchNorm1d(in_c * 2)
        self.linear2 = nn.Linear(in_features=in_c * 2,
                                 out_features=in_c,
                                 bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.linear1(x)
        # 处理 batch_size=1 情况，使用 running stats 无更新
        if self.training and output.size(0) == 1:
            output = F.batch_norm(output, self.bn1.running_mean, self.bn1.running_var,
                                  self.bn1.weight, self.bn1.bias, training=False,
                                  momentum=0.0, eps=self.bn1.eps)
        else:
            output = self.bn1(output)
        output = self.relu(output)
        output = self.linear2(output)
        output = torch.tanh(output)
        output = 1 + output

        return output


class TDM(nn.Module):

    def __init__(self, args, in_channels=None):
        """
        Args:
            args: 参数对象
            in_channels: 显式指定输入通道数（优先级高于 args.resnet）
        """
        super().__init__()

        self.args = args

        # 优先使用显式传入的通道数，否则根据 args.resnet 判断
        if in_channels is not None:
            self.in_c = in_channels
        elif hasattr(args, 'resnet') and args.resnet:
            self.in_c = 256
        else:
            self.in_c = 64

        self.prt_self = SandGlassBlock(self.in_c)
        self.prt_other = SandGlassBlock(self.in_c)
        self.qry_self = SandGlassBlock(self.in_c)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def add_noise(self, input):
        if self.training and hasattr(self.args, 'noise') and self.args.noise:
            noise_value = getattr(self.args, 'noise_value', 0.1)
            noise = ((torch.rand(input.shape).to(input.device) - .5) * 2) * noise_value
            input = input + noise
            input = input.clamp(min=0., max=2.)

        return input

    def dist(self, input, spt=False, normalize=True):

        if spt:
            way, c, m = input.shape
            input_C_gap = input.mean(dim=-2)

            input = input.reshape(way * c, m)
            input = input.unsqueeze(dim=1)
            input_C_gap = input_C_gap.unsqueeze(dim=0)

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            if normalize:
                dist = dist / m
            dist = dist.reshape(way, c, -1)
            dist = dist.transpose(-1, -2)

            indices_way = torch.arange(way, device=dist.device)
            dist_self = dist[indices_way, indices_way]

            if way == 1:
                # way=1 时，dist_self [1, c]，dist_other 空形状 [1, 0, c]
                dist_self = dist_self.unsqueeze(0) if dist_self.dim() == 1 else dist_self
                dist_other = torch.zeros(way, way - 1, c, device=dist.device, dtype=dist.dtype)
            else:
                indices_1 = indices_way.repeat_interleave((way - 1))
                indices_2 = []
                for i in indices_way:
                    indices_2_temp = torch.cat((indices_way[:i], indices_way[i + 1:]),
                                               dim=-1)
                    indices_2.append(indices_2_temp)
                indices_2 = torch.cat(indices_2, dim=0)

                dist_other = dist[indices_1, indices_2]
                dist_other = dist_other.view(way, way - 1, -1)

            return dist_self, dist_other

        else:
            batch, c, m = input.shape
            input_C_gap = input.mean(dim=-2).unsqueeze(dim=-2)

            dist = torch.sum(torch.pow(input - input_C_gap, 2), dim=-1)
            if normalize:
                dist = dist / m

            return dist

    def weight(self, spt, qry):
        # 自动检测模型类型
        if hasattr(self.args, 'model') and self.args.model == 'Proto':
            way, shot, c, m = spt.shape
            batch, _, _, _ = qry.shape

            prt = spt.mean(dim=1)
            qry = qry.squeeze(dim=1)

        else:  # FRN 或其他
            # 期望格式: spt [way, shot, m, c], qry [way, batch, m, c]
            # 其中 m 是空间维度(resolution), c 是通道维度
            if spt.dim() != 4 or qry.dim() != 4:
                raise ValueError(f"Expected 4D tensors, got spt: {spt.shape}, qry: {qry.shape}")

            way, shot, m, c = spt.shape
            _, batch, _, _ = qry.shape

            # 检查通道数是否匹配
            if c != self.in_c:
                raise ValueError(
                    f"Channel mismatch: TDM expects {self.in_c} channels but got {c}. "
                    f"Please check if args.resnet is correctly set. "
                    f"Current args.resnet={getattr(self.args, 'resnet', 'not set')}"
                )

            prt = spt.mean(dim=1)  # [way, m, c]
            prt = prt.transpose(-1, -2)  # [way, c, m]

            qry = qry.mean(dim=0)  # [batch, m, c] - 对 way 维度取平均
            qry = qry.transpose(-1, -2)  # [batch, c, m]

        # 计算距离
        dist_prt_self, dist_prt_other = self.dist(prt, spt=True)
        dist_qry_self = self.dist(qry)

        dist_prt_self = dist_prt_self.view(-1, c)

        # way=1 时，避免 min on empty，设置 dist_prt_other 为 zeros
        if way == 1:
            dist_prt_other = torch.zeros(way, c, device=dist_prt_self.device,
                                         dtype=dist_prt_self.dtype)
        else:
            dist_prt_other, _ = dist_prt_other.min(dim=-2)
            dist_prt_other = dist_prt_other.view(-1, c)

        dist_qry_self = dist_qry_self.view(-1, c)

        # 生成权重
        weight_prt_self = self.prt_self(dist_prt_self)
        weight_prt_self = weight_prt_self.view(way, 1, c)

        weight_prt_other = self.prt_other(dist_prt_other)
        weight_prt_other = weight_prt_other.view(way, 1, c)

        weight_qry_self = self.qry_self(dist_qry_self)
        weight_qry_self = weight_qry_self.view(1, batch, c)

        alpha_prt = 0.5
        alpha_prt_qry = 0.5

        beta_prt = 1. - alpha_prt
        beta_prt_qry = 1. - alpha_prt_qry

        weight_prt = alpha_prt * weight_prt_self + beta_prt * weight_prt_other
        weight = alpha_prt_qry * weight_prt + beta_prt_qry * weight_qry_self

        # 返回 [way, batch, c]
        return weight

    def forward(self, spt, qry):
        weight = self.weight(spt, qry)
        weight = self.add_noise(weight)

        return weight