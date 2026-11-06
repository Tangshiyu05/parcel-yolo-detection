# -*- coding: utf-8 -*-
"""
类型感知注意力模块（Type-Aware Attention Module, TAM）
由通道支路、空间支路、类型原型支路三部分组成，用于插入 YOLOv8s 的
Neck 输出与检测头之间，强化硬纸箱/软包/信封的材质判别并应对堆叠遮挡。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力：GAP/GMP 并行 + 共享 MLP（SE 思路）。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """空间注意力：堆叠遮挡时聚焦可见主体（CBAM 空间分支）。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class TypeAwareAttention(nn.Module):
    """
    TAM 完整结构
    :param channels: 输入特征通道数
    :param num_classes: 类别数（硬纸箱/软包/信封 = 3）
    """

    def __init__(self, channels: int, num_classes: int = 3, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
        # 类型原型：每个类别一个可学习原型向量
        self.prototypes = nn.Parameter(torch.randn(num_classes, channels))
        nn.init.xavier_uniform_(self.prototypes)
        # 将类别响应映射回通道调制权重
        self.proto_fc = nn.Conv2d(num_classes, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc = self.channel_att(x)                       # 通道权重
        ms = self.spatial_att(x)                       # 空间权重

        # 类型原型支路：特征与各原型相似度 -> 类别响应图 -> 通道调制
        b, c, h, w = x.shape
        feat = F.normalize(x.view(b, c, -1), dim=1)               # (B,C,HW)
        proto = F.normalize(self.prototypes, dim=1)                # (K,C)
        sim = torch.einsum("kc,bcn->bkn", proto, feat)            # (B,K,HW)
        sim = sim.view(b, -1, h, w)
        mt = torch.sigmoid(self.proto_fc(sim))                     # (B,C,H,W)

        refined = x * mc * ms * mt
        return x + refined                                        # 残差连接
