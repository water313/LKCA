import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class LKA(nn.Module):
    def __init__(self, inc=3, k1=5, k2=7, dilation=3):
        super(LKA, self).__init__()
        self.conv = weight_norm(nn.Conv2d(inc, inc, k1, padding=k1 // 2, groups=inc))
        self.conv_spatial = weight_norm(
            nn.Conv2d(inc, inc, k2, stride=1, padding=k2 // 2 * dilation, groups=inc, dilation=dilation))
        self.conv_channel = weight_norm(nn.Conv2d(inc * 3, inc, 1, groups=4))
        self.ca = CALayer(inc * 3)

    def forward(self, x):
        u = x.clone()
        attn1 = self.conv(x)
        attn2 = self.conv_spatial(attn1)
        attn = torch.cat([attn1, attn2, u], 1)
        attn = self.ca(attn)
        attn = self.conv_channel(attn)
        return u * attn


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LKB(nn.Module):
    def __init__(self, n_feats):
        super(LKB, self).__init__()

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.proj_1 = nn.Conv2d(n_feats, n_feats, 1)
        self.spatial_gating_unit = LKA(n_feats)
        self.proj_2 = nn.Conv2d(n_feats, n_feats, 1)
        self.drop = DropPath(0.2)

    def forward(self, x):
        shorcut = x.clone()
        x = F.gelu(self.proj_1(self.norm(x)))
        x = self.spatial_gating_unit(x)
        x = F.gelu(self.proj_2(x))
        x = self.drop(x * self.scale) + shorcut
        return x


def drop_path(x, keep_prob=1.0, inplace=False):
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)
    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x


class DropPath(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class Student(nn.Module):
    def __init__(self, n_colors=103, n_feats=128, n_lkb=3, scale=4):
        super(Student, self).__init__()

        self.n_colors = n_colors
        self.n_feats = n_feats
        self.n_lkb = n_lkb
        self.scale = scale

        self.conv = nn.Sequential(nn.Conv2d(n_colors, n_feats, 3, 1, 1, bias=True),
                                  nn.GELU())
        self.body = nn.Sequential(*[LKB(n_feats) for _ in range(n_lkb)])

        out_feats = n_colors * scale ** 2
        self.scale = nn.Parameter(torch.zeros((1, out_feats, 1, 1)), requires_grad=True)

        self.tail = nn.Conv2d(n_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=True, groups=8)

        self.up = nn.PixelShuffle(scale)

    def forward(self, x, lms):
        x = self.conv(x)
        x = self.body(x)
        x = self.tail(x) * self.scale
        upsampled_feature = self.up(x)  # 提取 PixelShuffle 上采样后的特征图
        output = upsampled_feature + lms
        return output, upsampled_feature  # 返回最终输出和上采样后的特征图
