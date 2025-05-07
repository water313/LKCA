import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class KD_Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.1):
        """
        改进版蒸馏损失，加入了梯度损失
        Args:
            alpha: 余弦相似度损失的权重
            beta: SAM 损失的权重
            gamma: 梯度损失的权重
        """
        super(KD_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # self.sam_loss = cal_sam(Itrue, Ifake)
        self.gra = torch.nn.L1Loss()

    def forward(self, teacher_features, student_features):
        """
        计算蒸馏损失
        Args:
            teacher_features (torch.Tensor): 教师特征，形状为 [B, C, H, W]
            student_features (torch.Tensor): 学生特征，形状为 [B, C, H, W]

        Returns:
            torch.Tensor: 损失值
        """
        assert teacher_features.shape == student_features.shape, \
            "Teacher and Student feature shapes must match"

        B, C, H, W = teacher_features.shape

        # 将 [B, C, H, W] 调整为 [B, H*W, C]
        teacher_flat = teacher_features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]
        student_flat = student_features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, H*W, C]

        # L2 归一化：在光谱维度 C 上归一化
        teacher_norm = F.normalize(teacher_flat, p=2, dim=-1)
        student_norm = F.normalize(student_flat, p=2, dim=-1)

        # 计算余弦相似度：逐像素（H*W）计算 C 维向量的点积
        cosine_similarity = torch.sum(teacher_norm * student_norm, dim=-1)  # [B, H*W]

        # 取负平均值作为损失
        cosine_loss = 1 - torch.mean(cosine_similarity)

        # 计算SAM损失
        sam_loss = cal_sam(teacher_features, student_features)

        # 计算梯度损失
        teacher_grad = cal_gradient(teacher_features)
        student_grad = cal_gradient(student_features)
        gradient_loss = self.gra(student_grad, teacher_grad)

        # 合并损失
        total_loss = self.alpha * cosine_loss + self.beta * sam_loss + self.gamma * gradient_loss
        return cosine_loss, sam_loss, total_loss



def cal_sam(Itrue, Ifake):
  esp = 1e-6
  InnerPro = torch.sum(Itrue*Ifake,1,keepdim=True)
  len1 = torch.norm(Itrue, p=2,dim=1,keepdim=True)
  len2 = torch.norm(Ifake, p=2,dim=1,keepdim=True)
  divisor = len1*len2
  mask = torch.eq(divisor,0)
  divisor = divisor + (mask.float())*esp
  cosA = torch.sum(InnerPro/divisor,1).clamp(-1+esp, 1-esp)
  sam = torch.acos(cosA)
  sam = torch.mean(sam) / np.pi
  return sam


def cal_gradient_c(x):
    c_x = x.size(1)
    g = x[:, 1:, 1:, 1:] - x[:, :c_x - 1, 1:, 1:]
    return g


def cal_gradient_x(x):
    c_x = x.size(2)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, :c_x - 1, 1:]
    return g


def cal_gradient_y(x):
    c_x = x.size(3)
    g = x[:, 1:, 1:, 1:] - x[:, 1:, 1:, :c_x - 1]
    return g


def cal_gradient(inp):
    x = cal_gradient_x(inp)
    y = cal_gradient_y(inp)
    c = cal_gradient_c(inp)
    g = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2) + torch.pow(c, 2) + 1e-6)
    return g
