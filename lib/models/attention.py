"""
轻量级注意力模块
"""
import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    Simple Attention Module (SimAM)
    无参数的注意力机制，通过能量函数计算注意力权重
    
    Paper: SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        
        n = w * h - 1
        
        # 计算均值和方差
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # 应用sigmoid激活得到注意力权重
        return x * self.activaton(y)


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA)
    轻量级通道注意力，使用1D卷积
    """
    def __init__(self, channels, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化
        y = self.avg_pool(x)
        
        # 1D卷积
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        # 应用sigmoid
        y = self.sigmoid(y)
        
        return x * y.expand_as(x)


class CBAM_Spatial(nn.Module):
    """
    CBAM空间注意力模块（轻量版）
    """
    def __init__(self, kernel_size=7):
        super(CBAM_Spatial, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道维度上的最大值和平均值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接并卷积
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        
        return x * self.sigmoid(y)
