#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2024-01-15 :27:25
# @Author  : Tian Yuyu
# @Link    : https://ieeexplore.ieee.org/document/9614130
# @Version : $Id$

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d
"""
GCN 模塊
"""


class ECANet(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet, self).__init__()
        self.in_channels = in_channels
        self.fgp = nn.AdaptiveAvgPool2d((1, 1))
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        output = self.fgp(x)
        output = output.squeeze(-1).transpose(-1, -2)
        output = self.con1(output).transpose(-1, -2).unsqueeze(-1)
        output = self.act1(output)
        output = torch.multiply(x, output)
        return output


class Sobel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Sobel, self).__init__()
        kernel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        kernel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        kernel_x = torch.FloatTensor(kernel_x).expand(out_channel, in_channel, 3, 3)
        kernel_x = kernel_x.type(torch.cuda.FloatTensor)
        kernel_y = torch.cuda.FloatTensor(kernel_y).expand(out_channel, in_channel, 3, 3)
        kernel_y = kernel_y.type(torch.cuda.FloatTensor)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).clone()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).clone()
        self.softmax = nn.Softmax()

    def forward(self, x):
        b, c, h, w = x.size()
        sobel_x = F.conv2d(x, self.weight_x, stride=1, padding=1)
        sobel_x = torch.abs(sobel_x)
        sobel_y = F.conv2d(x, self.weight_y, stride=1, padding=1)
        sobel_y = torch.abs(sobel_y)
        if c == 1:
            sobel_x = sobel_x.view(b, h, -1)
            sobel_y = sobel_y.view(b, h, -1).permute(0, 2, 1)
        else:
            sobel_x = sobel_x.view(b, c, -1)
            print('sobel_x:', sobel_x.shape)
            sobel_y = sobel_y.view(b, c, -1).permute(0, 2, 1)
            print('sobel_y:', sobel_y.shape)
        sobel_A = torch.bmm(sobel_x, sobel_y)
        print('sobel_A:', sobel_A.shape)
        sobel_A = self.softmax(sobel_A)
        return sobel_A


class Laplace(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Laplace, self).__init__()
        kernel_l = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        kernel_l = torch.FloatTensor(kernel_l).expand(out_channel, in_channel, 3, 3)
        # kernel_l = [[-2.0,-4.0,-4.0,-4.0,-2.0],[-4.0,0.0,8.0,0.0,-4.0],[-4.0,8.0,24.0,8.0,-4.0],[-4.0,0.0,8.0,0.0,-4.0],[-2.0,-4.0,-4.0,-4.0,-2.0],]
        # kernel_l = torch.FloatTensor(kernel_l).expand(out_channel,in_channel,5,5)
        kernel_l = kernel_l.type(torch.cuda.FloatTensor)
        self.weight_l = nn.Parameter(data=kernel_l, requires_grad=False).clone()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        laplace_l = F.conv2d(x, self.weight_l, stride=1, padding=1)
        laplace_l = torch.abs(laplace_l)

        # if c == 1:
        #     laplace_r = laplace_l.view(b, h, -1)
        #     laplace_c = laplace_l.view(b, h, -1).permute(0,2,1)
        # else:
        laplace_r = laplace_l.view(b, c, -1)
        laplace_c = laplace_l.view(b, c, -1).permute(0, 2, 1)

        laplace_A = torch.bmm(laplace_r, laplace_c)
        # print(laplace_A)
        # laplce_A = laplace_l.repeat(b, c, c)
        laplace_A = self.softmax(laplace_A)
        # print(laplace_A, '\n')
        return laplace_A


class GCNSpatial(nn.Module):
    def __init__(self, channels):
        super(GCNSpatial, self).__init__()
        # self.adjacency = Sobel(channels , channels)
        self.adjacency = Laplace(channels, channels)
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            A1 = A[i].to(device="cpu")
            # A1 = A[i].to(device="cuda:0")
            I = torch.eye(c, im)
            A1 = A1 + I
            # degree matrix
            d = A1.sum(1)
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)

        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)

        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()
        A = self.adjacency(x)
        # print('a',A.shape)
        A = self.normalize(A)
        x = x.view(b, c, -1)
        x = F.relu(self.fc1(A.bmm(x)))
        x = F.relu(self.fc2(A.bmm(x)))
        # x = self.fc3(A.bmm(x))
        out = x.view(b, c, h, w)
        return out


class Combine(nn.Module):
    def __init__(self, channels):
        super(Combine, self).__init__()
        self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax()

    def normalize(self, A):
        b, c, im = A.size()
        out = np.array([])
        for i in range(b):
            A1 = A[i].to(device="cpu")
            # A1 = A[i].to(device="cuda:0")
            I = torch.eye(c, im)
            A1 = A1 + I
            # degree matrix
            d = A1.sum(1)
            # D = D^-1/2
            D = torch.diag(torch.pow(d, -0.5))
            new_A = D.mm(A1).mm(D).detach().numpy()
            out = np.append(out, new_A)
        out = out.reshape(b, c, im)
        normalize_A = torch.from_numpy(out)
        normalize_A = normalize_A.type(torch.cuda.FloatTensor)
        return normalize_A

    def forward(self, x):
        b, c, h, w = x.size()
        # A = self.adjacency(x)
        A = self.softmax(torch.bmm(x.view(b, c, -1).permute(0, 2, 1), x.view(b, c, -1)))
        A = self.normalize(A)
        x = x.view(b, c, -1)
        x = F.relu(self.fc1(x.bmm(A)))
        x = F.relu(self.fc2(x.bmm(A)))
        # x = self.fc3(x.bmm(A))
        out = x.view(b, c, h, w)
        return out


# class GCNChannel(nn.Module):
#     def __init__(self , channels):
#         super(GCNChannel,self).__init__()
#         self.input = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
#             BatchNorm2d(channels),
#             nn.ReLU(inplace=True)
#             )
#         # self.adjacency = Sobel(1,1)
#         self.adjacency = Laplace(1,1)
#         self.fc1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
#         self.fc2 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
#         self.fc3 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
#
#     # def sobel_channel(self,x):
#     #     b,c,h,w = x.size()
#     #     sobel = Sobel(h*w,h*w)
#     #     return sobel
#
#     def pre(self,x):
#         b,c,h,w = x.size()
#         x = x.view(b, c, -1).permute(0, 2, 1)
#         x = x.view(b, 1, h*w, c)
#         return x
#
#     def normalize(self,A):
#         b,c,im = A.size()
#         out = np.array([])
#         for i in range(b):
#             # A = A = I
#             # A1 = A[i].to(device="cpu")
#             A1 = A[i].to(device="cuda:0")
#             I = torch.eye(c,im)
#             A1 = A1 + I
#             # degree matrix
#             d = A1.sum(1)
#             # D = D^-1/2
#             D = torch.diag(torch.pow(d , -0.5))
#             new_A = D.mm(A1).mm(D).detach().numpy()
#             out = np.append(out,new_A)
#         out = out.reshape(b,c,im)
#         normalize_A = torch.from_numpy(out)
#         normalize_A = normalize_A.type(torch.cuda.FloatTensor)
#         return normalize_A
#
# def forward(self,x):
#     b,c,h,w = x.size()
#     x = self.input(x)
#     b,c,h1,w1 = x.size()
#     x = self.pre(x)
#     # A = self.sobel_channel(x)
#     A = self.adjacency(x)
#     A = self.normalize(A)
#     x = x.view(b, -1, c)
#     x = F.relu(self.fc1(A.bmm(x).permute(0,2,1))).permute(0,2,1)
#     x = F.relu(self.fc2(A.bmm(x).permute(0,2,1))).permute(0,2,1)
#     x = self.fc3(A.bmm(x).permute(0,2,1))
#     out = x.view(b, c, h1 ,w1)
#     out = F.interpolate(out, size=(h,w), mode='bilinear', align_corners=True)
#     return out



# 3. 修正后的 SCGCN 模型（修正了输出层的维度错误）
class SCGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(SCGCN, self).__init__()

        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels)
        )

        self.spatial_in = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.gcn_s = GCNSpatial(out_channels)
        self.conv2d = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1)

        # 修正：最后一个BatchNorm2d的num_features应为num_classes，而不是out_channels
        self.output = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # 修正：ReLU不应有参数
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, bias=True),
            nn.BatchNorm2d(num_classes),  # 修正：此处应为num_classes
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.gcn_combine = Combine(num_classes)  # 修正：Combine的输入通道数应与output输出一致

    def forward(self, x):
        # GCN_Spatial路径
        x_spatial_in = self.spatial_in(x)
        print('1',x_spatial_in.shape)
        x_spatial = self.gcn_s(x_spatial_in)
        print('2',x_spatial.shape)
        x_spatial = x_spatial_in + x_spatial  # 残差连接

        # GCN_Channel路径（通道注意力）
        c_weight = self.avg_pool(x)
        c_weight = self.conv1d(c_weight.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_channel = x * c_weight.expand_as(x)

        # 特征融合与输出
        out = torch.cat((x_spatial, x_channel), 1)  # 在通道维度拼接
        out = self.output(out)
        out = self.gcn_combine(out)
        return out


# 4. 示例使用方式
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    batch_size = 2
    in_channels = 512
    out_channels =512
    num_classes = 8
    height, width = 16, 16
    model = SCGCN(in_channels=in_channels, out_channels=out_channels, num_classes=num_classes).to(device)
    x = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"输入数据形状: {x.shape}")
    output = model(x)
    print(f"输出数据形状: {output.shape}")
    from thop import clever_format, profile
    # 计算FLOPs和参数量
    flops, params = profile(model, (x,), verbose=False)
    # 使用clever_format美化输出
    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")
    print(f"FLOPs: {flops_formatted}, Params: {params_formatted}")
    # 添加到您的测试代码中




