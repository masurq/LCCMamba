import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
try:
    from thop import clever_format, profile
except ImportError:  # Optional; used only by the module's standalone profiler.
    clever_format = profile = None
from models.new_model.Module.gcn_1 import GCNSpatial, BatchNorm2d
import torch.nn.functional as F

class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class GCNFeatureFusion(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super().__init__()
        self.gcn_s = GCNSpatial(in_channels)
        self.channel_weights = SEModel(in_channels)
        self.spatial_in = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.apply(self._init_weights)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels *3, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),  # 修正：ReLU不应有参数
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(512, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),  # 修正：ReLU不应有参数
        )
    @classmethod
    def _init_weights(cls, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2,x3):
        orisize = x1.shape
        #两流特征经过图卷积
        x3 = F.interpolate(x3, size=orisize[2:], mode='bilinear', align_corners=False)
        x3=self.c3(x3)
        x_spatial_origin = torch.cat((x1, x2), dim=1)
        x_spatial_in=self.spatial_in( x_spatial_origin)
        x_spatial_fuse=self.gcn_s( x_spatial_in)
        # 预测图计算通道注意力
        c_weight = self.channel_weights(x3)
        #print(c_weight.shape)
        # 预测图赋值改变原始流
        x_spatial = x_spatial_fuse * c_weight + x_spatial_in
        #x_spatial = x_spatial_fuse + x_spatial_in
        c1_weight = self.avg_pool(x_spatial_origin)
        c1_weight = self.conv1d(c1_weight.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_channel = x_spatial_origin * c1_weight.expand_as(x_spatial_origin)

        out = torch.cat((x_spatial, x_channel), 1)
        out = self.output(out)
        return out

if __name__ == '__main__':
    model = GCNFeatureFusion(256, 256).cuda()

    left = torch.randn(1, 256, 16, 16).cuda()
    mid=torch.randn(1, 256, 16, 16).cuda()
    right = torch.randn(1,512, 8, 8).cuda()
    out = model(left,mid,right)
    print(out.shape)
    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left,mid,right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
