import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F
from models.new_model.Module.GCNFeaturec3 import GCNFeatureFusion
from models.new_model.encoders.mamba.fuse_modules import DGFM as MFB
from models.new_model.encoders.mamba.fuse_modules import DGFMSpatialGateOnly as MFBSpatialGateOnly
from models.new_model.encoders.mamba.fuse_modules import DGFMChannelGateOnly as MFBChannelGateOnly
from models.new_model.encoders.mamba.fuse_modules import DGFMGateOnlyNoSSM as MFBGateOnlyNoSSM
from models.new_model.encoders.mamba.fuse_modules import DGFMSSMNoGate as MFBSSMNoGate
from models.new_model.encoders.mamba.fuse_modules import DGFMConcatFusion as MFBConcatFusion
from models.new_model.encoders.mamba.fuse_modules import DGFMNoDisagree as MFBNoDisagree
from models.new_model.encoders.mamba.fuse_modules import FourierEdgeEnhancementModule as edge
from models.new_model.encoders.mamba.fuse_modules import EGM_FDEG as egm_fdeg
class MLP(nn.Module):
    """
    Linear Embedding:
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # B C H W -> B HW C
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane * 2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x1, x2):
        low_feature, h_feature = x1, x2
        #print(low_feature.shape, h_feature.shape)
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.interpolate(h_feature, size=size, mode="bilinear", align_corners=True)
        #print("1",low_feature.shape, h_feature.shape)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output





class DecoderHead_mloss(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead_mloss, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.Aligned_c4 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c3 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c2 = AlignedModule(embedding_dim, embedding_dim // 2)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred3 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.MFB1 = MFB(hidden_dim=c3_in_channels, drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                        d_state=16, mlp_ratio=2.)
        self.MFB2 = MFB(hidden_dim=c2_in_channels, drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                        d_state=16, mlp_ratio=2.)
        self.MFB3 = MFB(hidden_dim=c1_in_channels, drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                        d_state=16, mlp_ratio=2.)

    def forward(self, inputs,input2):

        x4,x3,x2,x1=inputs
        y4,y3,y2,y1=input2

        c1=(x1+y1)
        orisize = x4.shape
        n, _, h, w = x4.shape
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _x1 = self.linear_pred1(_c1)

        c2 = self.MFB1(x2, y2, _c1)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _x2 = self.linear_pred2(_c2)
        c3 = self.MFB2(x3, y3, _c2)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])

        _x3 = self.linear_pred3(_c3)

        c4 = self.MFB3(x4, y4, _c3)

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c1 = F.interpolate(_c1, size=orisize[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=orisize[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=orisize[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return _x1, _x2, _x3, x


class DecoderHead_mlossDeepGuideOnly(nn.Module):
    """Ablation: only use the deepest semantic feature to guide one MFB fusion."""

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead_mlossDeepGuideOnly, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.Aligned_c4 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c3 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c2 = AlignedModule(embedding_dim, embedding_dim // 2)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred3 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # Ablation: use MFB once, guided only by the deepest semantic feature _c1.
        self.MFB_deep = MFB(hidden_dim=c3_in_channels, mid_in_dim=embedding_dim,
                            drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                            d_state=16, mlp_ratio=2.)

    def forward(self, inputs, input2):

        x4, x3, x2, x1 = inputs
        y4, y3, y2, y1 = input2

        c1 = (x1 + y1)
        orisize = x4.shape
        n, _, h, w = x4.shape
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _x1 = self.linear_pred1(_c1)

        # Only this layer receives deep semantic guidance from _c1 through MFB.
        c2 = self.MFB_deep(x2, y2, _c1)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _x2 = self.linear_pred2(_c2)

        # Shallow layers stay as simple modal addition for this ablation.
        c3 = (x3 + y3)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _x3 = self.linear_pred3(_c3)

        c4 = (x4 + y4)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c1 = F.interpolate(_c1, size=orisize[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=orisize[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=orisize[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return _x1, _x2, _x3, x


class DecoderHead_mlossModalDiffOnly(nn.Module):
    """Ablation: MFB only uses modality-difference features as guidance.

    Difference guidance = abs(rgb_feature - sar_feature) at each scale.
    This keeps the original decoder classes untouched and isolates the
    "modal difference only" experiment behind a separate cfg.decoder value.
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead_mlossModalDiffOnly, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.Aligned_c4 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c3 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c2 = AlignedModule(embedding_dim, embedding_dim // 2)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred3 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # Ablation: each MFB receives only abs(modality1 - modality2) as its guidance.
        self.MFB_diff1 = MFB(hidden_dim=c4_in_channels, mid_in_dim=embedding_dim,
                             drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                             d_state=16, mlp_ratio=2.)
        self.MFB_diff2 = MFB(hidden_dim=c3_in_channels, mid_in_dim=embedding_dim,
                             drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                             d_state=16, mlp_ratio=2.)
        self.MFB_diff3 = MFB(hidden_dim=c2_in_channels, mid_in_dim=embedding_dim,
                             drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                             d_state=16, mlp_ratio=2.)
        self.MFB_diff4 = MFB(hidden_dim=c1_in_channels, mid_in_dim=embedding_dim,
                             drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                             d_state=16, mlp_ratio=2.)

    def forward(self, inputs, input2):

        x4, x3, x2, x1 = inputs
        y4, y3, y2, y1 = input2

        orisize = x4.shape
        n, _, h, w = x4.shape

        # Deepest scale: MFB guided by deepest modality-difference feature only.
        diff1 = torch.abs(x1 - y1)
        _diff1 = self.linear_c1(diff1).permute(0, 2, 1).reshape(n, -1, diff1.shape[2], diff1.shape[3])
        c1 = self.MFB_diff1(x1, y1, _diff1)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _x1 = self.linear_pred1(_c1)

        # Middle/deep scale: MFB guided by same-scale modality difference only.
        diff2 = torch.abs(x2 - y2)
        _diff2 = self.linear_c2(diff2).permute(0, 2, 1).reshape(n, -1, diff2.shape[2], diff2.shape[3])
        c2 = self.MFB_diff2(x2, y2, _diff2)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _x2 = self.linear_pred2(_c2)

        # Middle/shallow scale: MFB guided by same-scale modality difference only.
        diff3 = torch.abs(x3 - y3)
        _diff3 = self.linear_c3(diff3).permute(0, 2, 1).reshape(n, -1, diff3.shape[2], diff3.shape[3])
        c3 = self.MFB_diff3(x3, y3, _diff3)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _x3 = self.linear_pred3(_c3)

        # Shallowest scale: MFB guided by same-scale modality difference only.
        diff4 = torch.abs(x4 - y4)
        _diff4 = self.linear_c4(diff4).permute(0, 2, 1).reshape(n, -1, diff4.shape[2], diff4.shape[3])
        c4 = self.MFB_diff4(x4, y4, _diff4)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c1 = F.interpolate(_c1, size=orisize[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=orisize[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=orisize[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return _x1, _x2, _x3, x


class DecoderHead_mlossSpatialGateOnly(nn.Module):
    """Ablation: MFB only uses spatial gating.

    This decoder keeps the normal progressive MFB fusion path, but swaps the
    original MFB gate w = spatial_gate * channel_gate with w = spatial_gate.
    Existing decoder classes are not modified.
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead_mlossSpatialGateOnly, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.Aligned_c4 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c3 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c2 = AlignedModule(embedding_dim, embedding_dim // 2)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred3 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # Ablation: use spatial-gate-only MFB instead of the original spatial*channel gated MFB.
        self.MFB_spatial1 = MFBSpatialGateOnly(hidden_dim=c3_in_channels, mid_in_dim=embedding_dim,
                                               drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                               d_state=16, mlp_ratio=2.)
        self.MFB_spatial2 = MFBSpatialGateOnly(hidden_dim=c2_in_channels, mid_in_dim=embedding_dim,
                                               drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                               d_state=16, mlp_ratio=2.)
        self.MFB_spatial3 = MFBSpatialGateOnly(hidden_dim=c1_in_channels, mid_in_dim=embedding_dim,
                                               drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                               d_state=16, mlp_ratio=2.)

    def forward(self, inputs, input2):

        x4, x3, x2, x1 = inputs
        y4, y3, y2, y1 = input2

        c1 = (x1 + y1)
        orisize = x4.shape
        n, _, h, w = x4.shape
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _x1 = self.linear_pred1(_c1)

        # Spatial-gate-only MFB: x2/y2 guided by deep fused semantic _c1.
        c2 = self.MFB_spatial1(x2, y2, _c1)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _x2 = self.linear_pred2(_c2)

        # Spatial-gate-only MFB: x3/y3 guided by previous fused semantic _c2.
        c3 = self.MFB_spatial2(x3, y3, _c2)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _x3 = self.linear_pred3(_c3)

        # Spatial-gate-only MFB: x4/y4 guided by previous fused semantic _c3.
        c4 = self.MFB_spatial3(x4, y4, _c3)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c1 = F.interpolate(_c1, size=orisize[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=orisize[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=orisize[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return _x1, _x2, _x3, x


class DecoderHead_mlossChannelGateOnly(nn.Module):
    """Ablation: MFB only uses channel gating.

    This decoder keeps the normal progressive MFB fusion path, but swaps the
    original MFB gate w = spatial_gate * channel_gate with w = channel_gate.
    Existing decoder classes are not modified.
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead_mlossChannelGateOnly, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.Aligned_c4 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c3 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c2 = AlignedModule(embedding_dim, embedding_dim // 2)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred3 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # Ablation: use channel-gate-only MFB instead of the original spatial*channel gated MFB.
        self.MFB_channel1 = MFBChannelGateOnly(hidden_dim=c3_in_channels, mid_in_dim=embedding_dim,
                                               drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                               d_state=16, mlp_ratio=2.)
        self.MFB_channel2 = MFBChannelGateOnly(hidden_dim=c2_in_channels, mid_in_dim=embedding_dim,
                                               drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                               d_state=16, mlp_ratio=2.)
        self.MFB_channel3 = MFBChannelGateOnly(hidden_dim=c1_in_channels, mid_in_dim=embedding_dim,
                                               drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                               d_state=16, mlp_ratio=2.)

    def forward(self, inputs, input2):

        x4, x3, x2, x1 = inputs
        y4, y3, y2, y1 = input2

        c1 = (x1 + y1)
        orisize = x4.shape
        n, _, h, w = x4.shape
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _x1 = self.linear_pred1(_c1)

        # Channel-gate-only MFB: x2/y2 guided by deep fused semantic _c1.
        c2 = self.MFB_channel1(x2, y2, _c1)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _x2 = self.linear_pred2(_c2)

        # Channel-gate-only MFB: x3/y3 guided by previous fused semantic _c2.
        c3 = self.MFB_channel2(x3, y3, _c2)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _x3 = self.linear_pred3(_c3)

        # Channel-gate-only MFB: x4/y4 guided by previous fused semantic _c3.
        c4 = self.MFB_channel3(x4, y4, _c3)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c1 = F.interpolate(_c1, size=orisize[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=orisize[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=orisize[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return _x1, _x2, _x3, x


class DecoderHead_mlossGateOnlyNoSSM(nn.Module):
    """Ablation: MFB uses both spatial gating AND channel gating, but SSM is removed.

    This decoder keeps the normal progressive MFB fusion path with full gating
    (w = spatial_gate * channel_gate), but swaps the original MFB (which uses SSM)
    with a version where ssm1/ssm2 are replaced by Identity.

    Purpose: isolate the contribution of the dual-gate mechanism alone,
    without the SSM (selective scan) sequence modelling.
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead_mlossGateOnlyNoSSM, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.Aligned_c4 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c3 = AlignedModule(embedding_dim, embedding_dim // 2)
        self.Aligned_c2 = AlignedModule(embedding_dim, embedding_dim // 2)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred3 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # Ablation: use gate-only (no SSM) MFB instead of the original SSM + gate MFB.
        self.MFB_gatenossm1 = MFBGateOnlyNoSSM(hidden_dim=c3_in_channels, mid_in_dim=embedding_dim,
                                                drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                                d_state=16, mlp_ratio=2.)
        self.MFB_gatenossm2 = MFBGateOnlyNoSSM(hidden_dim=c2_in_channels, mid_in_dim=embedding_dim,
                                                drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                                d_state=16, mlp_ratio=2.)
        self.MFB_gatenossm3 = MFBGateOnlyNoSSM(hidden_dim=c1_in_channels, mid_in_dim=embedding_dim,
                                                drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                                d_state=16, mlp_ratio=2.)

    def forward(self, inputs, input2):

        x4, x3, x2, x1 = inputs
        y4, y3, y2, y1 = input2

        c1 = (x1 + y1)
        orisize = x4.shape
        n, _, h, w = x4.shape
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _x1 = self.linear_pred1(_c1)

        # Gate-only (no SSM) MFB: x2/y2 guided by deep fused semantic _c1.
        c2 = self.MFB_gatenossm1(x2, y2, _c1)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _x2 = self.linear_pred2(_c2)

        # Gate-only (no SSM) MFB: x3/y3 guided by previous fused semantic _c2.
        c3 = self.MFB_gatenossm2(x3, y3, _c2)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _x3 = self.linear_pred3(_c3)

        # Gate-only (no SSM) MFB: x4/y4 guided by previous fused semantic _c3.
        c4 = self.MFB_gatenossm3(x4, y4, _c3)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c1 = F.interpolate(_c1, size=orisize[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=orisize[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=orisize[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return _x1, _x2, _x3, x


# ================================================================
# Ablation #1: Concat + 1x1 Conv  (replace MFB entirely)
# ================================================================
class DecoderHead_mlossConcatFusion(nn.Module):
    """Ablation: replace all MFB modules with simple Concat + 1x1 Conv fusion.

    Instead of DGFM(SS2D + gates + discrepancy), each fusion step does:
        c_i = Conv1x1(Concat(x_i, y_i, mid_guidance))
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        embedding_dim = embed_dim

        self.linear_c4 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred3 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # Ablation: Concat+1x1 Conv instead of MFB
        self.MFB_cf1 = MFBConcatFusion(hidden_dim=c3_in_channels, mid_in_dim=embedding_dim,
                                        drop_path=0., attn_drop_rate=0, d_state=16, mlp_ratio=2.)
        self.MFB_cf2 = MFBConcatFusion(hidden_dim=c2_in_channels, mid_in_dim=embedding_dim,
                                        drop_path=0., attn_drop_rate=0, d_state=16, mlp_ratio=2.)
        self.MFB_cf3 = MFBConcatFusion(hidden_dim=c1_in_channels, mid_in_dim=embedding_dim,
                                        drop_path=0., attn_drop_rate=0, d_state=16, mlp_ratio=2.)

    def forward(self, inputs, input2):
        x4, x3, x2, x1 = inputs
        y4, y3, y2, y1 = input2

        c1 = (x1 + y1)
        orisize = x4.shape
        n, _, h, w = x4.shape
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _x1 = self.linear_pred1(_c1)

        c2 = self.MFB_cf1(x2, y2, _c1)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _x2 = self.linear_pred2(_c2)

        c3 = self.MFB_cf2(x3, y3, _c2)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _x3 = self.linear_pred3(_c3)

        c4 = self.MFB_cf3(x4, y4, _c3)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c1 = F.interpolate(_c1, size=orisize[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=orisize[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=orisize[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return _x1, _x2, _x3, x


# ================================================================
# Ablation #7: SS2D without discrepancy-aware gates
#   Keep SS2D + spatial/channel gates, remove discrepancy from gate input.
# ================================================================
class DecoderHead_mlossNoDisagree(nn.Module):
    """Ablation: DGFM with SS2D + dual gates, but WITHOUT discrepancy feature.

    Original DGFM gate input = concat(mid_fused, |rgb - ir|).
    This ablation: gate input = mid_fused only (use_disagree=False).
    """

    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        embedding_dim = embed_dim

        self.linear_c4 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.linear_pred1 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred3 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # Ablation: DGFM with use_disagree=False
        self.MFB_nd1 = MFBNoDisagree(hidden_dim=c3_in_channels, mid_in_dim=embedding_dim,
                                      drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                      d_state=16, mlp_ratio=2.)
        self.MFB_nd2 = MFBNoDisagree(hidden_dim=c2_in_channels, mid_in_dim=embedding_dim,
                                      drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                      d_state=16, mlp_ratio=2.)
        self.MFB_nd3 = MFBNoDisagree(hidden_dim=c1_in_channels, mid_in_dim=embedding_dim,
                                      drop_path=0., norm_layer=nn.LayerNorm, attn_drop_rate=0,
                                      d_state=16, mlp_ratio=2.)

    def forward(self, inputs, input2):
        x4, x3, x2, x1 = inputs
        y4, y3, y2, y1 = input2

        c1 = (x1 + y1)
        orisize = x4.shape
        n, _, h, w = x4.shape
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _x1 = self.linear_pred1(_c1)

        c2 = self.MFB_nd1(x2, y2, _c1)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _x2 = self.linear_pred2(_c2)

        c3 = self.MFB_nd2(x3, y3, _c2)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _x3 = self.linear_pred3(_c3)

        c4 = self.MFB_nd3(x4, y4, _c3)
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c1 = F.interpolate(_c1, size=orisize[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=orisize[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=orisize[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return _x1, _x2, _x3, x

