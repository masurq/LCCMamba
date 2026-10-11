from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
import torch.utils.checkpoint as checkpoint
import math
from einops import repeat
from .vmamba import Mlp, gMlp, Temporal_SSM, SS2D
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except Exception as _mamba_import_error:
    # Inference fallback for hosts without the optional mamba_ssm CUDA extension.
    from .csms6s import selective_scan_fn as _release_selective_scan

    def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                          delta_softplus=True, return_last_state=False, **kwargs):
        if z is not None or return_last_state:
            raise NotImplementedError("The release fallback supports inference without z/last-state output")
        return _release_selective_scan(
            u, delta, A, B, C, D, delta_bias,
            delta_softplus=delta_softplus, backend="torch"
        )

    selective_scan_ref = selective_scan_fn
  
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)    # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x = torch.cat([avg_out, max_out], dim=1)    # [B, 2, H, W]
        x = self.conv1(x)
        return self.sigmoid(x)


class TemporalAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(TemporalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


# ====================== Mamba Fusion Block ==========================
class DDIM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        hidden_space_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        # =============================
        mlp_ratio=4.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        # =============================
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_space_dim = hidden_space_dim
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm


        self.norm_i = norm_layer(hidden_dim)
        self.norm_e = norm_layer(hidden_dim)

        # self.mixatt = MixAttention(hidden_dim, reduction=4)

        # ========================= CTIM ===========================
        self.cca = CrossTemporalAttention(hidden_dim)

        self.ssm_i = Temporal_SSM(
            # basic dims ===========
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # dwconv ===============
            d_conv=ssm_conv, # < 2 means no conv 
            conv_bias=ssm_conv_bias,
            # ======================
            dropout=ssm_drop_rate,
            # dt init ==============
            initialize=ssm_init,
            # ======================
            forward_type="vc_noz",
            channel_first=channel_first,
            # ======================
            **kwargs,
        )
        self.ssm_e = Temporal_SSM(
            # basic dims ===========
            d_model=hidden_dim,
            d_ssm=hidden_space_dim,
            d_state=ssm_d_state,
            ssm_ratio=2.0,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # dwconv ===============
            d_conv=ssm_conv, # < 2 means no conv 
            conv_bias=ssm_conv_bias,
            # ======================
            dropout=ssm_drop_rate,
            # dt init ==============
            initialize=ssm_init,
            # ======================
            forward_type="vc_noz",
            channel_first=channel_first,
            # ======================
            **kwargs,
        )

        self.conv_i1 = nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 1)
        self.conv_e1 = nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 1)

        self.ca1 = TemporalAttention(hidden_dim)
        self.ca2 = TemporalAttention(hidden_dim)

        # ========================= CTIM ===========================


        # ========================= CSIM ===========================
        self.csa = CrossSpatialAttention()

        self.ss2d_i = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # ==========================
            initialize=ssm_init,
            # ==========================
            forward_type="v05_noz",
            channel_first=channel_first
        )
        self.ss2d_e = SS2D(
            d_model=hidden_dim, 
            d_state=ssm_d_state, 
            ssm_ratio=ssm_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # ==========================
            initialize=ssm_init,
            # ==========================
            forward_type="v05_noz",
            channel_first=channel_first
        )

        self.conv_i2 = nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 1)
        self.conv_e2 = nn.Conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = 1)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        # ========================= CSIM ===========================

        # 1x1 conv 用来调整通道数
        # 输入: concat(img_c, img_s)(B, 2C, H, W), concat(ev_c, ev_s)(B, 2C, H, W)
        # 输出: img(B, C, H, W), ev(B, C, H, W)
        self.conv_i3 = nn.Conv2d(in_channels = hidden_dim * 2, out_channels = hidden_dim, kernel_size = 1)
        self.conv_e3 = nn.Conv2d(in_channels = hidden_dim * 2, out_channels = hidden_dim, kernel_size = 1)

        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def _forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        """
        Args:
            img: [batch_size, hidden_dim, h, w]
            ev: [batch_size, hidden_dim, h, w]
        Process:
            1. Swap the channels between img and ev
            2.1.1 Apply SSM to the swapped channels
            2.1.2 Apply Channel Attention to the swapped channels
            2.2.1 Apply SS2D to the swapped channels
            2.2.2 Apply Spatial Attention to the swapped channels
            3. Concat the results from SSM and SS2D
            4. Apply 1x1 conv to the concatenated results
            5. Apply DropPath to the results
        """

        # Normalize the input tensors
        img_norm = self.norm_i(x_img)
        ev_norm = self.norm_e(x_ev)

        # Cross Temporal Attention
        img_cca, ev_cca = self.cca(img_norm, ev_norm)
        # Cross Spatial Attention
        img_csa, ev_csa = self.csa(img_norm, ev_norm)

        # Cross Temporal SSM
        x_img_c = self.ssm_i(img_cca) + self.conv_i1(img_cca)
        x_ev_c = self.ssm_e(ev_cca) + self.conv_e1(ev_cca)
        
        # Apply Temporal Attention
        x_img_c = x_img_c * self.ca1(x_img_c)   # [12,96,50,87] ca1(x_img_c) [12,96,1,1]
        x_ev_c = x_ev_c * self.ca2(x_ev_c)


        # Cross SS2D
        x_img_s = self.ss2d_i(img_csa) + self.conv_i2(img_csa)
        x_ev_s = self.ss2d_e(ev_csa) + self.conv_e2(ev_csa)

        # Apply Spatial Attention
        x_img_s = x_img_s * self.sa1(x_img_s)   # [12,96,50,87] sa1(x_img_s) [12,1,51,88]
        x_ev_s = x_ev_s * self.sa2(x_ev_s)

        # concat
        x_img_ = torch.cat((x_img_c, x_img_s), dim=1)
        x_ev_ = torch.cat((x_ev_c, x_ev_s), dim=1)
        # Apply 1x1 conv
        x_img_swap = self.conv_i3(x_img_)
        x_ev_swap = self.conv_e3(x_ev_)
        # Apply DropPath
        x_img = self.drop_path1(x_img_swap) + x_img
        x_ev = self.drop_path2(x_ev_swap) + x_ev
        
        return x_img, x_ev


    def forward(self, x_img: torch.Tensor, x_ev: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x_img, x_ev)
        else:
            return self._forward(x_img, x_ev)
        

        
class CrossTemporalAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x_img, x_ev):

        x_img, x_ev = self.SwapChannel(x_img, x_ev)

        attn_img = self.shared_fc(x_ev)
        attn_ev = self.shared_fc(x_img)

        x_img_new = x_img * attn_img
        x_ev_new = x_ev * attn_ev
        return x_img_new, x_ev_new
    
    @staticmethod
    def SwapChannel(x_img: torch.Tensor, x_ev: torch.Tensor):
        """
        Swap the channels between img and ev
        Args:
            x_img: [b, c, h, w]
            x_ev: [b, c, h, w]
        Returns:
            swapped_x_img: [b, c, h, w]
            swapped_x_ev: [b, c, h, w]
        """
        # Ensure that both tensors have the same shape
        assert x_img.shape == x_ev.shape, "Both input tensors must have the same shape"
        
        # Get channel indices for odd and even channels
        C = x_img.shape[1]
        odd_indices = torch.arange(1, C, 2).to(x_img.device)
        # even_indices = torch.arange(0, C, 2).to_img(x_img.device)
        
        # Swap odd channels
        swapped_x_img = x_img.clone()
        swapped_x_ev = x_ev.clone()
        
        # For image modality: keep even channels, take event's odd channels
        swapped_x_img[:, odd_indices, :, :] = x_ev[:, odd_indices, :, :]
        
        # For event modality: keep even channels, take image's odd channels
        swapped_x_ev[:, odd_indices, :, :] = x_img[:, odd_indices, :, :]
        
        return swapped_x_img, swapped_x_ev
      
class CrossSpatialAttention_(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
                    nn.Conv2d(6, 6*reduction, kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(6*reduction, 3, kernel_size),
                    nn.Sigmoid())

    def forward(self, x_img, x_ev):
        B, _, H, W = x_img.shape

        avg_img = torch.mean(x_img, dim=1, keepdim=True)
        max_img, _ = torch.max(x_img, dim=1, keepdim=True)
        avg_ev = torch.mean(x_ev, dim=1, keepdim=True)
        max_ev, _ = torch.max(x_ev, dim=1, keepdim=True)


        F = x_img + x_ev
        avg_F = torch.mean(F, dim=1, keepdim=True)
        max_F, _ = torch.max(F, dim=1, keepdim=True)


        x_cat = torch.cat((avg_img, max_img, avg_ev, max_ev, avg_F, max_F), dim=1) # B 6 H W

        att = self.mlp(x_cat).view(B, 3, 1, H, W)
        # att_img = att[:, 0, :, :, :]
        # att_ev = att[:, 1, :, :, :]
        Att_E = att[:, 0]
        Att_I = att[:, 1]
        Att_F = att[:, 2]
        # x_img_new = x_img * att_ev
        # x_ev_new = x_ev * att_img


        x_img_new = x_img * Att_E * Att_F
        x_ev_new = x_ev * Att_I * Att_F
        return x_img_new, x_ev_new

class CrossSpatialAttention(nn.Module):
    def __init__(self, kernel_size=1, reduction=4):
        super().__init__()
        self.mlp = nn.Sequential(
                    nn.Conv2d(4, 4*reduction, kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(4*reduction, 2, kernel_size),
                    nn.Sigmoid())

    def forward(self, x_img, x_ev):
        B, _, H, W = x_img.shape

        avg_img = torch.mean(x_img, dim=1, keepdim=True)
        max_img, _ = torch.max(x_img, dim=1, keepdim=True)
        avg_ev = torch.mean(x_ev, dim=1, keepdim=True)
        max_ev, _ = torch.max(x_ev, dim=1, keepdim=True)

        x_cat = torch.cat((avg_img, max_img, avg_ev, max_ev), dim=1) # B 4 H W

        att = self.mlp(x_cat).reshape(B, 2, 1, H, W)
        att_img = att[:, 0, :, :, :]
        att_ev = att[:, 1, :, :, :]
        x_img_new = x_img * att_ev
        x_ev_new = x_ev * att_img
        return x_img_new, x_ev_new


class single_SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        x = self.in_proj(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        return y


#
#
# class MFB(nn.Module):
#     def __init__(
#             self,
#             hidden_dim: int = 0,
#             drop_path: float = 0,
#             attn_drop_rate: float = 0,
#             d_state: int = 16,
#             mlp_ratio: float = 2.,
#             **kwargs,
#     ):
#         super().__init__()
#         self.skip_scale_rgb = nn.Parameter(torch.ones(hidden_dim))
#         self.skip_scale_ir = nn.Parameter(torch.ones(hidden_dim))
#         self.ln_rgb = nn.LayerNorm(hidden_dim)
#         self.ln_ir = nn.LayerNorm(hidden_dim)
#         self.ln_mid = nn.LayerNorm(hidden_dim)
#         self.ssm1 = single_SS2D_MFB(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate,
#                                     **kwargs)
#         self.ssm2 = single_SS2D_MFB(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate,
#                                     **kwargs)
#         self.in_proj_mid = nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio))
#         self.conv2d = nn.Conv2d(
#             in_channels=int(hidden_dim * mlp_ratio),
#             out_channels=int(hidden_dim * mlp_ratio),
#             groups=int(hidden_dim * mlp_ratio),
#             bias=True,
#             kernel_size=3,
#             padding=(3 - 1) // 2,
#         )
#         self.sigmoid = nn.Sigmoid()
#         self.out_proj = nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)
#         self.drop_path = DropPath(drop_path)
#
#         self.c3 = nn.Sequential(
#             nn.Conv2d(512, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),  # 修正：ReLU不应有参数
#         )
#     def forward(self, rgb, ir, mid):
#         orisize = rgb.shape
#         mid = F.interpolate(mid, size=orisize[2:], mode='bilinear', align_corners=False)
#         mid = self.c3(mid)
#         #print("mid",mid.shape)
#         # print(self.skip_scale_rgb.shape)
#         rgb = rgb.permute(0, 2, 3, 1).contiguous()
#         ir = ir.permute(0, 2, 3, 1).contiguous()
#         mid = mid.permute(0, 2, 3, 1).contiguous()
#         out1 = rgb
#         out2 = ir
#         rgb = self.ln_rgb(rgb)
#         ir = self.ln_ir(ir)
#         mid = self.ln_mid(mid)
#         mid = rgb + ir + mid
#         mid = self.in_proj_mid(mid)
#         mid = mid.permute(0, 3, 1, 2).contiguous()
#         mid = self.conv2d(mid)
#         mid = mid.permute(0, 2, 3, 1).contiguous()
#         w = self.sigmoid(mid)
#         rgb = self.ssm1(rgb)
#         ir = self.ssm2(ir)
#         out3 = rgb * w + ir * (1 - w)
#         result = self.out_proj(out3) + out1 * self.skip_scale_rgb + out2 * self.skip_scale_ir
#         result = result.permute(0, 3, 1, 2).contiguous()
#         return result



class DGFM(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mid_in_dim: int = 512,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        drop_path: float = 0.0,
        attn_drop_rate: float = 0.0,
        gate_tau: float = 2.0,
        use_disagree: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_ratio = mlp_ratio
        self.d_inner = int(hidden_dim * mlp_ratio)
        self.gate_tau = gate_tau
        self.use_disagree = use_disagree

        self.mid_align = nn.Sequential(
            nn.Conv2d(mid_in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.ln_rgb = nn.LayerNorm(hidden_dim)
        self.ln_ir  = nn.LayerNorm(hidden_dim)
        self.ln_mid = nn.LayerNorm(hidden_dim)
        self.ssm1 = single_SS2D(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio,
                                    dropout=attn_drop_rate, **kwargs)
        self.ssm2 = single_SS2D(d_model=hidden_dim, d_state=d_state, expand=mlp_ratio,
                                    dropout=attn_drop_rate, **kwargs)
        gate_in_ch = hidden_dim
        if self.use_disagree:
            gate_in_ch = hidden_dim * 2

        self.gate_up = nn.Sequential(
            nn.Conv2d(gate_in_ch, self.d_inner, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner, bias=True),
            nn.SiLU(),
            nn.Conv2d(self.d_inner, 1, kernel_size=1, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(self.d_inner, hidden_dim)
        self.calib = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=True),
        )
        self.skip_scale_rgb = nn.Parameter(torch.ones(1, 1, 1, hidden_dim))
        self.skip_scale_ir  = nn.Parameter(torch.ones(1, 1, 1, hidden_dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    def forward(self, rgb, ir, mid):
        B, C, H, W = rgb.shape

        if mid.shape[2:] != (H, W):
            mid = F.interpolate(mid, size=(H, W), mode="bilinear", align_corners=False)
        mid = self.mid_align(mid)
        rgb_n = rgb.permute(0, 2, 3, 1).contiguous()
        ir_n  = ir.permute(0, 2, 3, 1).contiguous()
        mid_n = mid.permute(0, 2, 3, 1).contiguous()
        res_rgb, res_ir = rgb_n, ir_n
        rgb_ln = self.ln_rgb(rgb_n)
        ir_ln  = self.ln_ir(ir_n)
        mid_ln = self.ln_mid(mid_n)
        mid_fused = rgb_ln + ir_ln + mid_ln
        if self.use_disagree:
            disagree = torch.abs(rgb_ln - ir_ln)
            gate_feat = torch.cat([mid_fused, disagree], dim=-1)
        else:
            gate_feat = mid_fused  # (B,H,W,C)
        gate_feat = gate_feat.permute(0, 3, 1, 2).contiguous()
        gate_feat = self.gate_up(gate_feat)

        w_s = self.spatial_gate(gate_feat)
        w_s = torch.sigmoid(w_s / max(self.gate_tau, 1e-6))
        w_c = self.channel_gate(gate_feat)
        w   = w_s * w_c
        w_n = w.permute(0, 2, 3, 1).contiguous()
        rgb_out = self.ssm1(rgb_ln)
        ir_out  = self.ssm2(ir_ln)
        fused_n = rgb_out * w_n + ir_out * (1.0 - w_n)
        fused_n = self.out_proj(fused_n) + res_rgb * self.skip_scale_rgb + res_ir * self.skip_scale_ir
        out = fused_n.permute(0, 3, 1, 2).contiguous()

        out = out + self.drop_path(self.calib(out))
        return out


class DGFMSpatialGateOnly(DGFM):
    """Ablation MFB: keep only the spatial gate and remove the channel gate.

    Original DGFM uses:
        w = spatial_gate(gate_feat) * channel_gate(gate_feat)

    This ablation uses:
        w = spatial_gate(gate_feat)

    The original DGFM class is left untouched.
    """

    def __init__(self, *args, **kwargs):
        super(DGFMSpatialGateOnly, self).__init__(*args, **kwargs)
        # Remove channel-gate parameters from this ablation module.
        self.channel_gate = nn.Identity()

    def forward(self, rgb, ir, mid):
        B, C, H, W = rgb.shape

        if mid.shape[2:] != (H, W):
            mid = F.interpolate(mid, size=(H, W), mode="bilinear", align_corners=False)
        mid = self.mid_align(mid)
        rgb_n = rgb.permute(0, 2, 3, 1).contiguous()
        ir_n = ir.permute(0, 2, 3, 1).contiguous()
        mid_n = mid.permute(0, 2, 3, 1).contiguous()
        res_rgb, res_ir = rgb_n, ir_n

        rgb_ln = self.ln_rgb(rgb_n)
        ir_ln = self.ln_ir(ir_n)
        mid_ln = self.ln_mid(mid_n)

        mid_fused = rgb_ln + ir_ln + mid_ln
        if self.use_disagree:
            disagree = torch.abs(rgb_ln - ir_ln)
            gate_feat = torch.cat([mid_fused, disagree], dim=-1)
        else:
            gate_feat = mid_fused

        gate_feat = gate_feat.permute(0, 3, 1, 2).contiguous()
        gate_feat = self.gate_up(gate_feat)

        # Ablation: spatial gate only. No channel gate is used here.
        w = self.spatial_gate(gate_feat)
        w = torch.sigmoid(w / max(self.gate_tau, 1e-6))
        w_n = w.permute(0, 2, 3, 1).contiguous()

        rgb_out = self.ssm1(rgb_ln)
        ir_out = self.ssm2(ir_ln)
        fused_n = rgb_out * w_n + ir_out * (1.0 - w_n)
        fused_n = self.out_proj(fused_n) + res_rgb * self.skip_scale_rgb + res_ir * self.skip_scale_ir
        out = fused_n.permute(0, 3, 1, 2).contiguous()

        out = out + self.drop_path(self.calib(out))
        return out


class DGFMChannelGateOnly(DGFM):
    """Ablation MFB: keep only the channel gate and remove the spatial gate.

    Original DGFM uses:
        w = spatial_gate(gate_feat) * channel_gate(gate_feat)

    This ablation uses:
        w = channel_gate(gate_feat)

    The original DGFM class is left untouched.
    """

    def __init__(self, *args, **kwargs):
        super(DGFMChannelGateOnly, self).__init__(*args, **kwargs)
        # Remove spatial-gate parameters from this ablation module.
        self.spatial_gate = nn.Identity()

    def forward(self, rgb, ir, mid):
        B, C, H, W = rgb.shape

        if mid.shape[2:] != (H, W):
            mid = F.interpolate(mid, size=(H, W), mode="bilinear", align_corners=False)
        mid = self.mid_align(mid)
        rgb_n = rgb.permute(0, 2, 3, 1).contiguous()
        ir_n = ir.permute(0, 2, 3, 1).contiguous()
        mid_n = mid.permute(0, 2, 3, 1).contiguous()
        res_rgb, res_ir = rgb_n, ir_n

        rgb_ln = self.ln_rgb(rgb_n)
        ir_ln = self.ln_ir(ir_n)
        mid_ln = self.ln_mid(mid_n)

        mid_fused = rgb_ln + ir_ln + mid_ln
        if self.use_disagree:
            disagree = torch.abs(rgb_ln - ir_ln)
            gate_feat = torch.cat([mid_fused, disagree], dim=-1)
        else:
            gate_feat = mid_fused

        gate_feat = gate_feat.permute(0, 3, 1, 2).contiguous()
        gate_feat = self.gate_up(gate_feat)

        # Ablation: channel gate only. No spatial gate is used here.
        w = self.channel_gate(gate_feat)
        w_n = w.permute(0, 2, 3, 1).contiguous()

        rgb_out = self.ssm1(rgb_ln)
        ir_out = self.ssm2(ir_ln)
        fused_n = rgb_out * w_n + ir_out * (1.0 - w_n)
        fused_n = self.out_proj(fused_n) + res_rgb * self.skip_scale_rgb + res_ir * self.skip_scale_ir
        out = fused_n.permute(0, 3, 1, 2).contiguous()

        out = out + self.drop_path(self.calib(out))
        return out


class DGFMGateOnlyNoSSM(DGFM):
    """Ablation MFB: keep both spatial gate and channel gate, but remove SSM.

    Original DGFM uses:
        rgb_out = self.ssm1(rgb_ln)     # SS2D (SSM): in_proj + conv2d + cross_scan + out_norm
        ir_out  = self.ssm2(ir_ln)      # SS2D (SSM): in_proj + conv2d + cross_scan + out_norm
        fused_n = rgb_out * w + ir_out * (1 - w)

    This ablation replaces SSM with a simple Linear projection (hidden_dim → d_inner):
        rgb_out = self.ssm1(rgb_ln)     # Linear only, no cross-scan
        ir_out  = self.ssm2(ir_ln)      # Linear only, no cross-scan
        fused_n = rgb_out * w + ir_out * (1 - w)

    The gating mechanism w = spatial_gate * channel_gate is fully preserved.
    The original DGFM class is left untouched.
    """

    def __init__(self, *args, **kwargs):
        super(DGFMGateOnlyNoSSM, self).__init__(*args, **kwargs)
        # Replace SSM modules with a simple Linear projection.
        # SSM internally does in_proj (hidden_dim → d_inner) + scan + out_norm.
        # Here we only keep the channel expansion, no sequence modelling.
        self.ssm1 = nn.Linear(self.hidden_dim, self.d_inner)
        self.ssm2 = nn.Linear(self.hidden_dim, self.d_inner)

    def forward(self, rgb, ir, mid):
        B, C, H, W = rgb.shape

        if mid.shape[2:] != (H, W):
            mid = F.interpolate(mid, size=(H, W), mode="bilinear", align_corners=False)
        mid = self.mid_align(mid)
        rgb_n = rgb.permute(0, 2, 3, 1).contiguous()
        ir_n = ir.permute(0, 2, 3, 1).contiguous()
        mid_n = mid.permute(0, 2, 3, 1).contiguous()
        res_rgb, res_ir = rgb_n, ir_n

        rgb_ln = self.ln_rgb(rgb_n)
        ir_ln = self.ln_ir(ir_n)
        mid_ln = self.ln_mid(mid_n)

        mid_fused = rgb_ln + ir_ln + mid_ln
        if self.use_disagree:
            disagree = torch.abs(rgb_ln - ir_ln)
            gate_feat = torch.cat([mid_fused, disagree], dim=-1)
        else:
            gate_feat = mid_fused

        gate_feat = gate_feat.permute(0, 3, 1, 2).contiguous()
        gate_feat = self.gate_up(gate_feat)

        # Both spatial gate and channel gate are used.
        w_s = self.spatial_gate(gate_feat)
        w_s = torch.sigmoid(w_s / max(self.gate_tau, 1e-6))
        w_c = self.channel_gate(gate_feat)
        w = w_s * w_c
        w_n = w.permute(0, 2, 3, 1).contiguous()

        # Ablation: NO SSM — simple Linear instead of SS2D cross-scan.
        rgb_out = self.ssm1(rgb_ln)   # Linear(hidden_dim → d_inner)
        ir_out = self.ssm2(ir_ln)     # Linear(hidden_dim → d_inner)
        fused_n = rgb_out * w_n + ir_out * (1.0 - w_n)
        fused_n = self.out_proj(fused_n) + res_rgb * self.skip_scale_rgb + res_ir * self.skip_scale_ir
        out = fused_n.permute(0, 3, 1, 2).contiguous()

        out = out + self.drop_path(self.calib(out))
        return out


class DGFMSSMNoGate(DGFM):
    """Ablation MFB: keep SSM processing, remove discrepancy-aware gating.

    Original DGFM uses:
        w = spatial_gate(gate_feat) * channel_gate(gate_feat)
        fused_n = ssm1(rgb) * w + ssm2(ir) * (1 - w)

    This ablation removes the gating mechanism completely and uses
    simple addition of the two SSM outputs (effectively w = 0.5):
        fused_n = (ssm1(rgb) + ssm2(ir)) / 2

    Purpose: isolate the contribution of SSM processing alone,
    without the discrepancy-aware gating mechanism.
    """

    def __init__(self, *args, **kwargs):
        super(DGFMSSMNoGate, self).__init__(*args, **kwargs)
        # Remove gating-related parameters for this ablation.
        self.gate_up = nn.Identity()
        self.spatial_gate = nn.Identity()
        self.channel_gate = nn.Identity()

    def forward(self, rgb, ir, mid):
        B, C, H, W = rgb.shape

        if mid.shape[2:] != (H, W):
            mid = F.interpolate(mid, size=(H, W), mode="bilinear", align_corners=False)
        mid = self.mid_align(mid)
        rgb_n = rgb.permute(0, 2, 3, 1).contiguous()
        ir_n = ir.permute(0, 2, 3, 1).contiguous()
        mid_n = mid.permute(0, 2, 3, 1).contiguous()
        res_rgb, res_ir = rgb_n, ir_n

        rgb_ln = self.ln_rgb(rgb_n)
        ir_ln = self.ln_ir(ir_n)
        mid_ln = self.ln_mid(mid_n)

        # Ablation: SSM processing only, no gating. Simple equal-weight fusion.
        rgb_out = self.ssm1(rgb_ln)
        ir_out = self.ssm2(ir_ln)
        fused_n = (rgb_out + ir_out) / 2.0
        fused_n = self.out_proj(fused_n) + res_rgb * self.skip_scale_rgb + res_ir * self.skip_scale_ir
        out = fused_n.permute(0, 3, 1, 2).contiguous()

        out = out + self.drop_path(self.calib(out))
        return out


# ================================================================
# Ablation #1: Concat + 1x1 Conv  (replace entire MFB)
# ================================================================
class DGFMConcatFusion(nn.Module):
    """Ablation: replace DGFM with simple Concat + 1x1 Conv fusion.

    Original DGFM:
        mid_align(mid) → ln → ssm1/ssm2 → spatial_gate * channel_gate → out_proj → calib

    This ablation:
        Concat(rgb, ir, mid) → 1x1 Conv → output
    No SSM, no gating, no discrepancy.
    """

    def __init__(
        self,
        hidden_dim: int,
        mid_in_dim: int = 512,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        drop_path: float = 0.0,
        attn_drop_rate: float = 0.0,
        gate_tau: float = 2.0,
        use_disagree: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.d_inner = int(hidden_dim * mlp_ratio)

        self.mid_align = nn.Sequential(
            nn.Conv2d(mid_in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Concat: rgb(C) + ir(C) + mid(C) = 3*C → 1x1 Conv → C
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, rgb, ir, mid):
        B, C, H, W = rgb.shape

        if mid.shape[2:] != (H, W):
            mid = F.interpolate(mid, size=(H, W), mode="bilinear", align_corners=False)
        mid = self.mid_align(mid)

        out = self.fuse_conv(torch.cat([rgb, ir, mid], dim=1))
        out = out + self.drop_path(out)
        return out


# ================================================================
# Ablation #7: SS2D without discrepancy-aware gates
#   Keep SS2D + spatial/channel gates, remove discrepancy feature
#   from gate input (use_disagree=False).
# ================================================================
class DGFMNoDisagree(DGFM):
    """Ablation MFB: SS2D retained, gating retained, but discrepancy removed.

    Original DGFM:
        gate_feat = concat(mid_fused, |rgb - ir|)  ← discrepancy-aware
        w = spatial_gate(gate_feat) * channel_gate(gate_feat)
        fused = ssm1(rgb) * w + ssm2(ir) * (1-w)

    This ablation:
        gate_feat = mid_fused                         ← no discrepancy
        w = spatial_gate(gate_feat) * channel_gate(gate_feat)
        fused = ssm1(rgb) * w + ssm2(ir) * (1-w)

    The SS2D (ssm1/ssm2) is fully preserved.
    """

    def __init__(self, *args, **kwargs):
        kwargs['use_disagree'] = False
        super(DGFMNoDisagree, self).__init__(*args, **kwargs)


import torch
import torch.nn as nn

class FourierEdgeEnhancementModule(nn.Module):
    """
    2D 版本傅里叶边缘增强模块 (FEM)
    适用于处理 2D 图像特征图 [B, C, H, W]。
    """
    def __init__(self, ratio_r=0.2):
        super(FourierEdgeEnhancementModule, self).__init__()
        self.ratio_r = ratio_r

    def forward(self, x):
        """
        前向传播。
        Args:
            x (torch.Tensor): 输入特征图，形状为 (Batch, Channels, Height, Width) 即 [B, C, H, W]。
        Returns:
            torch.Tensor: 增强后的特征图 f_i^{aug}，形状与输入 x 相同。
        """
        original_feature = x

        # 1. 对输入特征图执行2D快速傅里叶变换 (FFT)
        # fft2 在最后两个维度 (H, W) 上进行变换。
        x_fft = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        # x_fft 是复数张量，形状为 [B, C, H, W]

        # 2. 获取振幅谱和相位谱
        amplitude = torch.abs(x_fft)  # 振幅谱
        phase = torch.angle(x_fft)    # 相位谱

        # 3. 创建2D二值掩码 M_i 以分离高低频
        _, _, H, W = x.shape
        device = x.device

        # 计算中心区域的边界
        h_start = int(H // 2 * (1 - self.ratio_r))
        h_end = int(H // 2 * (1 + self.ratio_r)) + (H % 2)
        w_start = int(W // 2 * (1 - self.ratio_r))
        w_end = int(W // 2 * (1 + self.ratio_r)) + (W % 2)

        # 初始化全零掩码（1代表保留低频，0代表保留高频）
        mask = torch.zeros((H, W), device=device)
        # 将中心区域设为1
        mask[h_start:h_end, w_start:w_end] = 1

        # 4. 分离高频和低频振幅谱
        amplitude_low = amplitude * mask
        amplitude_high = amplitude * (1 - mask)

        # 5. 将高频振幅谱与原始相位谱结合，并执行逆FFT
        high_freq_complex = torch.polar(amplitude_high, phase)
        edge_enhanced_feature = torch.fft.ifft2(high_freq_complex, dim=(-2, -1), norm='ortho').real

        # 6. 将边缘增强特征与原始特征相加，得到最终输出
        augmented_feature = edge_enhanced_feature + original_feature

        return augmented_feature

import torch
import torch.nn as nn
import torch.nn.functional as F

class HaarDWT2D(nn.Module):
    """
    Haar 2D DWT (one level), depthwise fixed conv, stride=2.
    Input : [B, C, H, W]  (H,W 最好为偶数；若不是会自动 pad 到偶数)
    Output: LL,LH,HL,HH each [B, C, H/2, W/2]
    """
    def __init__(self):
        super().__init__()
        # 1D Haar filters (normalized)
        h = torch.tensor([1.0, 1.0]) / 2**0.5  # low-pass
        g = torch.tensor([1.0, -1.0]) / 2**0.5 # high-pass

        # 2D outer products -> 4 subbands
        LL = torch.ger(h, h)
        LH = torch.ger(h, g)
        HL = torch.ger(g, h)
        HH = torch.ger(g, g)

        # register as buffers
        self.register_buffer("kLL", LL[None, None, :, :])  # [1,1,2,2]
        self.register_buffer("kLH", LH[None, None, :, :])
        self.register_buffer("kHL", HL[None, None, :, :])
        self.register_buffer("kHH", HH[None, None, :, :])

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        # pad to even
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            H, W = x.shape[-2], x.shape[-1]

        # depthwise conv: groups=C, weight shape [C,1,2,2]
        def dwconv2(x, k):
            w = k.repeat(C, 1, 1, 1)  # [C,1,2,2]
            return F.conv2d(x, w, stride=2, padding=0, groups=C)

        LL = dwconv2(x, self.kLL)
        LH = dwconv2(x, self.kLH)
        HL = dwconv2(x, self.kHL)
        HH = dwconv2(x, self.kHH)
        return LL, LH, HL, HH


class EGM_FDEG(nn.Module):
    """
    Edge Guidance Module (EGM) from FDEG-Net (paper-aligned implementation).

    Input : F [B, C, H, W]
    Output: Fe_hat [B, C, H/2, W/2]  (增强后的边缘特征图)
            Also returns Coee (阈值前的边缘纹理图) if needed.
    """
    def __init__(self, eps: float = 1e-6, use_abs: bool = True):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.eps = eps
        self.use_abs = use_abs  # 实践里常用 abs 更稳定

    @torch.no_grad()
    def _adaptive_threshold_from_HH(self, HH: torch.Tensor):
        """
        按 Algorithm 1 用 HH 估计每个通道的阈值 Tc
        HH: [B, C, h, w]
        Return Tc: [B, C, 1, 1]
        """
        B, C, h, w = HH.shape
        flat = HH.reshape(B, C, -1)

        # Algorithm: median & max of HH coefficients
        hmed = flat.median(dim=-1).values  # [B,C]
        hmax = flat.max(dim=-1).values     # [B,C]

        # noise std: sigma_eps = hmed / 0.6745
        sigma_eps = hmed / 0.6745
        var_eps = sigma_eps ** 2

        # variance of F estimated by mean(h^2) over HH (paper uses sum(h^2)/N)
        var_F = (flat ** 2).mean(dim=-1)   # [B,C]

        # var_f = max(var_F - var_eps, 0)
        var_f = (var_F - var_eps).clamp_min(0.0)

        # if var_eps >= var_F -> Tc = hmax else Tc = var_eps / (2*var_f)
        Tc = torch.where(var_eps >= var_F, hmax, var_eps / (2.0 * (var_f + 1e-12)))
        return Tc.view(B, C, 1, 1)

    def forward(self, F_in: torch.Tensor):
        """
        F_in: [B,C,H,W]
        returns:
          Fe_hat: [B,C,H/2,W/2]
          Coee  : [B,C,H/2,W/2]
        """
        LL, LH, HL, HH = self.dwt(F_in)

        # Coee = (LH + HL) / LL   (eq.10)
        if self.use_abs:
            num = (LH.abs() + HL.abs())
            den = LL.abs() + self.eps
        else:
            num = (LH + HL)
            den = LL + self.eps
        Coee = num / den

        # adaptive threshold from HH (Algorithm 1)
        HH_for_T = HH.abs() if self.use_abs else HH
        Tc = self._adaptive_threshold_from_HH(HH_for_T)

        # hard thresholding on Coee -> Fe_hat
        Fe_hat = torch.where(Coee > Tc, Coee, torch.zeros_like(Coee))
        return Fe_hat, Coee


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math
import torch.nn.functional as F


#######################################################################################
#######   ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓  Semantic Consistency Inference ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓   #######
#######################################################################################
class SemanticConsistencyInference(nn.Module):
    def __init__(self, input_dim, lambda_init=0.5):
        super(SemanticConsistencyInference, self).__init__()

        # MLP for generating Msc
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),  # Concatenate Fin and Fvi
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )

        # Learnable parameter λ
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))  # Lambda is a learnable scalar
        self.sigmoid = nn.Sigmoid()

    def forward(self, Fin, Fvi, t = 0.4):

        cos_sim = F.cosine_similarity(Fin, Fvi, dim=-1)  # Shape [B, L]
        Sm = 0.1 * cos_sim.mean(dim=-1)  # Mean similarity for each sample in the batch [B]

        if torch.any(Sm < t):  # If there are any elements where Sm < t, apply difference elimination

            # Concatenate Fin and Fvi to form a common feature set
            concatenated = torch.cat((Fin, Fvi), dim=-1)  # Shape [B, L, 2*C]

            Mmid = self.mlp(concatenated).squeeze(-1)
            Msc = self.sigmoid(Mmid)  # Shape [B, L]

            Pin = Fin * (1 - Msc.unsqueeze(-1)) + Fvi * Msc.unsqueeze(-1)
            Pvi = Fvi * (1 - Msc.unsqueeze(-1)) + Fin * Msc.unsqueeze(-1)
            Kvi = Fvi - Pvi
            Kin = Fin - Pin

            aFvi = Fvi - self.lambda_param * Kvi
            aFin = Fin - self.lambda_param * Kin
        else:
            aFvi, aFin = Fvi, Fin

        return aFvi, aFin
class SemanticConsistency2D(nn.Module):
    def __init__(self, dim, lambda_init=0.5, t=0.4):
        super().__init__()
        self.sc = SemanticConsistencyInference(dim, lambda_init=lambda_init)
        self.t = t

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        Fin = x1.flatten(2).transpose(1, 2)
        Fvi = x2.flatten(2).transpose(1, 2)

        aFvi, aFin = self.sc(Fin, Fvi, t=self.t)

        x1 = aFin.transpose(1, 2).reshape(B, C, H, W)
        x2 = aFvi.transpose(1, 2).reshape(B, C, H, W)
        return x1, x2
