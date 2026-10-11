from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join
from .vmamba import Backbone_VSSM, Permute, LayerNorm2d
from .fuse_modules import DDIM
try:
    from thop import clever_format, profile
except ImportError:  # Optional for inference; required only for standalone profiling.
    clever_format = profile = None
import time
from models.new_model.encoders.mamba.fuse_modules import FourierEdgeEnhancementModule as edge
from models.new_model.encoders.mamba.fuse_modules import EGM_FDEG as egm_fdeg
#新增模块
from models.new_model.Module.GCNFeatureFusion import GCNFeatureFusion

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(True),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        )
    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)
class MambaSeg(nn.Module):
    def __init__(self, ver_img='vmamba_tiny', ver_ev='vmamba_tiny', num_classes=6, 
                 fuse=None, num_channels_img=3, pretrained_img=None, 
                 num_channels_ev=3, pretrained_ev=None, data_type=None, img_size=None, if_viz=False):
        super().__init__()
        self.num_channels_img = num_channels_img
        self.num_channels_ev = num_channels_ev
        self.out_indices=(0, 1, 2, 3)

        dim_dict = { 'vmamba_tiny': [96, 192, 384, 768],
                        'vmamba_small': [96, 192, 384, 768],
                        'vmamba_base': [128, 256, 512, 1024]}

        patchembed_version = "v2"
        _make_patch_embed = dict(
            v1=self._make_patch_embed, 
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )        

        self.patch_size = 4
        self.patch_norm = True
        self.norm_layer = "ln2d"
        self.channel_first = (self.norm_layer.lower() in ["bn", "ln2d"])

        dim_img = dim_dict[ver_img]
        dim_ev = dim_dict[ver_ev]
        # ceiling
        space_dim = [(ceil(img_size[0] / self.patch_size), ceil(img_size[1] / self.patch_size))]
        for i in range(len(dim_img)-1):
            space_dim.append((ceil(space_dim[i][0] / 2), ceil(space_dim[i][1] / 2)))
        self.space_dim = space_dim  # (H, W) for each stage
        self.num_tokens = [space_dim[i][0] * space_dim[i][1] for i in range(len(self.space_dim))]  # number of tokens for each stage

        # Select an encoder 设置图像-事件双分支的编码器
        
        self.encoder_img = Backbone_VSSM(patch_size=4, in_chans=3, num_classes=1000, 
                                         depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
                                         ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", 
                                         ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", 
                                         mlp_ratio=4.0,downsample_version="v3", patchembed_version="v2", 
                                         norm_layer=self.norm_layer, use_checkpoint=False,
                                         out_indices=self.out_indices, pretrained=pretrained_img)
        

        self.encoder_ev = Backbone_VSSM(patch_size=4, in_chans=3, num_classes=1000, 
                                         depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
                                         ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", 
                                         ssm_conv=3, ssm_conv_bias=False, forward_type="v05_noz", 
                                         mlp_ratio=4.0,downsample_version="v3", patchembed_version="v2", 
                                         norm_layer=self.norm_layer, use_checkpoint=False,
                                         out_indices=self.out_indices, pretrained=pretrained_ev)

        # Replace the first layer of the encoder with a channel-adaptive layer
        self.norm_layer: nn.Module = _NORMLAYERS.get(self.norm_layer.lower(), None)
        if self.num_channels_img != 3:
            self.encoder_img.patch_embed = _make_patch_embed(self.num_channels_img, dim_img[0], self.patch_size, self.patch_norm, self.norm_layer, channel_first=self.channel_first)

            self.encoder_ev.patch_embed = _make_patch_embed(self.num_channels_ev, dim_ev[0], self.patch_size, self.patch_norm, self.norm_layer, channel_first=self.channel_first)


        self.augs = nn.ModuleList()



        self.out_convs = nn.ModuleList([
            nn.Conv2d(in_channels = dim_img[i] * 2,
                      out_channels = dim_img[i],
                      kernel_size = 1)
            for i in range(len(dim_img))
        ])

        self.proj_head = ProjectionHead(dim_in=dim_img[self.out_indices[-1]], proj_dim=256)

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def _load_backbone_except_patch_embed(self, pretrained):
        state = torch.load(pretrained, map_location='cpu')
        if 'model' in state:
            state = state['model']

        img_state = {}
        ev_state = {}

        for k, v in state.items():
            # �� 关键：完全跳过 patch_embed
            if k.startswith('patch_embed'):
                continue

            img_state[k] = v
            ev_state[k] = v

        missing_img, unexpected_img = self.encoder_img.load_state_dict(
            img_state, strict=False
        )
        missing_ev, unexpected_ev = self.encoder_ev.load_state_dict(
            ev_state, strict=False
        )

        print('[Image branch] missing:', missing_img)
    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_img_ev_model(self, pretrained, share_backbone=True)
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError("pretrained must be str or None")

    def forward(self, x_ev, x_img):

        return self.forward_fuse_v2(x_ev, x_img)
    
    def forward_fuse_v2(self, x_ev, x_img):
        B, _, H0, W0 = x_ev.shape
        outs = []
        imgs=[]
        evs=[]
        """ Encode the event-image fusion representation """
        # patch_embed
        x_ev = self.encoder_ev.patch_embed(x_ev)
        x_img = self.encoder_img.patch_embed(x_img)
        outs_ev, outs_img = [], []
        last_img, last_ev = None, None
        # loop encoder_ev and encoder_img
        for i, (layer_ev, layer_img) in enumerate(zip(self.encoder_ev.layers, self.encoder_img.layers)):
            # Get Stage i output
            o_ev = layer_ev.blocks(x_ev)
            o_img = layer_img.blocks(x_img)
            if i in self.out_indices:
                # norm
                # if i<3:
                #     norm_layer_ev = getattr(self.encoder_ev, f'outnorm{i}')
                #     norm_layer_img = getattr(self.encoder_img, f'outnorm{i}')
                #     # o_ev = self.edge[i](o_ev)
                #     # o_img= self.edge[i](o_img)
                #
                #     out_ev = norm_layer_ev(o_ev)
                #     out_img = norm_layer_img(o_img)
                # else:
                norm_layer_ev = getattr(self.encoder_ev, f'outnorm{i}')
                norm_layer_img = getattr(self.encoder_img, f'outnorm{i}')
                # o_ev = self.edge[i](o_ev)
                # o_img = self.edge[i](o_img)

                out_ev = norm_layer_ev(o_ev)
                out_img = norm_layer_img(o_img)
                # add to backbone outs
                outs_ev.append(out_ev.contiguous())
                outs_img.append(out_img.contiguous())


            #fuse
            # img, ev = self.ccbs[i](outs_img[i], outs_ev[i])
            # # add fuse out to backbone next stage input
            # x_ev = o_ev + ev
            # x_img = o_img + img
            # x_ev = layer_ev.downsample(x_ev)
            # x_img = layer_img.downsample(x_img)
            # concat_out = torch.cat((img, ev), dim=1)
            # outs.append(self.out_convs[i](concat_out))
            # imgs.append(img)
            # evs.append(ev)
            #对比学习
            # if i == self.out_indices[-1]:
            #     last_img, last_ev = img, ev
        # ====== proj head（一定要在循环后做）======
        # if (last_img is not None) and (last_ev is not None) and hasattr(self, "proj_head") and (
        #         self.proj_head is not None):
        #     proj1 = self.proj_head(last_img)  # img 分支投影
        #     proj2 = self.proj_head(last_ev)  # ev  分支投影
        # # else:
        # #     proj1 = None
        # #     proj2 = None
        #
        # # ====== 返回 dict（你想要的 feat1/feat2/proj1/proj2）======
        # feat_dict = {
        #     "feat1": outs_img,  # 类比 SiamContrast 的 features1（多层 list）
        #     "feat2": outs_ev,  # 类比 SiamContrast 的 features2（多层 list）
        #     "proj1": proj1,  # 类比 SiamContrast 的 proj1
        #     "proj2": proj2,  # 类比 SiamContrast 的 proj2
        #     "out": outs,  # 你的 decoder 真正要吃的（多尺度融合特征 list）
        # }
        # return feat_dict
        # no fuse
            # ===== Ablation: w/o CCBS (DDIM) fusion =====
            img = outs_img[i]  # or out_img
            ev = outs_ev[i]  # or out_ev

            # 两个分支各自走各自的主干，不加跨模态残差
            x_ev = layer_ev.downsample(o_ev)
            x_img = layer_img.downsample(o_img)
            # 仍然给 decoder 提供 fused-like 多尺度特征（维度不变：2C -> C）
            concat_out = torch.cat((img, ev), dim=1)
            outs.append(self.out_convs[i](concat_out))
            imgs.append(img)
            evs.append(ev)
            if i == self.out_indices[-1]:
               #last_img, last_ev,  = img, ev,
               last_img, last_ev,last_fuse = img, ev,outs[i]

                # ====== proj head（一定要在循环后做）======
        if (last_img is not None) and (last_ev is not None) and hasattr(self, "proj_head") and (
            self.proj_head is not None):

                proj1 = self.proj_head(last_img)  # img 分支投影
                proj2 = self.proj_head(last_ev)  # ev  分支投影
                proj3 = self.proj_head(last_fuse)
        feat_dict = {
                "feat1": outs_img,  # 类比 SiamContrast 的 features1（多层 list）
                "feat2": outs_ev,  # 类比 SiamContrast 的 features2（多层 list）
                "proj1": proj1,  # 类比 SiamContrast 的 proj1
                "proj2": proj2,  # 类比 SiamContrast 的 proj2
                "projfuse":proj3,
                "out": outs,  # 你的 decoder 真正要吃的（多尺度融合特征 list）
        }
        return feat_dict


        """ Generate the segmentation mask """
        # x = self.decoder(outs)
        # x = F.interpolate(x, size=[H0, W0], mode='bilinear', align_corners=False)
        #return outs
        #return  outs_img,outs_ev
        #return  imgs,evs

def load_img_ev_model(model, model_file, share_backbone=True, is_restore=False):
    t_start = time.time()

    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location='cpu')
        if 'model' in raw_state_dict:
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file

    img_state = {}
    ev_state = {}

    for k, v in raw_state_dict.items():
        # ================== �� 核心修改开始 ==================
        # 跳过 patch_embed（通道/模态相关，不能直接迁移）
        if k.startswith('patch_embed'):
            continue
        # ================== �� 核心修改结束 ==================

        img_state[k] = v
        if share_backbone:
            ev_state[k] = v

    if is_restore:
        img_state = {f"module.{k}": v for k, v in img_state.items()}
        ev_state = {f"module.{k}": v for k, v in ev_state.items()}

    missing, unexpected = model.encoder_img.load_state_dict(img_state, strict=False)
    print(f"[IMG] missing={len(missing)}, unexpected={len(unexpected)}")

    if share_backbone:
        missing, unexpected = model.encoder_ev.load_state_dict(ev_state, strict=False)
        print(f"[EV ] missing={len(missing)}, unexpected={len(unexpected)}")

    t_end = time.time()
    print(f"Load img/ev model done, time: {t_end - t_start:.2f}s")




class MambaSeg_t(MambaSeg):
    def __init__(self, **kwargs):
        super(MambaSeg_t, self).__init__(  ver_img='vmamba_tiny',
                                           ver_ev='vmamba_tiny',
                                           num_classes=8,
                                           num_channels_img=1,
                                           num_channels_ev=4,
                                           # img_size=(512, 512),   # ⚠️ 一定要传
                                           img_size=(256,256),  # ⚠️ 一定要传
                                           pretrained_img=None,
                                           pretrained_ev=None, **kwargs)

if __name__ == '__main__':
    model = MambaSeg_t().cuda()
    # print(model)
    left = torch.randn(1, 1, 256, 256).cuda()
    right = torch.randn(1, 4, 256,256).cuda()

    # summary(model, [(4, 256, 256), (1, 256, 256)])
    flops, params = profile(model, (left, right), verbose=False)

    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % flops)
    print('Total params: %s' % params)
