"""Inference-only LCCMamba builder using the frozen release source snapshot."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_SOURCE = Path(__file__).resolve().parent / "model_source"
if str(MODEL_SOURCE) not in sys.path:
    sys.path.insert(0, str(MODEL_SOURCE))

from models.new_model.decoders.MLPAlignDecoder import DecoderHead
from models.new_model.decoders.MutilDecoder import DecoderHead_mloss
from models.new_model.encoders.mamba.segmentors import MambaSeg_t


class LCCMamba(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        dataset_cfg = config["dataset"]
        if model_cfg["backbone"] != "MambaSeg_t":
            raise ValueError("This release supports only the frozen MambaSeg_t backbone")
        if model_cfg["depths"] != [2, 2, 8, 2] or model_cfg["dims"] != [96, 192, 384, 768]:
            raise ValueError("Configuration does not match the released checkpoint architecture")

        self.channels = [96, 192, 384, 768]
        self.backbone = MambaSeg_t()
        input_channels = model_cfg["input_channels"]
        optical_channels = int(input_channels["optical"])
        sar_channels = int(input_channels["sar"])
        self.backbone.encoder_ev.patch_embed = self.backbone._make_patch_embed_v2(
            optical_channels, 96, 4, True, self.backbone.norm_layer, channel_first=True
        )
        self.backbone.encoder_img.patch_embed = self.backbone._make_patch_embed_v2(
            sar_channels, 96, 4, True, self.backbone.norm_layer, channel_first=True
        )
        self.decode_head = DecoderHead_mloss(
            in_channels=self.channels,
            num_classes=int(dataset_cfg["num_classes"]),
            norm_layer=nn.BatchNorm2d,
            embed_dim=int(model_cfg["decoder_embed_dim"]),
        )
        self.decode_head1 = DecoderHead(
            in_channels=self.channels,
            num_classes=int(dataset_cfg["num_classes"]),
            norm_layer=nn.BatchNorm2d,
            embed_dim=int(model_cfg["decoder_embed_dim"]),
        )

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        output_size = optical.shape[-2:]
        features = self.backbone(optical, sar)
        _, _, _, fused = self.decode_head(features["feat1"], features["feat2"])
        return F.interpolate(fused, size=output_size, mode="bilinear", align_corners=False)


def load_model(config: dict, checkpoint: str | Path, device: torch.device) -> LCCMamba:
    model = LCCMamba(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(torch.is_tensor(value) for value in state.values()):
        raise TypeError("Checkpoint must be a tensor-only state_dict")
    if state and next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
