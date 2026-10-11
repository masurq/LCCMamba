#!/usr/bin/env python3
"""Profile the final-output LCCMamba graph with the paper's fvcore protocol.

Protocol: paired 4-channel optical and 1-channel SAR inputs, batch size 1,
256 x 256, FP32, and 1 MAC = 2 FLOPs. Selective Scan uses the upstream
VMamba recurrence formula already present in the frozen model source.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from fvcore.nn import FlopCountAnalysis

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "inference"))

from model import load_model
from models.new_model.encoders.mamba.csms6s import selective_scan_flop_jit


# This is the same arithmetic-MAC convention used by the paper's unified
# Table-8 profiler. Layout, indexing, activation, reduction, and pooling ops do
# not contribute multiply-accumulates; the reported FLOPs are twice the MACs.
ZERO_ARITHMETIC_OPS = {
    "aten::reshape", "aten::view", "aten::permute", "aten::transpose",
    "aten::contiguous", "aten::flatten", "aten::flip", "aten::slice",
    "aten::select", "aten::index", "aten::index_select", "aten::expand",
    "aten::unsqueeze", "aten::squeeze", "aten::cat", "aten::split",
    "aten::chunk", "aten::clone", "aten::to", "aten::type_as",
    "aten::add", "aten::add_", "aten::sub", "aten::rsub", "aten::mul",
    "aten::mul_", "aten::div", "aten::pow", "aten::sqrt", "aten::abs",
    "aten::sum", "aten::mean", "aten::std", "aten::var_mean",
    "aten::sigmoid", "aten::silu", "aten::gelu", "aten::hardtanh_",
    "aten::leaky_relu", "aten::prelu", "aten::softmax", "aten::exp",
    "aten::neg", "aten::log_sigmoid", "aten::maximum", "aten::where",
    "aten::cumsum", "aten::tril", "aten::linspace", "aten::repeat",
    "aten::expand_as", "aten::feature_dropout", "aten::avg_pool2d",
    "aten::max_pool2d", "aten::adaptive_max_pool2d",
    "aten::upsample_bicubic2d",
}


def _zero_jit(inputs, outputs):
    return 0


class SelectiveScanCuda(torch.autograd.Function):
    """Shape-preserving trace proxy used by the existing Table-8 profiler."""

    @staticmethod
    def forward(ctx, u, D, A):
        return torch.zeros_like(u)


class SelectiveScanFn(torch.autograd.Function):
    """DGFM Selective Scan trace proxy used by the Table-8 profiler."""

    @staticmethod
    def forward(ctx, u, D, A):
        return torch.zeros_like(u)


def _encoder_scan(u, delta, A, B, C, D=None, delta_bias=None, *args, **kwargs):
    if D is None:
        D = torch.zeros(u.shape[1], device=u.device, dtype=u.dtype)
    # Keep only u/A/D as trace dependencies, matching the existing Table-8
    # final-output profiler, while retaining A at input index 2 for the frozen
    # selective_scan_flop_jit handler.
    return SelectiveScanCuda.apply(u, D, A)


def _dgfm_scan(u, delta, A, B, C, D=None, z=None, delta_bias=None, **kwargs):
    if D is None:
        D = torch.zeros(u.shape[1], device=u.device, dtype=u.dtype)
    return SelectiveScanFn.apply(u, D, A)


class SelectiveScanPatch(AbstractContextManager):
    """Temporarily expose Selective Scan as fvcore-countable PythonOps."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.saved: list[tuple[object, str, object]] = []

    def __enter__(self):
        for name, module in list(sys.modules.items()):
            if module is None:
                continue
            if name.endswith(".vmamba") and hasattr(module, "selective_scan_fn"):
                self.saved.append((module, "selective_scan_fn", module.selective_scan_fn))
                module.selective_scan_fn = _encoder_scan
        for module in self.model.modules():
            if module.__class__.__name__ == "single_SS2D" and hasattr(module, "selective_scan"):
                self.saved.append((module, "selective_scan", module.selective_scan))
                module.selective_scan = _dgfm_scan
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for obj, attribute, value in reversed(self.saved):
            setattr(obj, attribute, value)
        return False


def inference_parameters(model: nn.Module) -> int:
    """Count final-output dependencies without changing the loaded model."""
    excluded = [model.decode_head1]
    for name in ("proj_head", "out_convs"):
        if hasattr(model.backbone, name):
            excluded.append(getattr(model.backbone, name))
    for name in (
        "linear_pred1", "linear_pred2", "linear_pred3",
        "Aligned_c2", "Aligned_c3", "Aligned_c4",
    ):
        if hasattr(model.decode_head, name):
            excluded.append(getattr(model.decode_head, name))
    excluded_ids = {id(parameter) for module in excluded for parameter in module.parameters()}
    return sum(
        parameter.numel()
        for parameter in {id(p): p for p in model.parameters()}.values()
        if id(parameter) not in excluded_ids
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = load_model(config, args.checkpoint, device).float().eval()

    channels = config["model"]["input_channels"]
    optical = torch.zeros(
        1, int(channels["optical"]), 256, 256, device=device, dtype=torch.float32
    )
    sar = torch.zeros(
        1, int(channels["sar"]), 256, 256, device=device, dtype=torch.float32
    )

    with SelectiveScanPatch(model):
        analysis = FlopCountAnalysis(model, (optical, sar))
        analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        for operator in ZERO_ARITHMETIC_OPS:
            analysis.set_op_handle(operator, None)
        scan_handler = partial(selective_scan_flop_jit, backend="prefixsum", verbose=False)
        analysis.set_op_handle("prim::PythonOp.SelectiveScanCuda", scan_handler)
        analysis.set_op_handle("prim::PythonOp.SelectiveScanFn", scan_handler)
        for operator in (
            "prim::PythonOp.CrossScan", "prim::PythonOp.CrossMerge",
            "prim::PythonOp.CrossScanF", "prim::PythonOp.CrossMergeF",
            "prim::PythonOp.CrossScanTritonF", "prim::PythonOp.CrossMergeTritonF",
        ):
            analysis.set_op_handle(operator, _zero_jit)
        macs = float(analysis.total())
        by_operator = dict(analysis.by_operator())
        unsupported = dict(analysis.unsupported_ops())

    parameters = inference_parameters(model)
    scan_unsupported = {
        name: count for name, count in unsupported.items() if "selectivescan" in name.lower()
    }
    if scan_unsupported:
        raise RuntimeError(f"Selective Scan was not counted: {scan_unsupported}")

    print("Profiler: fvcore.nn.FlopCountAnalysis")
    print("Input optical: [1, 4, 256, 256]")
    print("Input SAR: [1, 1, 256, 256]")
    print("Batch size: 1")
    print("Precision: FP32")
    print(f"Parameters: {parameters:,} ({parameters / 1e6:.6f} M)")
    print(f"MACs: {int(macs):,} ({macs / 1e9:.6f} G)")
    print(f"FLOPs: {int(2.0 * macs):,} ({2.0 * macs / 1e9:.6f} G)")
    scan_macs = sum(value for name, value in by_operator.items() if "SelectiveScan" in name)
    print(f"Selective Scan MACs: {int(scan_macs):,} ({scan_macs / 1e9:.6f} G)")
    print(f"Unsupported operators: {unsupported if unsupported else '{}'}")
    print("Selective Scan handler: registered")
    print("Convention: 1 MAC = 2 FLOPs")


if __name__ == "__main__":
    main()
