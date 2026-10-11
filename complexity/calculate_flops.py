#!/usr/bin/env python3
"""Profile FP32 FLOPs at batch 1 and 256 x 256 (1 MAC = 2 FLOPs)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from thop import profile

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "inference"))
from model import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = load_model(config, args.checkpoint, device).float()
    channels = config["model"]["input_channels"]
    optical = torch.zeros(1, int(channels["optical"]), 256, 256, device=device, dtype=torch.float32)
    sar = torch.zeros(1, int(channels["sar"]), 256, 256, device=device, dtype=torch.float32)
    macs, parameters = profile(model, inputs=(optical, sar), verbose=False)
    print("Input: 256x256")
    print("Batch size: 1")
    print("Precision: FP32")
    print(f"Parameters(M): {parameters / 1e6:.6f}")
    print(f"FLOPs(G): {2.0 * macs / 1e9:.6f}")
    print("Convention: 1 multiply-accumulate (MAC) = 2 FLOPs; THOP operator coverage applies.")


if __name__ == "__main__":
    main()
