#!/usr/bin/env python3
"""Count LCCMamba parameters. Protocol: batch 1, FP32, 256 x 256."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

RELEASE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RELEASE_ROOT / "inference"))
from model import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    model = load_model(config, args.checkpoint, torch.device("cpu")).float()
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print("Input: 256x256")
    print("Batch size: 1")
    print("Precision: FP32")
    print(f"Parameters(M): {total / 1e6:.6f}")
    print(f"Trainable Parameters(M): {trainable / 1e6:.6f}")


if __name__ == "__main__":
    main()
