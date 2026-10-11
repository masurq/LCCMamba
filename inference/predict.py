#!/usr/bin/env python3
"""Run paired optical/SAR inference and save single-channel class-index masks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from model import load_model

SUPPORTED = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        try:
            import tifffile
            array = tifffile.imread(path)
        except ImportError:
            array = np.asarray(Image.open(path))
    else:
        array = np.asarray(Image.open(path))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim == 3 and array.shape[0] <= 4 and array.shape[-1] > 4:
        array = np.moveaxis(array, 0, -1)
    return array


def to_tensor(array: np.ndarray, mean: list[float], std: list[float], device: torch.device) -> torch.Tensor:
    if array.shape[-1] != len(mean):
        raise ValueError(f"Expected {len(mean)} channels, received shape {array.shape}")
    scale = 255.0 if np.issubdtype(array.dtype, np.integer) else 1.0
    array = array.astype(np.float32) / scale
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1))).unsqueeze(0).to(device)


def matching_pairs(optical_dir: Path, sar_dir: Path):
    for optical in sorted(path for path in optical_dir.iterdir() if path.suffix.lower() in SUPPORTED):
        candidates = [sar_dir / optical.name]
        candidates.extend(sar_dir / f"{optical.stem}{suffix}" for suffix in sorted(SUPPORTED))
        sar = next((path for path in candidates if path.is_file()), None)
        if sar is None:
            raise FileNotFoundError(f"No SAR image with stem '{optical.stem}'")
        yield optical, sar


def main() -> None:
    release_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--optical-dir", type=Path, default=release_root / "examples/optical")
    parser.add_argument("--sar-dir", type=Path, default=release_root / "examples/sar")
    parser.add_argument("--output-dir", type=Path, default=release_root / "examples/predictions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = load_model(config, args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(matching_pairs(args.optical_dir, args.sar_dir))
    if not pairs:
        print("No paired images found. Add same-stem files to examples/optical and examples/sar.")
        return

    dataset = config["dataset"]
    for optical_path, sar_path in pairs:
        optical_array = read_image(optical_path)
        sar_array = read_image(sar_path)
        if optical_array.shape[:2] != sar_array.shape[:2]:
            raise ValueError(f"Spatial-size mismatch for {optical_path.name}")
        optical = to_tensor(optical_array, dataset["optical_mean"], dataset["optical_std"], device)
        sar = to_tensor(sar_array, dataset["sar_mean"], dataset["sar_std"], device)
        with torch.inference_mode():
            prediction = model(optical, sar).argmax(1)[0].cpu().numpy().astype(np.uint8)
        destination = args.output_dir / f"{optical_path.stem}.png"
        Image.fromarray(prediction, mode="L").save(destination)
        print(destination)


if __name__ == "__main__":
    main()
