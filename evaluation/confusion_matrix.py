#!/usr/bin/env python3
"""Build a semantic-segmentation confusion matrix from class-index arrays."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def update_confusion(matrix: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/label shape mismatch: {prediction.shape} vs {target.shape}")
    classes = matrix.shape[0]
    valid = (target >= 0) & (target < classes) & (prediction >= 0) & (prediction < classes)
    encoded = classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    matrix += np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def load_mask(path: Path) -> np.ndarray:
    array = np.asarray(Image.open(path))
    if array.ndim != 2:
        raise ValueError(f"Mask must be single-channel class indices: {path}")
    return array


def from_directories(prediction_dir: Path, label_dir: Path, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    predictions = sorted(prediction_dir.glob("*.png"))
    if not predictions:
        raise FileNotFoundError(f"No PNG predictions in {prediction_dir}")
    for prediction_path in predictions:
        label_path = label_dir / prediction_path.name
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label: {label_path.name}")
        update_confusion(matrix, load_mask(prediction_path), load_mask(label_path))
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--num-classes", required=True, type=int)
    args = parser.parse_args()
    print(from_directories(args.predictions, args.labels, args.num_classes))


if __name__ == "__main__":
    main()
