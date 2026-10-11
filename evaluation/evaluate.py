#!/usr/bin/env python3
"""Evaluate class-index PNG masks and report OA, mIoU, mF1, and Kappa."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from confusion_matrix import from_directories
from metric import metrics_from_confusion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    matrix = from_directories(args.predictions, args.labels, int(dataset["num_classes"]))
    result = metrics_from_confusion(matrix, dataset["metric_classes"])
    result["confusion_matrix"] = matrix.tolist()
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
