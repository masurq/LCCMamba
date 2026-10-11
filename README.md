# LCCMamba Release

## 1. Introduction

This repository provides pretrained checkpoints, inference code, evaluation utilities, configuration files, and experimental records for the manuscript revision stage.

The complete training framework will be released after final acceptance.

## 2. Method overview

LCCMamba is a dual-stream optical/SAR semantic-segmentation model. The released architecture uses a `MambaSeg_t` backbone with stage depths `[2, 2, 8, 2]` and dimensions `[96, 192, 384, 768]`, followed by the full multi-level fusion decoder. The preserved training configurations use SHAS for the unimodal contrastive branches and CHS for the fusion contrastive branch, with a shared sampling budget of 1,024.

The release source is a frozen inference-only snapshot. Model definitions are retained because they are required to load the checkpoints; training code and loss execution paths are not included.

## 3. Repository structure

```text
LCCMamba_release/
├── README.md
├── .gitignore
├── configs/
│   ├── whu.yaml
│   ├── potsdam.yaml
│   └── yeseg.yaml
├── pretrained/
│   └── checkpoint_info.txt
├── inference/
│   ├── predict.py
│   ├── model.py
│   ├── requirements.txt
│   └── model_source/
├── evaluation/
│   ├── metric.py
│   ├── evaluate.py
│   └── confusion_matrix.py
├── complexity/
│   ├── count_parameters.py
│   └── calculate_flops.py
└── examples/
    ├── optical/
    ├── sar/
    └── predictions/
```

## 4. Environment

The preserved evaluation logs used Python with PyTorch `2.7.1+cu128`. The minimal pinned package list is in `inference/requirements.txt`. A CUDA-capable environment is recommended for practical inference. The selective-scan implementation can fall back to PyTorch when its optimized extension is unavailable, but that fallback is substantially slower.

## 5. Installation

Run commands from the repository root. Create an isolated environment, install a PyTorch build suitable for the local CUDA driver, and install the remaining dependencies:

```bash
python -m venv .venv
```

Activate the newly created environment on Linux/macOS:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies:

```text
python -m pip install -r inference/requirements.txt
```

If the pinned PyTorch build is not compatible with the target machine, install the matching official PyTorch build first and then install the other listed packages without replacing PyTorch.

## 6. Inference

Download the pretrained checkpoint from the GitHub Release assets. For example, to reproduce the WHU-OPT-SAR experiment, download:`LCCMamba_WHU.pth`

and place it in:
```text
pretrained/LCCMamba_WHU.pth
```

Paired optical and SAR images with the same filename stem should be placed in:

```text
examples/optical/
examples/sar/
```

The input images should be spatially aligned and have the channel numbers specified in the corresponding YAML configuration file.

Example command:

```bash
python inference/predict.py \
    --config configs/whu.yaml \
    --checkpoint pretrained/LCCMamba_WHU.pth
```

Predictions are saved in:

```text
examples/predictions/
```

as single-channel PNG masks containing predicted class indices.

Example images are not included because the original datasets are subject to redistribution restrictions.


## 7. Evaluation

Ground-truth and prediction masks must be single-channel PNG files with identical names and class-index values.

Ground-truth labels are not included in this repository because the original datasets are subject to redistribution restrictions. Users should obtain the corresponding datasets from their authorized sources and prepare the labels according to the original dataset protocols.

After generating predictions, run:

```bash
python evaluation/evaluate.py \
    --config configs/whu.yaml \
    --predictions examples/predictions \
    --labels path/to/labels
```

The evaluator reports OA, mIoU, mF1, Kappa, per-class IoU/F1, and the confusion matrix.

For every class, the implementation verifies:

\[
IoU = \frac{F1}{2-F1}
\]

The transformation is applied at the class level and does not imply that separately averaged mIoU and mF1 satisfy the same relationship.

## 8. Complexity analysis

After downloading the checkpoint into `pretrained/`, run the following commands. The fixed protocol is batch size 1, FP32, and a `256 × 256` optical/SAR pair:

```bash
python complexity/count_parameters.py \
    --config configs/whu.yaml \
    --checkpoint pretrained/LCCMamba_WHU.pth

python complexity/calculate_flops.py \
    --config configs/whu.yaml \
    --checkpoint pretrained/LCCMamba_WHU.pth

## 9. Available checkpoints

The pretrained checkpoints are distributed separately through GitHub Releases due to GitHub file size limitations.

The repository initially contains only `pretrained/checkpoint_info.txt` in the `pretrained/` directory.

Checkpoint download:

Download the following assets from this repository's GitHub Releases page after the release assets have been published:

- LCCMamba_WHU.pth
- LCCMamba_Potsdam.pth
- LCCMamba_YESeg.pth

Place the downloaded files in `pretrained/`. The directory after downloading should be:

```text
pretrained/
├── LCCMamba_WHU.pth
├── LCCMamba_Potsdam.pth
├── LCCMamba_YESeg.pth
└── checkpoint_info.txt
```

| Dataset | File | Input channels | Classes | Selection |
|---|---|---:|---:|---|
| WHU-OPT-SAR | `pretrained/LCCMamba_WHU.pth` | optical 4, SAR 1 | 8 | best training Pixel Acc checkpoint used by the preserved test log |
| Potsdam | `pretrained/LCCMamba_Potsdam.pth` | optical 4, SAR 1 | 6 | best training Pixel Acc checkpoint used by the preserved test log |
| YESeg-OPT-SAR | `pretrained/LCCMamba_YESeg.pth` | optical 3, SAR 1 | 8 | best training Pixel Acc checkpoint used by the preserved test log |

## 10. Dataset description

- **WHU-OPT-SAR**: paired four-channel optical and one-channel SAR imagery. The released configuration has eight label indices; the background class is included in OA and omitted from macro IoU/F1, matching the preserved evaluation protocol.
- **Potsdam**: paired four-channel optical and one-channel auxiliary-modality imagery with six label indices. The preserved report averages IoU/F1 over classes 1–5.
- **YESeg-OPT-SAR**: paired three-channel optical and one-channel SAR imagery with eight label indices. The preserved report averages IoU/F1 over classes 1–6.

Dataset images and labels are not redistributed. Users must obtain each dataset from its authorized source and reproduce the split described by the corresponding experiment protocol.

