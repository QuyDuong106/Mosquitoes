#!/usr/bin/env python3
"""
Run Integrated Gradients and SmoothGrad for the best classification checkpoint.

Best model selected from:
results on new labels/final results - top methods from phase1&2 ver/test_summary.csv

Best macro-F1:
resnet50_weighted_loss_lr0.0005_bs32_g1.0_winverse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, NoiseTunnel, Saliency
from PIL import Image
from torchvision import models, transforms


DEFAULT_ROOT = Path("/Users/quyduong106/projects/COMPSCI 760")
DEFAULT_REPO = DEFAULT_ROOT / "Mosquitoes"
DEFAULT_IMAGE_DIR = DEFAULT_ROOT / "mosquito_dataset_ai_v1/image_crop"
DEFAULT_CSV = DEFAULT_IMAGE_DIR / "cropped_annotations.csv"
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO
    / "results on new labels/final results - top methods from phase1&2 ver/weighted_loss_winverse_lr0.0005.pth"
)
DEFAULT_IMAGE = DEFAULT_IMAGE_DIR / "train_00000_0_crop.jpeg"
DEFAULT_OUT_DIR = DEFAULT_REPO / "xai_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IG and SmoothGrad xAI maps for the best classifier.")
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--smoothgrad-samples", type=int, default=30)
    parser.add_argument("--smoothgrad-stdev", type=float, default=0.1)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_classes(csv_path: Path) -> list[str]:
    df = pd.read_csv(csv_path)
    return sorted(df["class_label"].astype(str).unique())


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    clean_state = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("model.", "")
        clean_state[key] = value
    return clean_state


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)


def normalize_map(attr_map: np.ndarray) -> np.ndarray:
    attr_map = attr_map - attr_map.min()
    return attr_map / (attr_map.max() + 1e-8)


def overlay_heatmap(pil_img: Image.Image, attr_map: np.ndarray) -> np.ndarray:
    attr_map = cv2.resize(attr_map, pil_img.size)
    heatmap = cv2.applyColorMap(np.uint8(255 * attr_map), cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_bgr, 0.55, heatmap, 0.45, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def make_panel(original: Image.Image, ig_rgb: np.ndarray, smoothgrad_rgb: np.ndarray) -> np.ndarray:
    original_rgb = np.array(original.convert("RGB"))
    height = max(original_rgb.shape[0], ig_rgb.shape[0], smoothgrad_rgb.shape[0])

    def pad_to_height(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == height:
            return img
        pad = height - img.shape[0]
        return np.pad(img, ((0, pad), (0, 0), (0, 0)), constant_values=255)

    return np.concatenate([pad_to_height(original_rgb), pad_to_height(ig_rgb), pad_to_height(smoothgrad_rgb)], axis=1)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint_path}\n"
            "If this file is only on the GitHub branch, run:\n"
            "git checkout fine-grained-classification/imbalance-methods -- "
            "\"results on new labels/final results - top methods from phase1&2 ver/weighted_loss_winverse_lr0.0005.pth\""
        )

    device = get_device()
    class_names = load_classes(args.csv_path)
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    model = build_model(num_classes=len(class_names))
    load_checkpoint(model, args.checkpoint_path, device)
    model.to(device)
    model.eval()

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    pil_img = Image.open(args.image_path).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    pred_idx = int(output.argmax(dim=1).item())
    pred_class = idx_to_class[pred_idx]
    pred_prob = float(torch.softmax(output, dim=1)[0, pred_idx].detach().cpu().item())

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(input_tensor).to(device)
    ig_attr, delta = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=pred_idx,
        return_convergence_delta=True,
    )
    ig_map = ig_attr.squeeze().detach().cpu().numpy().transpose(1, 2, 0).sum(axis=2)
    ig_map = normalize_map(np.maximum(ig_map, 0))
    ig_rgb = overlay_heatmap(pil_img, ig_map)

    saliency = Saliency(model)
    smoothgrad = NoiseTunnel(saliency)
    sg_attr = smoothgrad.attribute(
        input_tensor,
        target=pred_idx,
        nt_type="smoothgrad",
        nt_samples=args.smoothgrad_samples,
        stdevs=args.smoothgrad_stdev,
    )
    sg_map = sg_attr.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    sg_map = normalize_map(np.abs(sg_map).mean(axis=2))
    sg_rgb = overlay_heatmap(pil_img, sg_map)

    stem = args.image_path.stem
    ig_path = args.out_dir / f"{stem}_integrated_gradients.png"
    smoothgrad_path = args.out_dir / f"{stem}_smoothgrad.png"
    panel_path = args.out_dir / f"{stem}_xai_panel.png"

    save_rgb(ig_path, ig_rgb)
    save_rgb(smoothgrad_path, sg_rgb)
    save_rgb(panel_path, make_panel(pil_img, ig_rgb, smoothgrad_rgb=sg_rgb))

    print("Best model: resnet50_weighted_loss_lr0.0005_bs32_g1.0_winverse")
    print("Best macro-F1 from test_summary.csv: 0.6573609807309273")
    print("Image:", args.image_path)
    print("Predicted class:", pred_class)
    print("Predicted probability:", f"{pred_prob:.4f}")
    print("Integrated Gradients convergence delta:", float(delta.detach().cpu().item()))
    print("Saved:", ig_path)
    print("Saved:", smoothgrad_path)
    print("Saved:", panel_path)


if __name__ == "__main__":
    main()
