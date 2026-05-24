#!/usr/bin/env python3
"""
Run image-classification xAI with Integrated Gradients and SmoothGrad.

By default, this script:
1. Reads cropped_annotations.csv.
2. Creates a stratified train/validation/test split by class_label.
3. Samples crop images from the test split.
4. Runs Integrated Gradients and SmoothGrad for each sampled image.

Use --image-path to explain one specific image instead of sampling from the split.
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
from sklearn.model_selection import train_test_split
from torchvision import models, transforms


DEFAULT_ROOT = Path("/Users/quyduong106/projects/COMPSCI 760")
DEFAULT_REPO = DEFAULT_ROOT / "Mosquitoes"
DEFAULT_IMAGE_DIR = DEFAULT_ROOT / "image_crop_new/image_crop_new"
DEFAULT_CSV = DEFAULT_IMAGE_DIR / "cropped_annotations.csv"
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO
    / "results on new labels/final results - top methods from phase1&2 ver/weighted_loss_winverse_lr0.0005.pth"
)
DEFAULT_OUT_DIR = DEFAULT_REPO / "xai_outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Integrated Gradients and SmoothGrad for the best classifier.")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--image-path", type=Path, default=None, help="Optional single image to explain.")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--smoothgrad-samples", type=int, default=30)
    parser.add_argument("--smoothgrad-stdev", type=float, default=0.1)
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_annotations(csv_path: Path, image_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates("img_fName").copy()
    df["image_path"] = df["img_fName"].apply(lambda name: image_dir / str(name))
    df = df[df["image_path"].apply(lambda path: path.exists())].reset_index(drop=True)
    return df


def load_split(df: pd.DataFrame, test_size: float, val_size: float, seed: int):
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=df["class_label"],
    )

    relative_test_size = test_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=seed,
        stratify=temp_df["class_label"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def sample_xai_images(test_df: pd.DataFrame, samples_per_class: int, seed: int) -> pd.DataFrame:
    return (
        test_df.groupby("class_label", group_keys=False)
        .apply(lambda group: group.sample(n=min(samples_per_class, len(group)), random_state=seed))
        .reset_index(drop=True)
    )


def single_image_df(df: pd.DataFrame, image_path: Path) -> pd.DataFrame:
    image_path = image_path.resolve()
    matches = df[df["image_path"].apply(lambda path: Path(path).resolve() == image_path)]

    if matches.empty:
        raise ValueError(
            f"Image not found in annotations: {image_path}\n"
            "Make sure --image-path points to a file listed in cropped_annotations.csv."
        )

    return matches.head(1).reset_index(drop=True)


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


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    pad = height - img.shape[0]
    return np.pad(img, ((0, pad), (0, 0), (0, 0)), constant_values=255)


def make_panel(original: Image.Image, ig_rgb: np.ndarray, smoothgrad_rgb: np.ndarray) -> np.ndarray:
    original_rgb = np.array(original.convert("RGB"))
    height = max(original_rgb.shape[0], ig_rgb.shape[0], smoothgrad_rgb.shape[0])
    return np.concatenate(
        [
            pad_to_height(original_rgb, height),
            pad_to_height(ig_rgb, height),
            pad_to_height(smoothgrad_rgb, height),
        ],
        axis=1,
    )


def explain_image(
    *,
    model: nn.Module,
    image_path: Path,
    true_class: str,
    class_names: list[str],
    preprocess,
    device: torch.device,
    out_dir: Path,
    smoothgrad_samples: int,
    smoothgrad_stdev: float,
) -> dict[str, object]:
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_idx = int(output.argmax(dim=1).item())
    pred_class = idx_to_class[pred_idx]
    pred_prob = float(probs[0, pred_idx].detach().cpu().item())

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

    smoothgrad = NoiseTunnel(Saliency(model))
    sg_attr = smoothgrad.attribute(
        input_tensor,
        target=pred_idx,
        nt_type="smoothgrad",
        nt_samples=smoothgrad_samples,
        stdevs=smoothgrad_stdev,
    )
    sg_map = sg_attr.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    sg_map = normalize_map(np.abs(sg_map).mean(axis=2))
    sg_rgb = overlay_heatmap(pil_img, sg_map)

    class_dir = out_dir / true_class
    stem = image_path.stem
    ig_path = class_dir / f"{stem}_ig.png"
    smoothgrad_path = class_dir / f"{stem}_smoothgrad.png"
    panel_path = class_dir / f"{stem}_panel.png"

    save_rgb(ig_path, ig_rgb)
    save_rgb(smoothgrad_path, sg_rgb)
    save_rgb(panel_path, make_panel(pil_img, ig_rgb, sg_rgb))

    return {
        "img_fName": image_path.name,
        "true_class": true_class,
        "pred_class": pred_class,
        "pred_prob": pred_prob,
        "correct": pred_class == true_class,
        "ig_delta": float(delta.detach().cpu().item()),
        "ig_path": str(ig_path),
        "smoothgrad_path": str(smoothgrad_path),
        "panel_path": str(panel_path),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint_path}\n"
            "Bring it from the classification branch with:\n"
            "git checkout fine-grained-classification/imbalance-methods -- "
            "\"results on new labels/final results - top methods from phase1&2 ver/weighted_loss_winverse_lr0.0005.pth\""
        )

    df = load_annotations(args.csv_path, args.image_dir)
    class_names = sorted(df["class_label"].astype(str).unique())

    if args.image_path is None:
        train_df, val_df, test_df = load_split(df, args.test_size, args.val_size, args.seed)
        sample_df = sample_xai_images(test_df, args.samples_per_class, args.seed)

        split_dir = args.out_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(split_dir / "train_split.csv", index=False)
        val_df.to_csv(split_dir / "val_split.csv", index=False)
        test_df.to_csv(split_dir / "test_split.csv", index=False)
        sample_df.to_csv(split_dir / "xai_sample_images.csv", index=False)

        print("Train images:", len(train_df))
        print("Val images:", len(val_df))
        print("Test images:", len(test_df))
        print("xAI images:", len(sample_df))
    else:
        sample_df = single_image_df(df, args.image_path)
        print("xAI images:", len(sample_df))

    device = get_device()
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

    rows = []
    for _, row in sample_df.iterrows():
        rows.append(
            explain_image(
                model=model,
                image_path=Path(row["image_path"]),
                true_class=str(row["class_label"]),
                class_names=class_names,
                preprocess=preprocess,
                device=device,
                out_dir=args.out_dir,
                smoothgrad_samples=args.smoothgrad_samples,
                smoothgrad_stdev=args.smoothgrad_stdev,
            )
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "xai_summary.csv", index=False)

    print("Saved xAI outputs to:", args.out_dir)
    print(summary[["img_fName", "true_class", "pred_class", "pred_prob", "correct", "panel_path"]])


if __name__ == "__main__":
    main()
