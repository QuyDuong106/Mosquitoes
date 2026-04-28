"""
Evaluate a trained RF-DETR model on the held-out test split (COCO format).

Expects the same layout as training: a folder containing images (or symlinks)
and `_annotations.coco.json`, e.g. `./rfdetr_dataset/test` after
`train_mosquito_model.py` has built `rfdetr_dataset`.

Usage:
  python3 test_mosquito_model.py --weights output/checkpoint_best_total.pth
  python3 test_mosquito_model.py --weights output/checkpoint_best_total.pth --max-images 200
  python3 test_mosquito_model.py --weights ... --max-side 1280   # lower GPU memory on big images
"""

from __future__ import annotations

import argparse
import gc
import glob
import itertools
import os
import sys

import numpy as np
import supervision as sv
import torch
from PIL import Image

try:
    from supervision.metrics import MeanAveragePrecision
except ImportError:
    try:
        from supervision.metrics.mean_average_precision import MeanAveragePrecision
    except ImportError:  # older supervision
        MeanAveragePrecision = sv.MeanAveragePrecision  # type: ignore[misc,assignment]

from rfdetr import RFDETRSmall


def default_weights_path() -> str | None:
    patterns = [
        os.path.join("output", "checkpoint_best_total.pth"),
        os.path.join("output", "checkpoint_best_ema.pth"),
        os.path.join("output", "checkpoint.pth"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p
    matches = sorted(glob.glob(os.path.join("output", "checkpoint*.pth")))
    return matches[-1] if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RF-DETR on test COCO split.")
    p.add_argument(
        "--test-dir",
        default=os.path.join("rfdetr_dataset", "test"),
        help="Directory with test images and _annotations.coco.json",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Trained checkpoint (.pth). If omitted, tries common files under output/",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="If set, only evaluate on this many images (faster smoke test)",
    )
    p.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip model.optimize_for_inference() (default: run it on CUDA to save memory/time)",
    )
    p.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="If set, shrink so max(h,w)<=this before inference, then map boxes back (saves GPU RAM)",
    )
    p.add_argument(
        "--clear-cache-every",
        type=int,
        default=25,
        help="Every N images run torch.cuda.empty_cache() + gc (0 to disable)",
    )
    p.add_argument(
        "--save-sample",
        default=None,
        help="If set, save one annotated test image to this path (e.g. test_sample.jpg)",
    )
    return p.parse_args()


def load_model(weights: str, optimize: bool) -> RFDETRSmall:
    try:
        model = RFDETRSmall(pretrain_weights=weights, num_classes=1)
    except TypeError:
        model = RFDETRSmall(pretrain_weights=weights)
    if optimize and torch.cuda.is_available() and hasattr(model, "optimize_for_inference"):
        model.optimize_for_inference()
    return model


def maybe_resize_image(
    image: np.ndarray, max_side: int | None
) -> tuple[np.ndarray, float]:
    """Return (possibly resized RGB uint8 image, scale_up) where scale_up maps pred coords to original."""
    if max_side is None:
        return image, 1.0
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(image).resize((new_w, new_h), Image.Resampling.BILINEAR)
    out = np.asarray(pil)
    return out, 1.0 / scale


def scale_detections_xyxy(det: sv.Detections, scale_up: float) -> sv.Detections:
    if scale_up == 1.0 or det.xyxy is None or len(det.xyxy) == 0:
        return det
    scaled_xyxy = (det.xyxy.astype(np.float64) * scale_up).astype(np.float32)
    return sv.Detections(
        xyxy=scaled_xyxy,
        mask=det.mask,
        confidence=det.confidence,
        class_id=det.class_id,
        tracker_id=det.tracker_id,
        metadata=det.metadata,
    )


def main() -> None:
    args = parse_args()
    ann_path = os.path.join(args.test_dir, "_annotations.coco.json")
    if not os.path.isdir(args.test_dir):
        sys.exit(f"Test directory not found: {args.test_dir}")
    if not os.path.isfile(ann_path):
        sys.exit(
            f"Missing {ann_path}. Build rfdetr_dataset first (run training setup) "
            "or point --test-dir to your test folder."
        )

    weights = args.weights or default_weights_path()
    if not weights or not os.path.isfile(weights):
        sys.exit(
            "No checkpoint found. Pass --weights path/to.pth "
            "(e.g. output/checkpoint_best_total.pth)."
        )

    optimize = not args.no_optimize
    print(f"Loading weights: {weights}")
    model = load_model(weights, optimize)
    if args.max_side:
        print(f"Inference resize: max side {args.max_side}px (boxes scaled back for mAP)")

    print(f"Loading COCO test data from: {args.test_dir}")
    try:
        dataset = sv.DetectionDataset.from_coco(
            images_directory_path=args.test_dir,
            annotations_path=ann_path,
        )
    except AttributeError:
        sys.exit(
            "This supervision version lacks DetectionDataset.from_coco. "
            "Upgrade supervision (pip install -U supervision)."
        )

    n_total = len(dataset)
    if n_total == 0:
        sys.exit("Test dataset is empty.")

    iterator = iter(dataset)
    if args.max_images is not None:
        iterator = itertools.islice(iterator, min(args.max_images, n_total))

    predictions: list[sv.Detections] = []
    targets: list[sv.Detections] = []
    sample_for_viz: tuple[np.ndarray, sv.Detections] | None = None

    with torch.inference_mode():
        for i, sample in enumerate(iterator, start=1):
            _path, image, target = sample
            infer_img, scale_up = maybe_resize_image(np.asarray(image), args.max_side)
            pred = model.predict(infer_img, threshold=args.threshold)
            if not isinstance(pred, sv.Detections):
                sys.exit(
                    f"Unexpected predict() return type {type(pred)}; expected sv.Detections."
                )
            pred = scale_detections_xyxy(pred, scale_up)
            predictions.append(pred)
            targets.append(target)
            if args.save_sample and sample_for_viz is None:
                sample_for_viz = (np.asarray(image), pred)

            if (
                args.clear_cache_every > 0
                and i % args.clear_cache_every == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()
                gc.collect()

    n = len(predictions)
    if n == 0:
        sys.exit("No images evaluated (check --max-images).")

    print(f"Evaluated {n} images (dataset size {n_total}). Computing mAP…")
    # Supervision >=0.26: Metric API (update + compute). Older: from_detections classmethod.
    if hasattr(MeanAveragePrecision, "from_detections"):
        mAP = MeanAveragePrecision.from_detections(
            predictions=predictions,
            targets=targets,
        )
    else:
        mAP = MeanAveragePrecision().update(predictions, targets).compute()

    map50_95 = float(getattr(mAP, "map50_95", getattr(mAP, "mAP", 0.0)))
    map50 = float(getattr(mAP, "map50", 0.0))
    map75 = float(getattr(mAP, "map75", 0.0))

    print("Test metrics (supervision / COCO-style mAP):")
    print(f"  mAP @[.50:.95]: {map50_95:.4f}")
    print(f"  mAP @0.50:      {map50:.4f}")
    print(f"  mAP @0.75:      {map75:.4f}")

    if args.save_sample and sample_for_viz is not None:
        img, dets = sample_for_viz
        scene = img.copy()
        scene = sv.BoxAnnotator().annotate(scene=scene, detections=dets)
        scene = sv.LabelAnnotator().annotate(scene=scene, detections=dets)
        Image.fromarray(scene).save(args.save_sample)
        print(f"Saved sample visualization: {args.save_sample}")


if __name__ == "__main__":
    main()
