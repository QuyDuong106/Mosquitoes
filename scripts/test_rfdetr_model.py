"""
Evaluate a trained RF-DETR model on the held-out test split (COCO format).

Expects the same layout as training: a folder containing images (or symlinks)
and `_annotations.coco.json`, e.g. `./rfdetr_dataset/test` after
`train_rfdetr_model.py` has built `rfdetr_dataset`.

Usage:
  python3 scripts/test_rfdetr_model.py --weights output/checkpoint_best_total.pth
  python3 scripts/test_rfdetr_model.py --weights output/checkpoint_best_total.pth --max-images 200
  python3 test_mosquito_model.py --weights ... --max-side 1280   # lower GPU memory on big images
  python3 test_mosquito_model.py --weights ... --worst-overlap 0 --best-overlap 20   # only top 20 by overlap
  python3 test_rfdetr_model.py --weights ... --save-predictions test_predictions-end-to-end.json
"""

from __future__ import annotations

import argparse
import gc
import glob
import itertools
import json
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

from detection_metrics import (
    greedy_match_tp_fp_fn,
    init_class_counts,
    mean_max_iou_per_gt,
    max_pairwise_iou_predictions,
    merge_class_counts,
    per_class_tp_fp_fn,
    precision_recall_f1,
    print_overlap_rank_lines,
    print_pooled_accuracy,
    rankable_overlap_rows,
)
from convert_to_coco import RFDETR_DATASET_DIR, RFDETR_DATASET_DIR_DET, RFDETR_OUTPUT_DIR, RFDETR_OUTPUT_DIR_DET
from rfdetr import RFDETRSmall


def default_weights_path(*, detection_only: bool = False) -> str | None:
    output_dir = RFDETR_OUTPUT_DIR_DET if detection_only else RFDETR_OUTPUT_DIR
    patterns = [
        os.path.join(output_dir, "checkpoint_best_total.pth"),
        os.path.join(output_dir, "checkpoint_best_ema.pth"),
        os.path.join(output_dir, "checkpoint.pth"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p
    matches = sorted(glob.glob(os.path.join(output_dir, "checkpoint*.pth")))
    return matches[-1] if matches else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RF-DETR on test COCO split.")
    p.add_argument(
        "--test-dir",
        default=None,
        help="Directory with test images and _annotations.coco.json "
        "(default: rfdetr_dataset/test or rfdetr_dataset_det/test with --detection-only)",
    )
    p.add_argument(
        "--detection-only",
        action="store_true",
        help="Use rfdetr_dataset_det/test and output_det/ checkpoints from detection-only training.",
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
    p.add_argument(
        "--worst-overlap",
        type=int,
        default=10,
        metavar="N",
        help="List N test images with lowest pred-vs-GT overlap (mean of max IoU per GT). "
        "0 disables.",
    )
    p.add_argument(
        "--best-overlap",
        type=int,
        default=10,
        metavar="N",
        help="List N test images with highest pred-vs-GT overlap (mean of max IoU per GT). "
        "0 disables.",
    )
    p.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for per-image precision/recall/F1 and pooled micro/macro accuracy "
        "(default 0.5, aligned with mAP@50).",
    )
    p.add_argument(
        "--save-predictions",
        default=None,
        metavar="PATH",
        help="After evaluation, write per-image predicted boxes (xyxy), scores, and class_ids "
        "to this JSON file (UTF-8).",
    )
    return p.parse_args()


def load_model(weights: str, optimize: bool, num_classes: int) -> RFDETRSmall:
    try:
        model = RFDETRSmall(pretrain_weights=weights, num_classes=num_classes)
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


def _xyxy_array(det: sv.Detections) -> np.ndarray:
    if det.xyxy is None or len(det.xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(det.xyxy, dtype=np.float64)


def _class_id_array(det: sv.Detections) -> np.ndarray:
    if det.class_id is None or len(det.class_id) == 0:
        return np.zeros(0, dtype=int)
    return np.asarray(det.class_id, dtype=int)


def serialise_predictions(
    image_paths: list[str],
    predictions: list[sv.Detections],
    category_names: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    """One entry per image: path plus list of {xyxy, score, class_id, class_name?}."""
    out: list[dict[str, object]] = []
    for path, pred in zip(image_paths, predictions, strict=True):
        pr_xy = _xyxy_array(pred)
        n = len(pr_xy)
        conf = pred.confidence
        cid = pred.class_id
        dets: list[dict[str, object]] = []
        for j in range(n):
            score_val: float | None
            if conf is not None and j < len(conf):
                score_val = float(conf[j])
            else:
                score_val = None
            class_val: int | None
            if cid is not None and j < len(cid):
                class_val = int(cid[j])
            else:
                class_val = None
            det_obj: dict[str, object] = {
                "xyxy": [
                    float(pr_xy[j, 0]),
                    float(pr_xy[j, 1]),
                    float(pr_xy[j, 2]),
                    float(pr_xy[j, 3]),
                ],
                "score": score_val,
                "class_id": class_val,
            }
            if category_names is not None and class_val is not None:
                det_obj["class_name"] = category_names.get(
                    class_val, str(class_val)
                )
            dets.append(det_obj)
        out.append({"image_path": path, "detections": dets})
    return out


def num_classes_and_category_names_from_coco(ann_path: str) -> tuple[int, dict[int, str]]:
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)
    cats = coco.get("categories") or [{"id": 0, "name": "object"}]
    names = {int(c["id"]): str(c["name"]) for c in cats}
    n = len(cats)
    return n, names


def main() -> None:
    args = parse_args()
    rf_root = RFDETR_DATASET_DIR_DET if args.detection_only else RFDETR_DATASET_DIR
    test_dir = args.test_dir or os.path.join(rf_root, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    if not os.path.isdir(test_dir):
        sys.exit(f"Test directory not found: {test_dir}")
    if not os.path.isfile(ann_path):
        sys.exit(
            f"Missing {ann_path}. Build {rf_root} first (run training setup) "
            "or point --test-dir to your test folder."
        )

    weights = args.weights or default_weights_path(detection_only=args.detection_only)
    if not weights or not os.path.isfile(weights):
        out_hint = RFDETR_OUTPUT_DIR_DET if args.detection_only else RFDETR_OUTPUT_DIR
        sys.exit(
            "No checkpoint found. Pass --weights path/to.pth "
            f"(e.g. {out_hint}/checkpoint_best_total.pth)."
        )

    num_classes, category_names = num_classes_and_category_names_from_coco(ann_path)
    class_ids = sorted(category_names.keys())
    class_counts = init_class_counts(class_ids)
    print(f"Inference head: num_classes={num_classes} (from COCO categories).")

    optimize = not args.no_optimize
    print(f"Loading weights: {weights}")
    model = load_model(weights, optimize, num_classes)
    if args.max_side:
        print(f"Inference resize: max side {args.max_side}px (boxes scaled back for mAP)")

    print(f"Loading COCO test data from: {test_dir}")
    try:
        dataset = sv.DetectionDataset.from_coco(
            images_directory_path=test_dir,
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
    image_paths: list[str] = []
    overlap_rows: list[dict[str, object]] = []
    sample_for_viz: tuple[np.ndarray, sv.Detections] | None = None

    with torch.inference_mode():
        for i, sample in enumerate(iterator, start=1):
            _path, image, target = sample
            image_paths.append(str(_path))
            infer_img, scale_up = maybe_resize_image(np.asarray(image), args.max_side)
            pred = model.predict(infer_img, threshold=args.threshold)
            if not isinstance(pred, sv.Detections):
                sys.exit(
                    f"Unexpected predict() return type {type(pred)}; expected sv.Detections."
                )
            pred = scale_detections_xyxy(pred, scale_up)
            predictions.append(pred)
            targets.append(target)
            gt_xy = _xyxy_array(target)
            pr_xy = _xyxy_array(pred)
            gt_cls = _class_id_array(target)
            pr_cls = _class_id_array(pred)
            tp, fp, fn = greedy_match_tp_fp_fn(pr_xy, gt_xy, args.match_iou)
            merge_class_counts(
                class_counts,
                per_class_tp_fp_fn(
                    pr_xy, pr_cls, gt_xy, gt_cls, args.match_iou, class_ids
                ),
            )
            prec, rec, f1 = precision_recall_f1(tp, fp, fn)
            overlap_rows.append(
                {
                    "path": _path,
                    "mean_max_iou_gt": mean_max_iou_per_gt(gt_xy, pr_xy),
                    "n_gt": int(len(gt_xy)),
                    "n_pred": int(len(pr_xy)),
                    "max_pred_pair_iou": max_pairwise_iou_predictions(pr_xy),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                }
            )
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

    sum_tp = sum(int(r["tp"]) for r in overlap_rows)
    sum_fp = sum(int(r["fp"]) for r in overlap_rows)
    sum_fn = sum(int(r["fn"]) for r in overlap_rows)
    print_pooled_accuracy(
        n_images=n,
        match_iou=args.match_iou,
        sum_tp=sum_tp,
        sum_fp=sum_fp,
        sum_fn=sum_fn,
        class_counts=class_counts,
        class_ids=class_ids,
    )

    if args.save_predictions:
        parent = os.path.dirname(os.path.abspath(args.save_predictions))
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = serialise_predictions(
            image_paths, predictions, category_names=category_names
        )
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved predictions ({len(payload)} images) to {args.save_predictions}")

    rankable = rankable_overlap_rows(overlap_rows)

    if args.worst_overlap > 0:
        worst = sorted(rankable, key=lambda r: float(r["mean_max_iou_gt"]))
        k = min(args.worst_overlap, len(worst))
        print()
        print(
            f"Lowest pred-vs-GT overlap ({k} images): mean of (max IoU to any prediction) "
            "per ground-truth box — lower is worse localization vs labels. "
            f"P/R/F1 use IoU≥{args.match_iou:g} greedy matching."
        )
        print_overlap_rank_lines(worst[:k])

    if args.best_overlap > 0:
        best = sorted(rankable, key=lambda r: float(r["mean_max_iou_gt"]), reverse=True)
        k = min(args.best_overlap, len(best))
        print()
        print(
            f"Highest pred-vs-GT overlap ({k} images): same mean-max-IoU-per-GT score — "
            "higher is closer agreement between boxes and labels. "
            f"P/R/F1 use IoU≥{args.match_iou:g} greedy matching."
        )
        print_overlap_rank_lines(best[:k])

    if args.save_sample and sample_for_viz is not None:
        img, dets = sample_for_viz
        scene = img.copy()
        scene = sv.BoxAnnotator().annotate(scene=scene, detections=dets)
        scene = sv.LabelAnnotator().annotate(scene=scene, detections=dets)
        Image.fromarray(scene).save(args.save_sample)
        print(f"Saved sample visualization: {args.save_sample}")


if __name__ == "__main__":
    main()
