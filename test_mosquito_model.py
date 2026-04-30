"""
Evaluate a trained RF-DETR model on the held-out test split (COCO format).

Expects the same layout as training: a folder containing images (or symlinks)
and `_annotations.coco.json`, e.g. `./rfdetr_dataset/test` after
`train_mosquito_model.py` has built `rfdetr_dataset`.

Usage:
  python3 test_mosquito_model.py --weights output/checkpoint_best_total.pth
  python3 test_mosquito_model.py --weights output/checkpoint_best_total.pth --max-images 200
  python3 test_mosquito_model.py --weights ... --max-side 1280   # lower GPU memory on big images
  python3 test_mosquito_model.py --weights ... --worst-overlap 0 --best-overlap 20   # only top 20 by overlap
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
        help="IoU threshold for per-image precision/recall/F1 and micro-averaged accuracy "
        "(default 0.5, aligned with mAP@50).",
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


def _xyxy_array(det: sv.Detections) -> np.ndarray:
    if det.xyxy is None or len(det.xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(det.xyxy, dtype=np.float64)


def iou_xyxy_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU, shape (len(a), len(b)). Either side may be length 0."""
    na, nb = len(boxes_a), len(boxes_b)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=np.float64)
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b.T
    bx1, bx2 = bx1.reshape(1, -1), bx2.reshape(1, -1)
    by1, by2 = by1.reshape(1, -1), by2.reshape(1, -1)
    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    iw = np.clip(inter_x2 - inter_x1, 0.0, None)
    ih = np.clip(inter_y2 - inter_y1, 0.0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0.0, None) * np.clip(ay2 - ay1, 0.0, None)
    area_b = np.clip(bx2 - bx1, 0.0, None) * np.clip(by2 - by1, 0.0, None)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def mean_max_iou_per_gt(gt_xyxy: np.ndarray, pred_xyxy: np.ndarray) -> float:
    """
    For each ground-truth box, take the best IoU to any prediction, then average.
    Low values mean predictions overlap the labeled mosquitoes poorly (worst localization).
    """
    if len(gt_xyxy) == 0:
        return float("nan")
    ious = iou_xyxy_matrix(pred_xyxy, gt_xyxy)  # (n_pred, n_gt)
    if ious.size == 0:
        return 0.0
    max_per_gt = np.max(ious, axis=0)
    return float(np.mean(max_per_gt))


def max_pairwise_iou_predictions(pred_xyxy: np.ndarray) -> float:
    """Largest IoU between two distinct predicted boxes (duplicate / crowded detections)."""
    n = len(pred_xyxy)
    if n < 2:
        return 0.0
    ious = iou_xyxy_matrix(pred_xyxy, pred_xyxy)
    np.fill_diagonal(ious, 0.0)
    return float(np.max(ious))


def greedy_match_tp_fp_fn(
    pred_xyxy: np.ndarray,
    gt_xyxy: np.ndarray,
    iou_threshold: float,
) -> tuple[int, int, int]:
    """
    Greedy one-to-one matching by descending IoU (same spirit as VOC/COCO @50).
    Returns (tp, fp, fn).
    """
    n_p, n_g = len(pred_xyxy), len(gt_xyxy)
    if n_g == 0:
        return 0, n_p, 0
    if n_p == 0:
        return 0, 0, n_g
    ious = iou_xyxy_matrix(pred_xyxy, gt_xyxy)
    pairs: list[tuple[float, int, int]] = []
    for pi in range(n_p):
        for gi in range(n_g):
            pairs.append((float(ious[pi, gi]), pi, gi))
    pairs.sort(key=lambda t: t[0], reverse=True)
    matched_p: set[int] = set()
    matched_g: set[int] = set()
    tp = 0
    for iou, pi, gi in pairs:
        if iou < iou_threshold:
            break
        if pi in matched_p or gi in matched_g:
            continue
        matched_p.add(pi)
        matched_g.add(gi)
        tp += 1
    fp = n_p - tp
    fn = n_g - tp
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Per-image or pooled counts → precision, recall, F1 in [0, 1]."""
    denom_p = tp + fp
    denom_r = tp + fn
    prec = float(tp / denom_p) if denom_p > 0 else (1.0 if tp + fp + fn == 0 else 0.0)
    rec = float(tp / denom_r) if denom_r > 0 else 1.0
    if prec + rec <= 0:
        f1 = 0.0
    else:
        f1 = 2.0 * prec * rec / (prec + rec)
    return prec, rec, f1


def _rankable_overlap_rows(
    overlap_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        r
        for r in overlap_rows
        if int(r["n_gt"]) > 0 and np.isfinite(float(r["mean_max_iou_gt"]))
    ]


def _print_overlap_rank_lines(entries: list[dict[str, object]]) -> None:
    for rank, r in enumerate(entries, start=1):
        print(
            f"  {rank:2d}. overlap_mean_max_iou={float(r['mean_max_iou_gt']):.4f}  "
            f"P={float(r['precision']):.3f} R={float(r['recall']):.3f} F1={float(r['f1']):.3f}  "
            f"TP/FP/FN={r['tp']}/{r['fp']}/{r['fn']}  "
            f"n_gt={r['n_gt']} n_pred={r['n_pred']}  max_pred_pair_iou={float(r['max_pred_pair_iou']):.4f}"
        )
        print(f"      {r['path']}")


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
    overlap_rows: list[dict[str, object]] = []
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
            gt_xy = _xyxy_array(target)
            pr_xy = _xyxy_array(pred)
            tp, fp, fn = greedy_match_tp_fp_fn(pr_xy, gt_xy, args.match_iou)
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
    mic_p, mic_r, mic_f1 = precision_recall_f1(sum_tp, sum_fp, sum_fn)
    print()
    print(
        f"Detection accuracy @IoU≥{args.match_iou:g} (greedy match, pooled over {n} images):"
    )
    print(f"  TP={sum_tp}  FP={sum_fp}  FN={sum_fn}")
    print(f"  micro precision: {mic_p:.4f}  micro recall: {mic_r:.4f}  micro F1: {mic_f1:.4f}")

    rankable = _rankable_overlap_rows(overlap_rows)

    if args.worst_overlap > 0:
        worst = sorted(rankable, key=lambda r: float(r["mean_max_iou_gt"]))
        k = min(args.worst_overlap, len(worst))
        print()
        print(
            f"Lowest pred-vs-GT overlap ({k} images): mean of (max IoU to any prediction) "
            "per ground-truth box — lower is worse localization vs labels. "
            f"P/R/F1 use IoU≥{args.match_iou:g} greedy matching."
        )
        _print_overlap_rank_lines(worst[:k])

    if args.best_overlap > 0:
        best = sorted(rankable, key=lambda r: float(r["mean_max_iou_gt"]), reverse=True)
        k = min(args.best_overlap, len(best))
        print()
        print(
            f"Highest pred-vs-GT overlap ({k} images): same mean-max-IoU-per-GT score — "
            "higher is closer agreement between boxes and labels. "
            f"P/R/F1 use IoU≥{args.match_iou:g} greedy matching."
        )
        _print_overlap_rank_lines(best[:k])

    if args.save_sample and sample_for_viz is not None:
        img, dets = sample_for_viz
        scene = img.copy()
        scene = sv.BoxAnnotator().annotate(scene=scene, detections=dets)
        scene = sv.LabelAnnotator().annotate(scene=scene, detections=dets)
        Image.fromarray(scene).save(args.save_sample)
        print(f"Saved sample visualization: {args.save_sample}")


if __name__ == "__main__":
    main()
