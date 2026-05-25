"""Shared detection overlap and pooled micro/macro P/R/F1 helpers."""

from __future__ import annotations

import numpy as np


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
    """For each GT box, best IoU to any prediction, then average."""
    if len(gt_xyxy) == 0:
        return float("nan")
    ious = iou_xyxy_matrix(pred_xyxy, gt_xyxy)
    if ious.size == 0:
        return 0.0
    return float(np.mean(np.max(ious, axis=0)))


def max_pairwise_iou_predictions(pred_xyxy: np.ndarray) -> float:
    """Largest IoU between two distinct predicted boxes."""
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
    """Greedy one-to-one matching by descending IoU. Returns (tp, fp, fn)."""
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


def _filter_boxes_by_class(
    xyxy: np.ndarray,
    cls: np.ndarray,
    class_id: int,
) -> np.ndarray:
    if len(xyxy) == 0 or len(cls) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    mask = cls == class_id
    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float64)
    return xyxy[mask]


def per_class_tp_fp_fn(
    pred_xyxy: np.ndarray,
    pred_cls: np.ndarray,
    gt_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    iou_threshold: float,
    class_ids: list[int],
) -> dict[int, tuple[int, int, int]]:
    """Class-filtered greedy IoU matching; one (tp, fp, fn) tuple per class."""
    return {
        class_id: greedy_match_tp_fp_fn(
            _filter_boxes_by_class(pred_xyxy, pred_cls, class_id),
            _filter_boxes_by_class(gt_xyxy, gt_cls, class_id),
            iou_threshold,
        )
        for class_id in class_ids
    }


def merge_class_counts(
    totals: dict[int, list[int]],
    per_image: dict[int, tuple[int, int, int]],
) -> None:
    """Add per-image class counts into running totals (mutates totals)."""
    for class_id, (tp, fp, fn) in per_image.items():
        bucket = totals[class_id]
        bucket[0] += tp
        bucket[1] += fp
        bucket[2] += fn


def macro_precision_recall_f1(
    class_counts: dict[int, list[int]],
    class_ids: list[int],
) -> tuple[float, float, float]:
    """Unweighted mean of per-class P/R/F1 (classes with no boxes score 0)."""
    if not class_ids:
        return 0.0, 0.0, 0.0
    precs: list[float] = []
    recs: list[float] = []
    f1s: list[float] = []
    for class_id in class_ids:
        tp, fp, fn = class_counts.get(class_id, [0, 0, 0])
        if tp + fp + fn == 0:
            precs.append(0.0)
            recs.append(0.0)
            f1s.append(0.0)
            continue
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
    return float(np.mean(precs)), float(np.mean(recs)), float(np.mean(f1s))


def init_class_counts(class_ids: list[int]) -> dict[int, list[int]]:
    return {class_id: [0, 0, 0] for class_id in class_ids}


def rankable_overlap_rows(
    overlap_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        row
        for row in overlap_rows
        if int(row["n_gt"]) > 0 and np.isfinite(float(row["mean_max_iou_gt"]))
    ]


def print_overlap_rank_lines(entries: list[dict[str, object]]) -> None:
    for rank, row in enumerate(entries, start=1):
        print(
            f"  {rank:2d}. overlap_mean_max_iou={float(row['mean_max_iou_gt']):.4f}  "
            f"P={float(row['precision']):.3f} R={float(row['recall']):.3f} F1={float(row['f1']):.3f}  "
            f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}  "
            f"n_gt={row['n_gt']} n_pred={row['n_pred']}  "
            f"max_pred_pair_iou={float(row['max_pred_pair_iou']):.4f}"
        )
        print(f"      {row['path']}")


def print_pooled_accuracy(
    *,
    n_images: int,
    match_iou: float,
    sum_tp: int,
    sum_fp: int,
    sum_fn: int,
    class_counts: dict[int, list[int]],
    class_ids: list[int],
) -> None:
    mic_p, mic_r, mic_f1 = precision_recall_f1(sum_tp, sum_fp, sum_fn)
    mac_p, mac_r, mac_f1 = macro_precision_recall_f1(class_counts, class_ids)
    print()
    print(
        f"Detection accuracy @IoU≥{match_iou:g} (greedy match, pooled over {n_images} images):"
    )
    print(f"  TP={sum_tp}  FP={sum_fp}  FN={sum_fn}")
    print(
        f"  micro precision: {mic_p:.4f}  micro recall: {mic_r:.4f}  micro F1: {mic_f1:.4f}"
    )
    print(
        f"  macro precision: {mac_p:.4f}  macro recall: {mac_r:.4f}  macro F1: {mac_f1:.4f}"
    )
