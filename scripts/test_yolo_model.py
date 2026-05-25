"""
Evaluate a trained Ultralytics YOLO model on the held-out test split.

Expects ``yolo_dataset/`` (or --yolo-dir) built by convert_to_yolo.py / train_yolo_model.py.

Reports COCO-style mAP (via Ultralytics) plus micro/macro precision/recall/F1 and overlap
stats aligned with test_mosquito_model.py (greedy IoU matching @ --match-iou).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import yaml
from PIL import Image

from convert_to_yolo import DEFAULT_OUT_DIR, DEFAULT_OUT_DIR_DET
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

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def candidate_weight_paths(project: str, name: str) -> list[str]:
    """Paths Ultralytics may write when project is ``runs/mosquito`` (→ ``runs/detect/runs/mosquito/``)."""
    stems = [
        os.path.join(project, name, "weights"),
        os.path.join("runs", "detect", project, name, "weights"),
        os.path.join("runs", "detect", name, "weights"),
    ]
    if project.startswith("runs/"):
        # e.g. project=runs/mosquito → also runs/detect/mosquito/name
        short = project.split("/", 1)[-1]
        stems.append(os.path.join("runs", "detect", short, name, "weights"))
    paths: list[str] = []
    for stem in stems:
        paths.append(os.path.join(stem, "best.pt"))
        paths.append(os.path.join(stem, "last.pt"))
    env = os.environ.get("YOLO_WEIGHTS", "").strip()
    if env:
        paths.insert(0, os.path.abspath(os.path.expanduser(env)))
    return paths


def default_weights_path(project: str, name: str) -> str | None:
    for p in candidate_weight_paths(project, name):
        if os.path.isfile(p):
            return p
    for p in candidate_weight_paths(project, name):
        stem = os.path.dirname(p)
        if os.path.isdir(stem):
            matches = sorted(glob.glob(os.path.join(stem, "*.pt")))
            if matches:
                return matches[-1]
    # Last resort: any matching run name under runs/detect (any repo cwd on shared FS)
    for pattern in (
        os.path.join("runs", "detect", "**", name, "weights", "best.pt"),
        os.path.join("runs", "detect", "**", name, "weights", "last.pt"),
    ):
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[-1]
    return None


def format_weights_search_hint(project: str, name: str) -> str:
    tried = candidate_weight_paths(project, name)
    lines = ["Searched for weights (first existing wins):"]
    lines.extend(f"  - {os.path.abspath(p)}" for p in tried[:8])
    if len(tried) > 8:
        lines.append(f"  - ... and {len(tried) - 8} more")
    lines.append(
        f"\nIf training finished elsewhere, pass the path from the train log, e.g.\n"
        f"  --weights /data/.../runs/detect/{project}/{name}/weights/best.pt"
    )
    if name.endswith("_det"):
        lines.append(
            "\nNote: multi-class training uses --name yolo_train; detection-only uses yolo_train_det."
        )
    return "\n".join(lines)


def class_ids_from_data_yaml(data_yaml: str) -> list[int]:
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    names = data.get("names") or {}
    if isinstance(names, dict):
        return sorted(int(k) for k in names.keys())
    if isinstance(names, list):
        return list(range(len(names)))
    return []


def class_names_from_model(model) -> dict[int, str]:
    raw = getattr(model, "names", None) or {}
    return {int(k): str(v) for k, v in raw.items()}


def result_to_pred(result) -> tuple[np.ndarray, np.ndarray]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=int)
    xyxy = np.asarray(boxes.xyxy.cpu().numpy(), dtype=np.float64)
    cls_ids = np.asarray(boxes.cls.cpu().numpy().astype(int), dtype=int)
    return xyxy, cls_ids


def list_test_images(test_images_dir: str) -> list[str]:
    paths = [
        os.path.join(test_images_dir, name)
        for name in sorted(os.listdir(test_images_dir))
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS
    ]
    return paths


def yolo_label_path_for_image(image_path: str, labels_dir: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(labels_dir, f"{stem}.txt")


def load_yolo_gt(label_path: str, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO normalized class xc yc w h → xyxy pixels and class ids."""
    if not os.path.isfile(label_path):
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=int)
    boxes: list[list[float]] = []
    classes: list[int] = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            xc, yc, wn, hn = map(float, parts[1:5])
            bw = wn * img_w
            bh = hn * img_h
            xtl = xc * img_w - bw / 2.0
            ytl = yc * img_h - bh / 2.0
            boxes.append([xtl, ytl, xtl + bw, ytl + bh])
            classes.append(class_id)
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64), np.zeros(0, dtype=int)
    return np.asarray(boxes, dtype=np.float64), np.asarray(classes, dtype=int)


def serialise_predictions(
    results,
    category_names: dict[int, str],
) -> list[dict[str, object]]:
    """One entry per image: path plus list of {xyxy, score, class_id, class_name}."""
    out: list[dict[str, object]] = []
    for result in results:
        dets: list[dict[str, object]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            for j in range(len(xyxy)):
                class_id = int(cls_ids[j])
                dets.append(
                    {
                        "xyxy": [
                            float(xyxy[j, 0]),
                            float(xyxy[j, 1]),
                            float(xyxy[j, 2]),
                            float(xyxy[j, 3]),
                        ],
                        "score": float(confs[j]),
                        "class_id": class_id,
                        "class_name": category_names.get(class_id, str(class_id)),
                    }
                )
        out.append({"image_path": result.path, "detections": dets})
    return out


def run_test_inference(
    model,
    test_images_dir: str,
    *,
    conf: float,
    imgsz: int,
) -> list:
    image_paths = list_test_images(test_images_dir)
    if not image_paths:
        sys.exit(f"No images found under {test_images_dir}")
    print(
        f"Running inference on {len(image_paths)} test images "
        f"(conf={conf}, imgsz={imgsz})..."
    )
    return list(
        model.predict(
            source=test_images_dir,
            conf=conf,
            imgsz=imgsz,
            stream=True,
            verbose=False,
        )
    )


def compute_overlap_metrics(
    results,
    test_labels_dir: str,
    match_iou: float,
    class_ids: list[int],
) -> tuple[list[dict[str, object]], dict[int, list[int]], int]:
    overlap_rows: list[dict[str, object]] = []
    class_counts = init_class_counts(class_ids)
    for result in results:
        image_path = str(result.path)
        with Image.open(image_path) as img:
            img_w, img_h = img.size
        label_path = yolo_label_path_for_image(image_path, test_labels_dir)
        gt_xy, gt_cls = load_yolo_gt(label_path, img_w, img_h)
        pr_xy, pr_cls = result_to_pred(result)
        tp, fp, fn = greedy_match_tp_fp_fn(pr_xy, gt_xy, match_iou)
        merge_class_counts(
            class_counts,
            per_class_tp_fp_fn(
                pr_xy, pr_cls, gt_xy, gt_cls, match_iou, class_ids
            ),
        )
        prec, rec, f1 = precision_recall_f1(tp, fp, fn)
        overlap_rows.append(
            {
                "path": image_path,
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
    return overlap_rows, class_counts, len(overlap_rows)


def print_overlap_summary(
    overlap_rows: list[dict[str, object]],
    *,
    match_iou: float,
    worst_overlap: int,
    best_overlap: int,
    class_counts: dict[int, list[int]],
    class_ids: list[int],
) -> None:
    n = len(overlap_rows)
    if n == 0:
        return

    sum_tp = sum(int(r["tp"]) for r in overlap_rows)
    sum_fp = sum(int(r["fp"]) for r in overlap_rows)
    sum_fn = sum(int(r["fn"]) for r in overlap_rows)
    print_pooled_accuracy(
        n_images=n,
        match_iou=match_iou,
        sum_tp=sum_tp,
        sum_fp=sum_fp,
        sum_fn=sum_fn,
        class_counts=class_counts,
        class_ids=class_ids,
    )

    rankable = rankable_overlap_rows(overlap_rows)

    if worst_overlap > 0:
        worst = sorted(rankable, key=lambda r: float(r["mean_max_iou_gt"]))
        k = min(worst_overlap, len(worst))
        print()
        print(
            f"Lowest pred-vs-GT overlap ({k} images): mean of (max IoU to any prediction) "
            "per ground-truth box — lower is worse localization vs labels. "
            f"P/R/F1 use IoU≥{match_iou:g} greedy matching."
        )
        print_overlap_rank_lines(worst[:k])

    if best_overlap > 0:
        best = sorted(rankable, key=lambda r: float(r["mean_max_iou_gt"]), reverse=True)
        k = min(best_overlap, len(best))
        print()
        print(
            f"Highest pred-vs-GT overlap ({k} images): same mean-max-IoU-per-GT score — "
            "higher is closer agreement between boxes and labels. "
            f"P/R/F1 use IoU≥{match_iou:g} greedy matching."
        )
        print_overlap_rank_lines(best[:k])


def save_test_predictions(payload: list[dict[str, object]], out_path: str) -> None:
    out_path = os.path.abspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    n_with_dets = sum(1 for entry in payload if entry["detections"])
    print(
        f"Saved predictions ({len(payload)} images, "
        f"{n_with_dets} with >=1 detection) to {out_path}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate YOLO on the test split.")
    p.add_argument(
        "--yolo-dir",
        default=None,
        help="Directory containing data.yaml (default: yolo_dataset or yolo_dataset_det)",
    )
    p.add_argument(
        "--detection-only",
        action="store_true",
        help="Use yolo_dataset_det/ and runs/mosquito/yolo_train_det weights by default.",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Trained .pt weights. If omitted, tries runs/mosquito/yolo_train/weights/best.pt",
    )
    p.add_argument(
        "--project",
        default=None,
        help="Ultralytics project (default: runs/mosquito)",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Run name (default: yolo_train or yolo_train_det with --detection-only)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Validation image size",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for validation and saved predictions",
    )
    p.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold for per-image precision/recall/F1 and pooled micro/macro accuracy "
        "(default 0.5, aligned with mAP@50).",
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
        "--save-predictions",
        default=None,
        metavar="PATH",
        help="Write per-image test predictions to JSON (default: test_yolo_predictions-end-to-end.json "
        "or test_yolo_predictions-detection-only.json)",
    )
    p.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Skip writing the predictions JSON",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics is not installed. Install with: pip install ultralytics")

    default_yolo = DEFAULT_OUT_DIR_DET if args.detection_only else DEFAULT_OUT_DIR
    yolo_dir = os.path.abspath(args.yolo_dir or default_yolo)
    project = args.project or "runs/mosquito"
    name = args.name or ("yolo_train_det" if args.detection_only else "yolo_train")
    save_predictions = args.save_predictions or (
        "test_yolo_predictions-detection-only.json"
        if args.detection_only
        else "test_yolo_predictions-end-to-end.json"
    )
    data_yaml = os.path.join(yolo_dir, "data.yaml")
    test_images_dir = os.path.join(yolo_dir, "images", "test")
    test_labels_dir = os.path.join(yolo_dir, "labels", "test")
    if not os.path.isfile(data_yaml):
        sys.exit(
            f"Missing {data_yaml}. Run train_yolo_model.py or convert_to_yolo.py first."
        )
    if not os.path.isdir(test_images_dir):
        sys.exit(f"Test images directory not found: {test_images_dir}")

    weights = args.weights or default_weights_path(project, name)
    if not weights or not os.path.isfile(weights):
        sys.exit(
            "No checkpoint found. Pass --weights path/to/best.pt "
            f"or train with --project {project} --name {name}.\n\n"
            + format_weights_search_hint(project, name)
            + f"\n\ncwd: {os.getcwd()}"
        )

    print(f"Loading weights: {weights}")
    model = YOLO(weights)

    print(f"Validating on test split ({data_yaml})...")
    metrics = model.val(
        data=data_yaml,
        split="test",
        imgsz=args.imgsz,
        conf=args.conf,
    )

    print("Test metrics (Ultralytics / COCO-style):")
    if metrics is not None:
        box = getattr(metrics, "box", None)
        if box is not None:
            map50_95 = float(getattr(box, "map", 0.0))
            map50 = float(getattr(box, "map50", 0.0))
            map75 = float(getattr(box, "map75", 0.0))
            print(f"  mAP @[.50:.95]: {map50_95:.4f}")
            print(f"  mAP @0.50:      {map50:.4f}")
            print(f"  mAP @0.75:      {map75:.4f}")
        else:
            print(f"  {metrics}")

    class_ids = class_ids_from_data_yaml(data_yaml)
    if not class_ids:
        class_ids = sorted(class_names_from_model(model).keys())

    results = run_test_inference(
        model,
        test_images_dir,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    overlap_rows, class_counts, _n = compute_overlap_metrics(
        results,
        test_labels_dir,
        args.match_iou,
        class_ids,
    )
    print_overlap_summary(
        overlap_rows,
        match_iou=args.match_iou,
        worst_overlap=args.worst_overlap,
        best_overlap=args.best_overlap,
        class_counts=class_counts,
        class_ids=class_ids,
    )

    if not args.no_save_predictions:
        category_names = class_names_from_model(model)
        payload = serialise_predictions(results, category_names)
        save_test_predictions(payload, save_predictions)


if __name__ == "__main__":
    main()
