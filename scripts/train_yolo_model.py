"""
Train an Ultralytics YOLO detector on the mosquito dataset (same splits as RF-DETR).

Pipeline: manual_labels.csv → COCO JSONs → yolo_dataset/ → YOLO train.
"""

from __future__ import annotations

import argparse
import os
import sys

from convert_to_coco import convert_and_split_csv, resolve_dataset_layout
from convert_to_yolo import DEFAULT_OUT_DIR, DEFAULT_OUT_DIR_DET, build_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO on the mosquito dataset (COCO splits → YOLO format)."
    )
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        default=None,
        help="Dataset root (images). Default: MOSQUITOES_DATASET or Kaggle download.",
    )
    parser.add_argument(
        "--yolo-dir",
        metavar="DIR",
        default=None,
        help=f"YOLO dataset directory (default: ./{DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Ultralytics checkpoint or architecture (e.g. yolo11n.pt, yolo11s.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without val improvement).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (square).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size (-1 = Ultralytics auto batch).",
    )
    parser.add_argument(
        "--project",
        default="runs/mosquito",
        help="Ultralytics project directory for runs.",
    )
    parser.add_argument(
        "--name",
        default="yolo_train",
        help="Run name under --project.",
    )
    parser.add_argument(
        "--no-regenerate-coco",
        action="store_true",
        help="Skip COCO regeneration; require existing labels/*_coco.json.",
    )
    parser.add_argument(
        "--skip-yolo-export",
        action="store_true",
        help="Skip rebuilding yolo_dataset/ (use existing data.yaml).",
    )
    parser.add_argument(
        "--skip-dataset-export",
        action="store_true",
        help="Skip COCO conversion and yolo_dataset(_det)/ rebuild; use existing export "
        "(run export_yolo_detection_dataset.py first for detection-only).",
    )
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Single-class training from the same manual_labels.csv and splits as multi-class: "
        "*_coco_det.json, yolo_dataset_det/, runs/mosquito/yolo_train_det by default. "
        "Does not overwrite multi-class artifacts.",
    )
    args = parser.parse_args()
    if args.skip_dataset_export:
        args.no_regenerate_coco = True
        args.skip_yolo_export = True

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is not installed. Install with:\n"
            "  pip install ultralytics"
        )

    dataset_arg = os.path.abspath(os.path.expanduser(args.dataset)) if args.dataset else None

    if args.no_regenerate_coco:
        image_root, labels_dir = resolve_dataset_layout(dataset_arg)
        source_path = image_root
        print(f"Using existing COCO JSONs under {labels_dir}/")
    else:
        if dataset_arg:
            print(f"Using dataset root from --dataset: {dataset_arg}")
        elif os.environ.get("MOSQUITOES_DATASET", "").strip():
            print("Using dataset root from MOSQUITOES_DATASET.")
        else:
            print("No --dataset / MOSQUITOES_DATASET; fetching via kagglehub…")
        source_path = convert_and_split_csv(
            dataset_arg, detection_only=args.detection_only
        )
        _, labels_dir = resolve_dataset_layout(dataset_arg)

    default_yolo = DEFAULT_OUT_DIR_DET if args.detection_only else DEFAULT_OUT_DIR
    yolo_dir = os.path.abspath(args.yolo_dir or os.path.join(os.getcwd(), default_yolo))
    if args.detection_only and args.project == "runs/mosquito" and args.name == "yolo_train":
        args.name = "yolo_train_det"
        print(f"Detection-only: using run name {args.name!r} (override with --name).")
    data_yaml = os.path.join(yolo_dir, "data.yaml")

    if args.skip_yolo_export:
        if not os.path.isfile(data_yaml):
            sys.exit(
                f"Missing {data_yaml}. Run export_yolo_detection_dataset.py or convert_to_yolo.py "
                "first, or drop --skip-yolo-export / --skip-dataset-export."
            )
        print(f"Using existing YOLO dataset: {yolo_dir}")
    else:
        build_yolo_dataset(
            source_path,
            out_dir=yolo_dir,
            labels_dir=labels_dir,
            detection_only=args.detection_only,
        )

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    print("Starting YOLO training...")
    train_kw: dict = {
        "data": data_yaml,
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": args.name,
    }
    if args.batch >= 0:
        train_kw["batch"] = args.batch

    results = model.train(**train_kw)
    print("Training finished.")
    save_dir = getattr(results, "save_dir", None) if results is not None else None
    if save_dir:
        print(f"Run directory: {save_dir}")
    best = None
    if save_dir:
        candidate = os.path.join(save_dir, "weights", "best.pt")
        if os.path.isfile(candidate):
            best = candidate
    if best is None:
        direct = os.path.join(args.project, args.name, "weights", "best.pt")
        under_detect = os.path.join(
            "runs", "detect", args.project, args.name, "weights", "best.pt"
        )
        for candidate in (direct, under_detect):
            if os.path.isfile(candidate):
                best = candidate
                break
    if best and os.path.isfile(best):
        print(f"Best weights: {os.path.abspath(best)}")
    else:
        print(
            "Could not locate best.pt automatically. After training, evaluate with:\n"
            f"  python3 scripts/test_yolo_model.py --weights <path/to/best.pt> "
            f'--name {args.name}'
            + (" --detection-only" if args.detection_only else "")
        )


if __name__ == "__main__":
    main()
