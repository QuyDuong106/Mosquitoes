"""
Export detection-only YOLO datasets from manual_labels.csv.

Same images and train/val/test splits as multi-class; separate artifacts:
  <cache>/labels/*_coco_det.json
  ./yolo_dataset_det/

Does not modify train_coco.json, yolo_dataset/, or multi-class runs/.
"""

from __future__ import annotations

import argparse
import os

from convert_to_coco import convert_and_split_csv, resolve_dataset_layout
from convert_to_yolo import DEFAULT_OUT_DIR_DET, build_yolo_dataset


def export_yolo_detection_dataset(
    dataset_path: str | None = None,
    *,
    yolo_dir: str | None = None,
    skip_coco: bool = False,
) -> tuple[str, str]:
    if skip_coco:
        image_root, labels_dir = resolve_dataset_layout(dataset_path)
        print(f"Using existing detection-only COCO JSONs under {labels_dir}/")
    else:
        if dataset_path:
            print(f"Using dataset root from --dataset: {dataset_path}")
        elif os.environ.get("MOSQUITOES_DATASET", "").strip():
            print("Using dataset root from MOSQUITOES_DATASET.")
        else:
            print("No --dataset / MOSQUITOES_DATASET; fetching from Kaggle cache via kagglehub…")
        image_root = convert_and_split_csv(dataset_path, detection_only=True)
        _, labels_dir = resolve_dataset_layout(dataset_path)

    yolo_out = os.path.abspath(
        yolo_dir or os.path.join(os.getcwd(), DEFAULT_OUT_DIR_DET)
    )
    build_yolo_dataset(
        image_root,
        out_dir=yolo_out,
        labels_dir=labels_dir,
        detection_only=True,
    )
    print("Detection-only YOLO export complete:")
    print(f"  COCO labels: {labels_dir}/*_coco_det.json")
    print(f"  YOLO:        {yolo_out}/")
    return image_root, yolo_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export detection-only COCO + yolo_dataset_det/ from manual_labels.csv."
    )
    parser.add_argument("--dataset", metavar="PATH", default=None)
    parser.add_argument(
        "--yolo-dir",
        metavar="DIR",
        default=None,
        help=f"YOLO layout directory (default: ./{DEFAULT_OUT_DIR_DET}).",
    )
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="Skip COCO regeneration; require existing *_coco_det.json.",
    )
    args = parser.parse_args()
    dataset_arg = (
        os.path.abspath(os.path.expanduser(args.dataset)) if args.dataset else None
    )
    export_yolo_detection_dataset(
        dataset_arg,
        yolo_dir=args.yolo_dir,
        skip_coco=args.skip_coco,
    )


if __name__ == "__main__":
    main()
