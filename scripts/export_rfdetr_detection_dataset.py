"""
Export detection-only RF-DETR datasets from manual_labels.csv.

Same images and train/val/test splits as multi-class; separate artifacts:
  <cache>/labels/*_coco_det.json
  ./rfdetr_dataset_det/

Does not modify train_coco.json, rfdetr_dataset/, or output/.
"""

from __future__ import annotations

import argparse
import os

from convert_to_coco import (
    RFDETR_DATASET_DIR_DET,
    convert_and_split_csv,
    resolve_dataset_layout,
)
from train_rfdetr_model import create_roboflow_structure


def export_rfdetr_detection_dataset(
    dataset_path: str | None = None,
    *,
    rf_dataset_dir: str | None = None,
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

    rf_out = os.path.abspath(
        rf_dataset_dir or os.path.join(os.getcwd(), RFDETR_DATASET_DIR_DET)
    )
    create_roboflow_structure(
        image_root,
        labels_dir=labels_dir,
        target_dir=rf_out,
        detection_only=True,
    )
    print("Detection-only RF-DETR export complete:")
    print(f"  COCO labels: {labels_dir}/*_coco_det.json")
    print(f"  RF-DETR:     {rf_out}/")
    return image_root, rf_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export detection-only COCO + rfdetr_dataset_det/ from manual_labels.csv."
    )
    parser.add_argument("--dataset", metavar="PATH", default=None)
    parser.add_argument(
        "--rf-dataset-dir",
        metavar="DIR",
        default=None,
        help=f"Symlink tree (default: ./{RFDETR_DATASET_DIR_DET}).",
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
    export_rfdetr_detection_dataset(
        dataset_arg,
        rf_dataset_dir=args.rf_dataset_dir,
        skip_coco=args.skip_coco,
    )


if __name__ == "__main__":
    main()
