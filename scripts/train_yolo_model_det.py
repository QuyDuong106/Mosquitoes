"""
Train YOLO detection-only (single mosquito class).

Workflow:
  python3 scripts/export_yolo_detection_dataset.py
  python3 scripts/train_yolo_model_det.py --skip-dataset-export
"""

from __future__ import annotations

import sys


def main() -> None:
    if "--detection-only" not in sys.argv:
        sys.argv.insert(1, "--detection-only")
    from train_yolo_model import main as train_main

    train_main()


if __name__ == "__main__":
    main()
