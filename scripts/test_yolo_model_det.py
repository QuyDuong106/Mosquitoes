"""Evaluate detection-only YOLO (yolo_dataset_det/, runs/mosquito/yolo_train_det/)."""

from __future__ import annotations

import sys


def main() -> None:
    if "--detection-only" not in sys.argv:
        sys.argv.insert(1, "--detection-only")
    from test_yolo_model import main as test_main

    test_main()


if __name__ == "__main__":
    main()
