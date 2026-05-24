"""
Evaluate a trained Ultralytics YOLO model on the held-out test split.

Expects ``yolo_dataset/`` (or --yolo-dir) built by convert_to_yolo.py / train_yolo_model.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def default_weights_path(project: str, name: str) -> str | None:
    patterns = [
        os.path.join(project, name, "weights", "best.pt"),
        os.path.join(project, name, "weights", "last.pt"),
        os.path.join("runs", "detect", project, name, "weights", "best.pt"),
        os.path.join("runs", "detect", project, name, "weights", "last.pt"),
    ]
    for p in patterns:
        if os.path.isfile(p):
            return p
    for pattern in (
        os.path.join(project, name, "weights", "*.pt"),
        os.path.join("runs", "detect", project, name, "weights", "*.pt"),
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def class_names_from_model(model) -> dict[int, str]:
    raw = getattr(model, "names", None) or {}
    return {int(k): str(v) for k, v in raw.items()}


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


def save_test_predictions(
    model,
    test_images_dir: str,
    out_path: str,
    *,
    conf: float,
    imgsz: int,
) -> None:
    if not os.path.isdir(test_images_dir):
        sys.exit(f"Test images directory not found: {test_images_dir}")

    n_images = sum(
        1
        for name in os.listdir(test_images_dir)
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS
    )
    if n_images == 0:
        sys.exit(f"No images found under {test_images_dir}")

    category_names = class_names_from_model(model)
    print(
        f"Running inference on {n_images} test images "
        f"(conf={conf}, imgsz={imgsz})..."
    )
    results = model.predict(
        source=test_images_dir,
        conf=conf,
        imgsz=imgsz,
        stream=True,
        verbose=False,
    )
    payload = serialise_predictions(results, category_names)

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
        default="yolo_dataset",
        help="Directory containing data.yaml and images/labels splits",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Trained .pt weights. If omitted, tries runs/mosquito/yolo_train/weights/best.pt",
    )
    p.add_argument(
        "--project",
        default="runs/mosquito",
        help="Ultralytics project (used with --name when --weights omitted)",
    )
    p.add_argument(
        "--name",
        default="yolo_train",
        help="Run name (used when --weights omitted)",
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
        "--save-predictions",
        default="test_yolo_predictions.json",
        metavar="PATH",
        help="Write per-image test predictions to JSON (same fields as test_rfdetr_model.py)",
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

    yolo_dir = os.path.abspath(args.yolo_dir)
    data_yaml = os.path.join(yolo_dir, "data.yaml")
    test_images_dir = os.path.join(yolo_dir, "images", "test")
    if not os.path.isfile(data_yaml):
        sys.exit(
            f"Missing {data_yaml}. Run train_yolo_model.py or convert_to_yolo.py first."
        )

    weights = args.weights or default_weights_path(args.project, args.name)
    if not weights or not os.path.isfile(weights):
        sys.exit(
            "No checkpoint found. Pass --weights path/to/best.pt "
            f"or train with --project {args.project} --name {args.name}."
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

    if not args.no_save_predictions:
        save_test_predictions(
            model,
            test_images_dir,
            args.save_predictions,
            conf=args.conf,
            imgsz=args.imgsz,
        )


if __name__ == "__main__":
    main()
