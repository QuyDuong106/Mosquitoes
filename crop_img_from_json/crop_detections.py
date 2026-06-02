#!/usr/bin/env python3
"""
Crop images using bounding box predictions from RF-DETR and YOLO JSON files.

Expected JSON format:
[
  {
    "image_path": "...path/to/image.jpeg",
    "detections": [
      {
        "xyxy": [x1, y1, x2, y2],
        "score": 0.95,
        "class_id": 0,
        "class_name": "mosquito"
      },
      ...
    ]
  },
  ...
]

Output structure:
  cropped_rfdetr/
    train_00000_det0_score0.95.jpeg
    train_00000_det1_score0.87.jpeg
    ...
  cropped_yolo/
    train_00000_det0_score0.91.jpeg
    ...
"""

import json
import os
import sys
from pathlib import Path
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────

# Folder containing the original uncropped images
IMAGES_DIR = Path("images/images")

# JSON prediction files (adjust paths if needed)
RFDETR_JSON = Path("test_rf-detr_predictions-detection-only.json")
YOLO_JSON   = Path("test_yolo_predictions-detection-only.json")

# Output directories
OUT_RFDETR  = Path("cropped_rfdetr")
OUT_YOLO    = Path("cropped_yolo")

# Optional: add a small padding (in pixels) around each crop
PADDING = 0
#0 PADDING for default

# Only save crops whose confidence score is at or above this threshold (0 = keep all)
SCORE_THRESHOLD = 0.0

# ─────────────────────────────────────────────────────────────────────────────


def find_image(image_path_in_json: str, images_dir: Path) -> Path | None:
    """
    Locate the actual image file on disk.
    The JSON path is usually an absolute path from a training machine, so we
    extract just the filename and look for it in IMAGES_DIR.
    """
    filename = Path(image_path_in_json).name

    # Direct hit
    candidate = images_dir / filename
    if candidate.exists():
        return candidate

    # Try common extensions in case the suffix differs
    stem = Path(filename).stem
    for ext in (".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"):
        candidate = images_dir / (stem + ext)
        if candidate.exists():
            return candidate

    return None


def crop_and_save(img: Image.Image, xyxy: list, padding: int,
                  out_path: Path) -> bool:
    """Crop the image to the bounding box (with optional padding) and save."""
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in xyxy]

    # Apply padding and clamp to image bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Skip degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return False

    crop = img.crop((x1, y1, x2, y2))
    crop.save(out_path)
    return True


def process_json(json_path: Path, output_dir: Path, label: str):
    """Read a prediction JSON file and crop every detection."""
    if not json_path.exists():
        print(f"[WARN] JSON file not found: {json_path} — skipping {label}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        records = json.load(f)

    total_crops = 0
    skipped_images = 0
    skipped_detections = 0

    for record in records:
        raw_path   = record.get("image_path", "")
        detections = record.get("detections", [])

        img_path = find_image(raw_path, IMAGES_DIR)
        if img_path is None:
            print(f"  [MISS] {Path(raw_path).name}")
            skipped_images += 1
            continue

        img = Image.open(img_path).convert("RGB")
        stem = img_path.stem  # e.g. "train_00000"

        for idx, det in enumerate(detections):
            score = det.get("score", 1.0)
            if score < SCORE_THRESHOLD:
                skipped_detections += 1
                continue

            xyxy       = det.get("xyxy", [])
            class_name = det.get("class_name", "obj")

            if len(xyxy) != 4:
                skipped_detections += 1
                continue

            out_name = f"{stem}_det{idx:02d}_{class_name}_score{score:.3f}.jpeg"
            out_path = output_dir / out_name

            saved = crop_and_save(img, xyxy, PADDING, out_path)
            if saved:
                total_crops += 1
            else:
                skipped_detections += 1

    print(f"\n[{label}] Done.")
    print(f"  Crops saved    : {total_crops}")
    print(f"  Images missing : {skipped_images}")
    print(f"  Dets skipped   : {skipped_detections}")
    print(f"  Output folder  : {output_dir.resolve()}")


def main():
    print("=" * 60)
    print("Crop Detections Script")
    print("=" * 60)
    print(f"Images dir   : {IMAGES_DIR.resolve()}")
    print(f"RF-DETR JSON : {RFDETR_JSON}")
    print(f"YOLO JSON    : {YOLO_JSON}")
    print(f"Padding      : {PADDING}px")
    print(f"Score thresh : {SCORE_THRESHOLD}")
    print()

    if not IMAGES_DIR.exists():
        print(f"[ERROR] Images directory not found: {IMAGES_DIR.resolve()}")
        print("  Edit IMAGES_DIR in this script to point to your images folder.")
        sys.exit(1)

    # Install Pillow if needed
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Pillow not found — installing...")
        os.system("pip install Pillow --break-system-packages -q")

    print("── RF-DETR crops ──────────────────────────────────────")
    process_json(RFDETR_JSON, OUT_RFDETR, "RF-DETR")

    print()
    print("── YOLO crops ─────────────────────────────────────────")
    process_json(YOLO_JSON, OUT_YOLO, "YOLO")

    print()
    print("All done!")


if __name__ == "__main__":
    main()