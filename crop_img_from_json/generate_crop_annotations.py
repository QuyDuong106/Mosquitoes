#!/usr/bin/env python3
"""
Generate cropped_annotations.csv for cropped_rfdetr/ and cropped_yolo/
"""

import json
import csv
from pathlib import Path
from PIL import Image
from collections import defaultdict, Counter

# ── Configuration ─────────────────────────────────────────────────────────────

ORIGINAL_ANNOTATIONS = Path("labels/annotations.csv")   # adjust if needed

RFDETR_JSON     = Path("test_rf-detr_predictions-detection-only.json")
YOLO_JSON       = Path("test_yolo_predictions-detection-only.json")
RFDETR_CROP_DIR = Path("cropped_rfdetr")
YOLO_CROP_DIR   = Path("cropped_yolo")

# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = ["img_fName", "img_w", "img_h",
               "bbx_xtl", "bbx_ytl", "bbx_xbr", "bbx_ybr",
               "class_label", "source_image"]


def load_label_map(annotations_csv: Path) -> dict:
    label_map = defaultdict(list)
    with open(annotations_csv, newline="", encoding="utf-8-sig") as f:
        sample = f.read(2048)
        f.seek(0)
        tab_count   = sample.count("\t")
        comma_count = sample.count(",")
        delimiter   = "\t" if tab_count > comma_count else ","
        print(f"  Detected delimiter: {'TAB' if delimiter == chr(9) else 'COMMA'}")

        reader = csv.DictReader(f, delimiter=delimiter)
        print(f"  Columns found: {reader.fieldnames}")

        for row in reader:
            fname = (row.get("img_fName") or "").strip()
            label = (row.get("class_label") or "").strip()
            if fname and label:
                stem = Path(fname).stem.replace("_crop", "")
                label_map[stem].append(label)

    return {stem: Counter(labels).most_common(1)[0][0]
            for stem, labels in label_map.items()}


def find_annotations() -> Path:
    candidates = [
        ORIGINAL_ANNOTATIONS,
        Path("labels Youmin/annotations.csv"),
        Path("annotations_fixed.csv"),
        Path("labels/annotations_fixed.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def process_json(json_path, crop_dir, label_map, out_csv):
    if not json_path.exists():
        print(f"[WARN] JSON not found: {json_path} — skipping")
        return

    with open(json_path) as f:
        records = json.load(f)

    rows = []
    missing_labels = 0
    missing_crops  = 0

    for record in records:
        raw_path   = record.get("image_path", "")
        detections = record.get("detections", [])
        source_img = Path(raw_path).name
        stem       = Path(source_img).stem

        class_label = label_map.get(stem)
        if class_label is None:
            missing_labels += 1
            continue

        for idx, det in enumerate(detections):
            xyxy  = det.get("xyxy", [])
            score = det.get("score", 0)
            cname = det.get("class_name", "mosquito")

            if len(xyxy) != 4:
                continue

            crop_fname = f"{stem}_det{idx:02d}_{cname}_score{score:.3f}.jpeg"
            crop_path  = crop_dir / crop_fname

            if not crop_path.exists():
                missing_crops += 1
                continue

            with Image.open(crop_path) as img:
                w, h = img.size

            rows.append({
                "img_fName":    crop_fname,
                "img_w":        w,
                "img_h":        h,
                "bbx_xtl":      0,
                "bbx_ytl":      0,
                "bbx_xbr":      w,
                "bbx_ybr":      h,
                "class_label":  class_label,
                "source_image": source_img,
            })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, delimiter=",")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Rows written      : {len(rows)}")
    print(f"  Missing labels    : {missing_labels}")
    print(f"  Missing crop files: {missing_crops}")
    print(f"  Output            : {out_csv.resolve()}")
    dist = Counter(r["class_label"] for r in rows)
    print("  Class distribution:")
    for cls, cnt in dist.most_common():
        print(f"    {cls:<25} {cnt}")


def main():
    print("=" * 60)
    print("Generate Crop Annotations CSVs")
    print("=" * 60)

    annotations_path = find_annotations()
    if annotations_path is None:
        print(f"[ERROR] Could not find original annotations CSV.")
        print("  Edit ORIGINAL_ANNOTATIONS at the top of this script.")
        return

    print(f"Loading labels from: {annotations_path}")
    label_map = load_label_map(annotations_path)
    print(f"  Loaded {len(label_map)} image labels")
    sample = list(label_map.items())[:3]
    print(f"  Sample: {sample}\n")

    print("── RF-DETR ─────────────────────────────────────────────")
    process_json(RFDETR_JSON, RFDETR_CROP_DIR, label_map,
                 RFDETR_CROP_DIR / "cropped_annotations.csv")

    print("\n── YOLO ────────────────────────────────────────────────")
    process_json(YOLO_JSON, YOLO_CROP_DIR, label_map,
                 YOLO_CROP_DIR / "cropped_annotations.csv")

    print("\nDone!")


if __name__ == "__main__":
    main()
