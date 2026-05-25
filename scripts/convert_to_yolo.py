"""
Build an Ultralytics YOLO dataset from the same COCO splits as RF-DETR.

Writes ``yolo_dataset/`` under the current working directory:
  images/{train,val,test}/  (symlinks to source images)
  labels/{train,val,test}/  (one .txt per image, normalized xywh)
  data.yaml

Run after ``convert_and_split_csv()`` (or call ``build_yolo_dataset(..., regenerate_coco=True)``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict

from convert_to_coco import (
    coco_split_json_names,
    convert_and_split_csv,
    resolve_dataset_layout,
)
from dataset_images import build_image_index, resolve_image_path

DEFAULT_OUT_DIR = "yolo_dataset"
DEFAULT_OUT_DIR_DET = "yolo_dataset_det"


def coco_bbox_to_yolo_line(bbox: list[float], img_w: int, img_h: int, class_id: int) -> str:
    """COCO [xtl, ytl, w, h] pixels → YOLO class xc yc w h (normalized)."""
    xtl, ytl, bw, bh = bbox
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {img_w}x{img_h}")
    xc = (xtl + bw / 2.0) / img_w
    yc = (ytl + bh / 2.0) / img_h
    wn = bw / img_w
    hn = bh / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def write_data_yaml(out_dir: str, categories: list[dict]) -> str:
    """Write data.yaml; return absolute path."""
    sorted_cats = sorted(categories, key=lambda c: int(c["id"]))
    names_block = "\n".join(
        f"  {int(c['id'])}: {c['name']}" for c in sorted_cats
    )
    yaml_path = os.path.join(out_dir, "data.yaml")
    content = (
        f"path: {os.path.abspath(out_dir)}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"{names_block}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    return yaml_path


def export_split(
    coco_path: str,
    split_name: str,
    out_dir: str,
    image_index: dict[str, str],
) -> tuple[int, int]:
    """
    Symlink images and write label .txt files for one split.
    Returns (num_images, num_boxes).
    """
    images_dir = os.path.join(out_dir, "images", split_name)
    labels_dir = os.path.join(out_dir, "labels", split_name)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    with open(coco_path, encoding="utf-8") as f:
        coco = json.load(f)

    id_to_image: dict[int, dict] = {int(img["id"]): img for img in coco.get("images", [])}
    boxes_by_stem: dict[str, list[str]] = defaultdict(list)
    linked_names: set[str] = set()

    for ann in coco.get("annotations", []):
        image_id = int(ann["image_id"])
        image = id_to_image.get(image_id)
        if image is None:
            continue

        img_name = image["file_name"]
        src_img, canonical_name = resolve_image_path(img_name, image_index)
        if src_img is None:
            continue

        stem, _ = os.path.splitext(canonical_name)
        img_w = int(image["width"])
        img_h = int(image["height"])
        class_id = int(ann["category_id"])
        line = coco_bbox_to_yolo_line(ann["bbox"], img_w, img_h, class_id)
        boxes_by_stem[stem].append(line)

        dst_img = os.path.join(images_dir, canonical_name)
        if canonical_name not in linked_names:
            if not os.path.exists(dst_img):
                os.symlink(src_img, dst_img)
            linked_names.add(canonical_name)

    for stem, lines in boxes_by_stem.items():
        label_path = os.path.join(labels_dir, f"{stem}.txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    n_images = len(linked_names)
    n_boxes = sum(len(v) for v in boxes_by_stem.values())
    return n_images, n_boxes


def build_yolo_dataset(
    source_dataset_path: str,
    out_dir: str | None = None,
    *,
    labels_dir: str | None = None,
    wipe: bool = True,
    detection_only: bool = False,
) -> str:
    """
    Build ``yolo_dataset/`` from COCO JSONs under ``<source>/labels/``.

    :return: Absolute path to ``data.yaml``.
    """
    default_out = DEFAULT_OUT_DIR_DET if detection_only else DEFAULT_OUT_DIR
    out_dir = os.path.abspath(out_dir or os.path.join(os.getcwd(), default_out))
    labels_dir = labels_dir or os.path.join(source_dataset_path, "labels")
    split_json = coco_split_json_names(detection_only)

    if wipe and os.path.exists(out_dir):
        print(f"Removing existing YOLO dataset at {out_dir}...")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    image_index = build_image_index(source_dataset_path)
    if not image_index:
        raise RuntimeError(f"No images found under {source_dataset_path}")

    categories: list[dict] | None = None
    split_counts: dict[str, tuple[int, int]] = {}

    for split_name, json_name in split_json.items():
        coco_path = os.path.join(labels_dir, json_name)
        if not os.path.isfile(coco_path):
            raise FileNotFoundError(
                f"Missing {coco_path}. Run convert_to_coco.py first."
            )
        if categories is None:
            with open(coco_path, encoding="utf-8") as f:
                categories = json.load(f).get("categories", [])
        print(f" -> Exporting {split_name}...")
        n_img, n_box = export_split(coco_path, split_name, out_dir, image_index)
        split_counts[split_name] = (n_img, n_box)
        print(f"    {n_img} images, {n_box} boxes")

    empty = [s for s, (ni, _) in split_counts.items() if ni == 0]
    if empty:
        raise RuntimeError(
            f"Empty YOLO split(s): {', '.join(empty)}. Regenerate COCO splits."
        )

    yaml_path = write_data_yaml(out_dir, categories or [{"id": 0, "name": "mosquito"}])
    print(f"YOLO dataset ready: {out_dir}")
    print(f"data.yaml: {yaml_path}")
    if categories:
        print(
            "Classes: "
            + ", ".join(f"{c['id']}={c['name']}" for c in sorted(categories, key=lambda x: x["id"]))
        )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO layout from COCO splits (same train/val/test as RF-DETR)."
    )
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        default=None,
        help="Dataset root (images). Default: MOSQUITOES_DATASET or Kaggle download.",
    )
    parser.add_argument(
        "--out",
        metavar="DIR",
        default=None,
        help=f"Output directory (default: ./{DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--no-regenerate-coco",
        action="store_true",
        help="Skip convert_and_split_csv; use existing *_coco.json under labels/.",
    )
    parser.add_argument(
        "--detection-only",
        action="store_true",
        help="Export from *_coco_det.json into yolo_dataset_det/ (does not touch yolo_dataset/).",
    )
    args = parser.parse_args()

    dataset_arg = os.path.abspath(os.path.expanduser(args.dataset)) if args.dataset else None

    if args.no_regenerate_coco:
        image_root, labels_dir = resolve_dataset_layout(dataset_arg)
        source_path = image_root
        print(f"Using existing COCO JSONs under {labels_dir}/")
    else:
        source_path = convert_and_split_csv(
            dataset_arg, detection_only=args.detection_only
        )
        _, labels_dir = resolve_dataset_layout(dataset_arg)

    out_dir = args.out
    if out_dir is None and args.detection_only:
        out_dir = os.path.join(os.getcwd(), DEFAULT_OUT_DIR_DET)
    build_yolo_dataset(
        source_path,
        out_dir=out_dir,
        labels_dir=labels_dir,
        detection_only=args.detection_only,
    )


if __name__ == "__main__":
    main()
