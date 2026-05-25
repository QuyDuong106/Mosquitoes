import pandas as pd
import json
import os
import random
import kagglehub

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Server / local runs: point at the kagglehub cache root OR a version folder.
#   export MOSQUITOES_DATASET=/path/to/mosquitoes-compsci760
#   export MOSQUITOES_DATASET_VERSION=3   # default; images from .../versions/3/
# COCO JSONs are written under <cache>/labels/ (sibling of versions/).
# Bounding boxes still come from manual_labels.csv in the repo (or MOSQUITOES_LABELS_CSV).
#
# Label CSV resolution (first match wins):
#   1. MOSQUITOES_LABELS_CSV
#   2. <this repo>/manual_labels.csv
#   3. <dataset_root>/labels/manual_labels.csv
#
# manual_labels.csv may have multiple rows per img_fName (multiple boxes per image).
# Train/val/test splits are by image so every box for an image stays in one split.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MANUAL_LABELS_NAME = "manual_labels.csv"
KAGGLE_DATASET_HANDLE = "duongnguyenquy/mosquitoes-compsci760"
DEFAULT_DATASET_VERSION = "3"


def dataset_version() -> str:
    """Kaggle dataset version folder name (e.g. ``3`` under ``versions/``)."""
    ver = os.environ.get("MOSQUITOES_DATASET_VERSION", DEFAULT_DATASET_VERSION).strip()
    return ver or DEFAULT_DATASET_VERSION


def kagglehub_dataset_handle(version: str | None = None) -> str:
    ver = version or dataset_version()
    return f"{KAGGLE_DATASET_HANDLE}/versions/{ver}"


def resolve_image_and_labels_dirs(root: str) -> tuple[str, str]:
    """
    Map a user/kagglehub path to (image_root, labels_dir).

    Kaggle Hub cache layout::

        mosquitoes-compsci760/
          labels/           <- COCO JSON output / optional manual_labels.csv
          versions/3/       <- images (version 3)
          versions/11/      <- other versions
    """
    root = os.path.abspath(os.path.expanduser(root))
    versions_parent = os.path.join(root, "versions")
    if os.path.isdir(versions_parent):
        ver = dataset_version()
        image_root = os.path.join(versions_parent, ver)
        labels_dir = os.path.join(root, "labels")
        if not os.path.isdir(image_root):
            available = sorted(
                name
                for name in os.listdir(versions_parent)
                if os.path.isdir(os.path.join(versions_parent, name))
            )
            raise FileNotFoundError(
                f"Dataset version {ver!r} not found under {versions_parent}. "
                f"Available version folders: {available or '(none)'}. "
                "Set MOSQUITOES_DATASET_VERSION or MOSQUITOES_DATASET=.../versions/<N>."
            )
        return image_root, labels_dir

    parent = os.path.dirname(root)
    if os.path.basename(parent) == "versions" and os.path.basename(root).isdigit():
        cache_root = os.path.dirname(parent)
        labels_at_cache = os.path.join(cache_root, "labels")
        if os.path.isdir(labels_at_cache):
            return root, labels_at_cache

    return root, os.path.join(root, "labels")

# Multi-class (species) vs detection-only (single "mosquito" class) use separate COCO files
# so a detection-only run never overwrites train_coco.json / val_coco.json / test_coco.json.
COCO_SPLIT_JSON = {
    False: {
        "train": "train_coco.json",
        "val": "val_coco.json",
        "test": "test_coco.json",
    },
    True: {
        "train": "train_coco_det.json",
        "val": "val_coco_det.json",
        "test": "test_coco_det.json",
    },
}

RFDETR_DATASET_DIR = "rfdetr_dataset"
RFDETR_DATASET_DIR_DET = "rfdetr_dataset_det"
RFDETR_OUTPUT_DIR = "output"
RFDETR_OUTPUT_DIR_DET = "output_det"


def coco_split_json_names(detection_only: bool) -> dict[str, str]:
    """Return train/val/test COCO basename mapping for the given training mode."""
    return dict(COCO_SPLIT_JSON[bool(detection_only)])


def resolve_labels_csv(
    dataset_path: str, labels_dir: str | None = None
) -> str:
    """Return absolute path to manual_labels.csv."""
    env = os.environ.get("MOSQUITOES_LABELS_CSV", "").strip()
    if env:
        path = os.path.abspath(os.path.expanduser(env))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MOSQUITOES_LABELS_CSV not found: {path}")
        return path

    labels_dir = labels_dir or os.path.join(dataset_path, "labels")
    candidates = [
        os.path.join(REPO_ROOT, MANUAL_LABELS_NAME),
        os.path.join(SCRIPT_DIR, MANUAL_LABELS_NAME),
        os.path.join(labels_dir, MANUAL_LABELS_NAME),
        os.path.join(dataset_path, "labels", MANUAL_LABELS_NAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    raise FileNotFoundError(
        "Could not find manual_labels.csv. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in candidates)
        + "\nOr set MOSQUITOES_LABELS_CSV to an absolute path."
    )


def resolve_dataset_layout(dataset_path=None) -> tuple[str, str]:
    """
    Return ``(image_root, labels_dir)`` for training/export.

    Resolution order when *dataset_path* is None:
    1. ``MOSQUITOES_DATASET`` (cache root or ``.../versions/<N>``)
    2. ``kagglehub.dataset_download('.../versions/<N>')`` with ``MOSQUITOES_DATASET_VERSION``
       (default version ``3``)
    """
    if dataset_path is not None:
        root = os.path.abspath(os.path.expanduser(str(dataset_path)))
    elif os.environ.get("MOSQUITOES_DATASET", "").strip():
        root = os.path.abspath(
            os.path.expanduser(os.environ["MOSQUITOES_DATASET"].strip())
        )
    else:
        handle = kagglehub_dataset_handle()
        print(f"Resolving dataset via kagglehub: {handle}")
        root = os.path.abspath(kagglehub.dataset_download(handle))

    image_root, labels_dir = resolve_image_and_labels_dirs(root)

    if not os.path.isdir(image_root):
        hints: list[str] = []
        root_lower = image_root.lower()
        if "path/to" in root_lower or image_root.rstrip("/").endswith("dataset_root"):
            hints.append(
                "This path looks like a README placeholder. "
                "Set MOSQUITOES_DATASET to your kagglehub cache folder."
            )
        hints.append(f"The image directory does not exist: {image_root}")
        if os.path.basename(root.rstrip("/")) == "labels":
            hints.append(
                "You pointed at the `labels/` folder. Use the cache root "
                "(parent of `labels/` and `versions/`) or `.../versions/3`."
            )
        hint_block = "\n\n".join(hints) if hints else ""
        raise FileNotFoundError(
            f"Dataset image root is not a directory:\n  {image_root}\n\n"
            + (f"Hints:\n{hint_block}\n" if hint_block else "")
        )

    os.makedirs(labels_dir, exist_ok=True)
    resolve_labels_csv(image_root, labels_dir)
    print(f"Dataset images (version {dataset_version()}): {image_root}")
    print(f"Dataset labels directory: {labels_dir}")
    return image_root, labels_dir


def resolve_dataset_root(dataset_path=None) -> str:
    """Return image root only (see :func:`resolve_dataset_layout` for labels path)."""
    image_root, _ = resolve_dataset_layout(dataset_path)
    return image_root

def build_image_index(dataset_path):
    """Recursively map lowercased image file names to canonical file names."""
    file_index = {}
    duplicate_names = set()

    for root, _, files in os.walk(dataset_path):
        if os.path.basename(root).lower() == "labels":
            continue
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            key = file_name.lower()
            if key in file_index and file_index[key] != file_name:
                duplicate_names.add(file_name)
                continue
            file_index.setdefault(key, file_name)

    return file_index, duplicate_names

def resolve_to_available_filename(raw_name, available_files):
    """Resolve CSV image names to actual files in the images folder."""
    direct = available_files.get(raw_name.lower())
    if direct:
        return direct

    stem, ext = os.path.splitext(raw_name)
    candidates = [ext, ".jpg", ".jpeg", ".png"]
    seen = set()
    for candidate_ext in candidates:
        key = f"{stem}{candidate_ext}".lower()
        if key in seen:
            continue
        seen.add(key)
        matched = available_files.get(key)
        if matched:
            return matched

    return None

def build_category_layout(df):
    """
    If annotations include class_label (species), build COCO categories (0..K-1).
    Otherwise a single generic 'mosquito' class.
    """
    col = "class_label"
    if col not in df.columns:
        categories = [{"id": 0, "name": "mosquito", "supercategory": "insect"}]
        return categories, {}, False

    work = df.dropna(subset=[col])
    if work.empty:
        categories = [{"id": 0, "name": "mosquito", "supercategory": "insect"}]
        return categories, {}, False

    unique = sorted(work[col].astype(str).unique())
    categories = [
        {"id": i, "name": name, "supercategory": "mosquito"}
        for i, name in enumerate(unique)
    ]
    label_to_id = {name: i for i, name in enumerate(unique)}
    return categories, label_to_id, True


def report_and_validate_multi_box_labels(df: pd.DataFrame) -> None:
    """
    Log multi-box stats and warn if the same image has inconsistent img_w/img_h.

    manual_labels.csv uses one CSV row per bounding box; the same img_fName may appear
    on many rows. This is supported end-to-end (COCO allows many annotations per image).
    """
    n_rows = len(df)
    n_images = df["img_fName"].nunique()
    boxes_per_image = df.groupby("img_fName", sort=False).size()
    multi = int((boxes_per_image > 1).sum())
    max_boxes = int(boxes_per_image.max()) if n_images else 0

    print(
        f"Label rows: {n_rows} ({n_images} unique images; "
        f"{multi} images with multiple boxes; max {max_boxes} boxes/image)"
    )

    if multi == 0:
        print("Note: every image has exactly one box in the CSV.")
        return

    # Warn when dimensions differ across rows for the same file (unusual but possible).
    dim_cols = ["img_w", "img_h"]
    if not all(c in df.columns for c in dim_cols):
        return

    inconsistent: list[str] = []
    for img_name, group in df.groupby("img_fName", sort=False):
        if group[dim_cols].drop_duplicates().shape[0] > 1:
            inconsistent.append(str(img_name))
    if inconsistent:
        sample = ", ".join(inconsistent[:5])
        extra = f" (+{len(inconsistent) - 5} more)" if len(inconsistent) > 5 else ""
        print(
            f"Warning: {len(inconsistent)} image(s) have differing img_w/img_h across rows; "
            f"COCO image size uses the first row per file. Examples: {sample}{extra}"
        )


def load_manual_labels_dataframe(
    image_root: str,
    csv_path: str | None = None,
    *,
    labels_dir: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load and normalize ``manual_labels.csv`` (same pipeline for multi-class and detection-only).

    :return: (dataframe, absolute path to the CSV used)
    """
    csv_path = csv_path or resolve_labels_csv(image_root, labels_dir)
    print(f"Labels source: {csv_path} ({MANUAL_LABELS_NAME})")
    df = pd.read_csv(csv_path)

    available_files, duplicate_names = build_image_index(image_root)
    if not available_files:
        raise RuntimeError(f"No images found under dataset path: {image_root}")
    if duplicate_names:
        print(
            f"Warning: found duplicate image basenames; using first match for {len(duplicate_names)} files."
        )

    print("Normalizing annotation filenames against available images...")
    df["resolved_img_fName"] = df["img_fName"].apply(
        lambda name: resolve_to_available_filename(name, available_files)
    )
    missing_rows = int(df["resolved_img_fName"].isna().sum())
    if missing_rows:
        print(f"Warning: dropping {missing_rows} annotation rows with missing images.")
    df = df.dropna(subset=["resolved_img_fName"]).copy()
    df["img_fName"] = df["resolved_img_fName"]
    df = df.drop(columns=["resolved_img_fName"])
    if df.empty:
        raise RuntimeError("No valid annotation rows remain after filename resolution.")

    if "class_label" in df.columns:
        n_miss = int(df["class_label"].isna().sum())
        if n_miss:
            print(f"Warning: dropping {n_miss} rows with missing class_label.")
        df = df.dropna(subset=["class_label"]).copy()
        if df.empty:
            raise RuntimeError("No rows left after dropping missing class_label.")

    return df, csv_path


def image_sets_from_existing_coco(labels_dir: str) -> dict[str, set[str]] | None:
    """Return train/val/test image file_name sets from multi-class COCO JSONs, or None if missing."""
    json_names = coco_split_json_names(False)
    splits: dict[str, set[str]] = {}
    for split_key in ("train", "val", "test"):
        path = os.path.join(labels_dir, json_names[split_key])
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        splits[split_key] = {str(img["file_name"]) for img in data.get("images", [])}
    return splits


def compute_shuffled_image_splits(unique_images: list[str]) -> tuple[set[str], set[str], set[str]]:
    """70/15/15 split by image with seed 42 (same logic for both training modes)."""
    images = list(unique_images)
    random.seed(42)
    random.shuffle(images)

    total_imgs = len(images)
    if total_imgs < 3:
        raise RuntimeError(
            f"Need at least 3 valid images to create train/val/test splits, found {total_imgs}."
        )

    train_split_idx = int(total_imgs * 0.70)
    val_split_idx = int(total_imgs * 0.85)

    train_split_idx = max(1, min(train_split_idx, total_imgs - 2))
    val_split_idx = max(train_split_idx + 1, min(val_split_idx, total_imgs - 1))

    train_imgs = set(images[:train_split_idx])
    val_imgs = set(images[train_split_idx:val_split_idx])
    test_imgs = set(images[val_split_idx:])
    return train_imgs, val_imgs, test_imgs


def resolve_train_val_test_splits(
    df: pd.DataFrame,
    labels_dir: str,
    *,
    detection_only: bool,
) -> tuple[set[str], set[str], set[str]]:
    """
    Assign each image to train, val, or test.

    Detection-only: if multi-class ``train_coco.json`` (etc.) already exist, reuse their
    image assignments so both modes see the exact same images per split. Otherwise use
    the same seed-42 shuffle as multi-class (identical when ``df`` is the same).
    """
    unique_images = df["img_fName"].unique().tolist()
    all_images = set(unique_images)

    if detection_only:
        existing = image_sets_from_existing_coco(labels_dir)
        if existing is not None:
            train_imgs = existing["train"] & all_images
            val_imgs = existing["val"] & all_images
            test_imgs = existing["test"] & all_images
            covered = train_imgs | val_imgs | test_imgs
            overlap = (
                (train_imgs & val_imgs)
                | (train_imgs & test_imgs)
                | (val_imgs & test_imgs)
            )
            if overlap:
                print(
                    f"Warning: overlapping images across multi-class splits ({len(overlap)}); "
                    "recomputing splits with seed 42."
                )
            elif covered != all_images:
                missing = len(all_images - covered)
                extra = len(covered - all_images)
                print(
                    "Warning: multi-class COCO splits do not cover the current label set "
                    f"(missing {missing} images, extra {extra}); recomputing splits with seed 42."
                )
            else:
                mc_names = coco_split_json_names(False)
                print(
                    "Reusing train/val/test image assignments from existing multi-class COCO "
                    f"({mc_names['train']}, etc.) — same images and boxes as classification+detection, "
                    "single class for detection-only export."
                )
                return train_imgs, val_imgs, test_imgs

    print("Shuffling and splitting images (by image, not by box; seed 42)...")
    return compute_shuffled_image_splits(unique_images)


def build_coco_dict(df, image_set, categories, label_to_id, use_class_labels: bool):
    """Build a COCO dict for a subset of images (all box rows per image are included)."""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": list(categories),
    }
    class_col = "class_label"

    images_dict = {}
    image_id_counter = 0
    annotation_id_counter = 0

    subset_df = df[df["img_fName"].isin(image_set)]

    for index, row in subset_df.iterrows():
        img_name = row["img_fName"]

        if img_name not in images_dict:
            images_dict[img_name] = image_id_counter
            coco_data["images"].append(
                {
                    "id": image_id_counter,
                    "file_name": img_name,
                    "width": int(row["img_w"]),
                    "height": int(row["img_h"]),
                }
            )
            image_id_counter += 1

        xtl = float(row["bbx_xtl"])
        ytl = float(row["bbx_ytl"])
        xbr = float(row["bbx_xbr"])
        ybr = float(row["bbx_ybr"])

        bbox_width = xbr - xtl
        bbox_height = ybr - ytl

        if use_class_labels and pd.notna(row.get(class_col, None)):
            cat_id = label_to_id[str(row[class_col])]
        else:
            cat_id = 0

        coco_data["annotations"].append(
            {
                "id": annotation_id_counter,
                "image_id": images_dict[img_name],
                "category_id": cat_id,
                "bbox": [xtl, ytl, bbox_width, bbox_height],
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
            }
        )
        annotation_id_counter += 1

    return coco_data

def convert_and_split_csv(dataset_path=None, *, detection_only: bool = False):
    """
    Write COCO split JSON files under ``<dataset_root>/labels/``.

    Multi-class (default): ``train_coco.json``, ``val_coco.json``, ``test_coco.json``
    with one category per ``class_label`` species.

    Detection-only (``detection_only=True``): ``train_coco_det.json``, etc., with a
    single ``mosquito`` class (same ``manual_labels.csv`` rows, boxes, and image splits;
    species stored only in multi-class JSON). Does not overwrite the multi-class JSON files.

    :param dataset_path: Dataset root, or None to use ``MOSQUITOES_DATASET`` then Kaggle download.
    :param detection_only: If True, export single-class COCO files with ``*_coco_det.json`` names.
    :return: Absolute dataset root (same layout as input).
    """
    image_root, labels_dir = resolve_dataset_layout(dataset_path)

    df, csv_path = load_manual_labels_dataframe(image_root, labels_dir=labels_dir)

    if detection_only:
        categories = [{"id": 0, "name": "mosquito", "supercategory": "insect"}]
        label_to_id: dict[str, int] = {}
        use_class_labels = False
        print(
            "Detection-only export: same manual_labels.csv rows and splits as multi-class; "
            "category_id=0 for all boxes (species labels not written to COCO)."
        )
    else:
        categories, label_to_id, use_class_labels = build_category_layout(df)

    print(
        f"COCO categories ({len(categories)}): "
        + ", ".join(str(c["name"]) for c in categories)
    )
    report_and_validate_multi_box_labels(df)

    train_imgs, val_imgs, test_imgs = resolve_train_val_test_splits(
        df, labels_dir, detection_only=detection_only
    )
    total_imgs = len(train_imgs) + len(val_imgs) + len(test_imgs)
    print(f"Total Images: {total_imgs}")

    # --- BUILD AND SAVE JSONS ---
    print("Generating COCO JSON files...")
    train_coco = build_coco_dict(
        df, train_imgs, categories, label_to_id, use_class_labels
    )
    val_coco = build_coco_dict(df, val_imgs, categories, label_to_id, use_class_labels)
    test_coco = build_coco_dict(df, test_imgs, categories, label_to_id, use_class_labels)

    print(
        f" -> Training: {len(train_imgs)} images, {len(train_coco['annotations'])} boxes"
    )
    print(
        f" -> Validation: {len(val_imgs)} images, {len(val_coco['annotations'])} boxes"
    )
    print(
        f" -> Testing: {len(test_imgs)} images, {len(test_coco['annotations'])} boxes"
    )

    json_names = coco_split_json_names(detection_only)
    for key, coco_dict in (
        ("train", train_coco),
        ("val", val_coco),
        ("test", test_coco),
    ):
        out_path = os.path.join(labels_dir, json_names[key])
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f)

    written = ", ".join(json_names[k] for k in ("train", "val", "test"))
    print(f"Done! Wrote {written}.")
    if detection_only:
        multi = coco_split_json_names(False)
        print(
            "Multi-class COCO files were not modified: "
            + ", ".join(multi[k] for k in ("train", "val", "test")) + "."
        )
    print(f"Label CSV used for this export: {csv_path}")
    return image_root


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=(
            "Build COCO JSON splits from manual_labels.csv. "
            "Multiple CSV rows per image (multiple boxes) are supported."
        )
    )
    ap.add_argument(
        "--dataset",
        metavar="PATH",
        default=None,
        help="Dataset root (images). Labels: manual_labels.csv in repo or labels/. "
        "Default: MOSQUITOES_DATASET env, else Kaggle download.",
    )
    ap.add_argument(
        "--detection-only",
        action="store_true",
        help="Single-class COCO (*_coco_det.json) from the same manual_labels.csv and "
        "train/val/test images as multi-class (reuses existing train_coco.json splits when present).",
    )
    ns = ap.parse_args()
    convert_and_split_csv(ns.dataset, detection_only=ns.detection_only)