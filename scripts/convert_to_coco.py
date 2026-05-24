import pandas as pd
import json
import os
import random
import kagglehub

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Server / local runs: point to the folder kagglehub (or your copy) unpacks — the dataset
# root with images (not the `labels/` folder itself). Labels come from manual_labels.csv.
#   export MOSQUITOES_DATASET=/path/to/mosquitoes-compsci760
# Or: python convert_to_coco.py --dataset /path/to/mosquitoes-compsci760
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


def resolve_labels_csv(dataset_path: str) -> str:
    """Return absolute path to manual_labels.csv."""
    env = os.environ.get("MOSQUITOES_LABELS_CSV", "").strip()
    if env:
        path = os.path.abspath(os.path.expanduser(env))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MOSQUITOES_LABELS_CSV not found: {path}")
        return path

    candidates = [
        os.path.join(REPO_ROOT, MANUAL_LABELS_NAME),
        os.path.join(SCRIPT_DIR, MANUAL_LABELS_NAME),
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


def resolve_dataset_root(dataset_path=None):
    """
    Return absolute dataset root (images tree for training).

    Resolution order when *dataset_path* is None:
    1. Environment variable ``MOSQUITOES_DATASET``
    2. ``kagglehub.dataset_download(...)``
    """
    if dataset_path is not None:
        root = os.path.abspath(os.path.expanduser(str(dataset_path)))
    elif os.environ.get("MOSQUITOES_DATASET", "").strip():
        root = os.path.abspath(
            os.path.expanduser(os.environ["MOSQUITOES_DATASET"].strip())
        )
    else:
        root = os.path.abspath(
            kagglehub.dataset_download("duongnguyenquy/mosquitoes-compsci760")
        )

    if not os.path.isdir(root):
        hints: list[str] = []
        root_lower = root.lower()
        if "path/to" in root_lower or root.rstrip("/").endswith("dataset_root"):
            hints.append(
                "This path looks like a README placeholder, not a real folder on disk. "
                "Set MOSQUITOES_DATASET (or --dataset) to your actual Kaggle dataset root."
            )
        hints.append(f"The directory does not exist: {root}")
        if os.path.basename(root.rstrip("/")) == "labels":
            hints.append(
                "You pointed at the `labels/` folder. Use its parent instead "
                "(MOSQUITOES_DATASET should be the folder that *contains* `labels/`)."
            )
        hint_block = "\n\n".join(hints) if hints else ""
        raise FileNotFoundError(
            f"Dataset root is not a directory:\n  {root}\n\n"
            "Use the dataset ROOT (directory with images). On a server this is often the path "
            "from `kagglehub.dataset_download('duongnguyenquy/mosquitoes-compsci760')`.\n\n"
            + (f"Hints:\n{hint_block}\n" if hint_block else "")
        )

    # Ensure manual_labels.csv is reachable before conversion runs.
    resolve_labels_csv(root)
    return root

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

def convert_and_split_csv(dataset_path=None):
    """
    Write ``train_coco.json``, ``val_coco.json``, ``test_coco.json`` under
    ``<dataset_root>/labels/``.

    :param dataset_path: Dataset root, or None to use ``MOSQUITOES_DATASET`` then Kaggle download.
    :return: Absolute dataset root (same layout as input).
    """
    dataset_path = resolve_dataset_root(dataset_path)
    print(f"Dataset root: {dataset_path}")

    csv_path = resolve_labels_csv(dataset_path)
    labels_dir = os.path.join(dataset_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Reading labels from: {csv_path}")
    df = pd.read_csv(csv_path)

    available_files, duplicate_names = build_image_index(dataset_path)
    if not available_files:
        raise RuntimeError(f"No images found under dataset path: {dataset_path}")
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

    categories, label_to_id, use_class_labels = build_category_layout(df)
    print(
        f"COCO categories ({len(categories)}): "
        + ", ".join(str(c["name"]) for c in categories)
    )
    report_and_validate_multi_box_labels(df)

    # --- THE SPLITTING LOGIC ---
    print("Shuffling and splitting images (by image, not by box)...")
    
    # Get a list of all unique images
    unique_images = df['img_fName'].unique().tolist()
    
    # Shuffle the list randomly (Seed 42 ensures we get the same shuffle if we run it twice)
    random.seed(42)
    random.shuffle(unique_images)
    
    total_imgs = len(unique_images)
    if total_imgs < 3:
        raise RuntimeError(
            f"Need at least 3 valid images to create train/val/test splits, found {total_imgs}."
        )

    train_split_idx = int(total_imgs * 0.70)
    val_split_idx = int(total_imgs * 0.85)  # 70% train + 15% val

    train_split_idx = max(1, min(train_split_idx, total_imgs - 2))
    val_split_idx = max(train_split_idx + 1, min(val_split_idx, total_imgs - 1))
    
    # Slice the list into three groups
    train_imgs = set(unique_images[:train_split_idx])
    val_imgs = set(unique_images[train_split_idx:val_split_idx])
    test_imgs = set(unique_images[val_split_idx:])
    
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

    with open(os.path.join(labels_dir, 'train_coco.json'), 'w') as f:
        json.dump(train_coco, f)
    with open(os.path.join(labels_dir, 'val_coco.json'), 'w') as f:
        json.dump(val_coco, f)
    with open(os.path.join(labels_dir, 'test_coco.json'), 'w') as f:
        json.dump(test_coco, f)
        
    print("Done! train_coco.json, val_coco.json, and test_coco.json have been created.")
    return dataset_path


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
    ns = ap.parse_args()
    convert_and_split_csv(ns.dataset)