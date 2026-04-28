import pandas as pd
import json
import os
import random
import kagglehub

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

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

def build_coco_dict(df, image_set):
    """Helper function to build a COCO dictionary for a specific subset of images."""
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "mosquito", "supercategory": "insect"}]
    }

    images_dict = {}
    image_id_counter = 0
    annotation_id_counter = 0

    # Filter the dataframe to ONLY include rows for the images in our current set
    subset_df = df[df["img_fName"].isin(image_set)]

    for index, row in subset_df.iterrows():
        img_name = row['img_fName']
        
        # 1. Register the image
        if img_name not in images_dict:
            images_dict[img_name] = image_id_counter
            coco_data["images"].append({
                "id": image_id_counter,
                "file_name": img_name,
                "width": int(row['img_w']),
                "height": int(row['img_h'])
            })
            image_id_counter += 1
            
        # 2. Register the annotation
        xtl = float(row['bbx_xtl'])
        ytl = float(row['bbx_ytl'])
        xbr = float(row['bbx_xbr']) 
        ybr = float(row['bbx_ybr'])
        
        bbox_width = xbr - xtl
        bbox_height = ybr - ytl
        
        coco_data["annotations"].append({
            "id": annotation_id_counter,
            "image_id": images_dict[img_name],
            "category_id": 0,
            "bbox": [xtl, ytl, bbox_width, bbox_height],
            "area": bbox_width * bbox_height,
            "iscrowd": 0 
        })
        annotation_id_counter += 1
        
    return coco_data

def convert_and_split_csv(dataset_path=None):
    print("Locating Kaggle dataset...")
    if dataset_path is None:
        dataset_path = kagglehub.dataset_download("duongnguyenquy/mosquitoes-compsci760")
    
    csv_path = os.path.join(dataset_path, "labels", "annotations.csv")
    labels_dir = os.path.join(dataset_path, "labels")
    
    print(f"Reading CSV from: {csv_path}")
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

    # --- THE SPLITTING LOGIC ---
    print("Shuffling and splitting images...")
    
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

    train_split_idx = int(total_imgs * 0.80)
    val_split_idx = int(total_imgs * 0.90) # 80% + 10%

    train_split_idx = max(1, min(train_split_idx, total_imgs - 2))
    val_split_idx = max(train_split_idx + 1, min(val_split_idx, total_imgs - 1))
    
    # Slice the list into three groups
    train_imgs = set(unique_images[:train_split_idx])
    val_imgs = set(unique_images[train_split_idx:val_split_idx])
    test_imgs = set(unique_images[val_split_idx:])
    
    print(f"Total Images: {total_imgs}")
    print(f" -> Training: {len(train_imgs)} images")
    print(f" -> Validation: {len(val_imgs)} images")
    print(f" -> Testing: {len(test_imgs)} images")

    # --- BUILD AND SAVE JSONS ---
    print("Generating COCO JSON files...")
    train_coco = build_coco_dict(df, train_imgs)
    val_coco = build_coco_dict(df, val_imgs)
    test_coco = build_coco_dict(df, test_imgs)

    with open(os.path.join(labels_dir, 'train_coco.json'), 'w') as f:
        json.dump(train_coco, f)
    with open(os.path.join(labels_dir, 'val_coco.json'), 'w') as f:
        json.dump(val_coco, f)
    with open(os.path.join(labels_dir, 'test_coco.json'), 'w') as f:
        json.dump(test_coco, f)
        
    print("Done! train_coco.json, val_coco.json, and test_coco.json have been created.")

if __name__ == "__main__":
    convert_and_split_csv()