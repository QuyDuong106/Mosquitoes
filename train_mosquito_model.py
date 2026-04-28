import os
import json
import shutil
from PIL import Image
import supervision as sv
from rfdetr import RFDETRSmall 
import kagglehub 
from convert_to_coco import convert_and_split_csv

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def build_image_index(dataset_path):
    """Recursively map lowercased image file names to absolute paths."""
    image_index = {}
    for root, _, files in os.walk(dataset_path):
        if os.path.basename(root).lower() == "labels":
            continue
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            key = file_name.lower()
            absolute_path = os.path.join(root, file_name)
            image_index.setdefault(key, absolute_path)
    return image_index

def resolve_image_path(img_name, available_files):
    """Return an existing source image path and its canonical file name."""
    direct_path = available_files.get(img_name.lower())
    if direct_path:
        return direct_path, os.path.basename(direct_path)

    stem, ext = os.path.splitext(img_name)
    candidate_exts = [ext, ".jpg", ".jpeg", ".png"]
    seen = set()
    for candidate_ext in candidate_exts:
        normalized_ext = candidate_ext.lower()
        if normalized_ext in seen:
            continue
        seen.add(normalized_ext)

        candidate_name = f"{stem}{candidate_ext}"
        candidate_path = available_files.get(candidate_name.lower())
        if candidate_path:
            return candidate_path, os.path.basename(candidate_path)

    for candidate_ext in candidate_exts:
        candidate_name = f"{stem}{candidate_ext}".lower()
        candidate_path = available_files.get(candidate_name)
        if candidate_path:
            return candidate_path, os.path.basename(candidate_path)

    return None, None

def create_roboflow_structure(source_dataset_path):
    """Tricks RF-DETR by building the exact folder structure it demands using absolute symlinks."""
    target_dir = os.path.join(os.getcwd(), "rfdetr_dataset")
    
    # Wipe the old directory to clear out broken shortcuts
    if os.path.exists(target_dir):
        print("Cleaning up old dataset directory...")
        shutil.rmtree(target_dir)
        
    print("Restructuring data with absolute symlinks...")
    os.makedirs(target_dir, exist_ok=True)
    
    # RF-DETR expects 'valid' for the validation set
    splits = {
        "train": "train_coco.json",
        "valid": "val_coco.json", 
        "test": "test_coco.json"
    }
    
    source_labels_dir = os.path.abspath(os.path.join(source_dataset_path, "labels"))
    available_files = build_image_index(source_dataset_path)
    if not available_files:
        raise RuntimeError(f"No image files found under dataset path: {source_dataset_path}")

    split_counts = {}
    
    for split_name, json_name in splits.items():
        print(f" -> Organizing {split_name} split...")
        split_dir = os.path.join(target_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # 1. Copy the JSON to the exact name RF-DETR expects
        source_json = os.path.join(source_labels_dir, json_name)
        target_json = os.path.join(split_dir, "_annotations.coco.json")
        shutil.copy2(source_json, target_json)
            
        # 2. Parse JSON and keep only images that exist on disk.
        with open(source_json, 'r') as f:
            data = json.load(f)
            images_to_link = data.get("images", [])

        kept_images = []
        valid_image_ids = set()
        missing_images = []

        for image in images_to_link:
            img_name = image["file_name"]
            src_img, canonical_name = resolve_image_path(img_name, available_files)
            if src_img is None:
                missing_images.append(img_name)
                continue

            image["file_name"] = canonical_name
            kept_images.append(image)
            valid_image_ids.add(image["id"])

            dst_img = os.path.join(split_dir, canonical_name)
            if not os.path.exists(dst_img):
                os.symlink(src_img, dst_img)

        data["images"] = kept_images
        data["annotations"] = [
            ann for ann in data.get("annotations", [])
            if ann.get("image_id") in valid_image_ids
        ]

        with open(target_json, "w") as f:
            json.dump(data, f)

        split_counts[split_name] = len(kept_images)
        print(
            f"    Kept {len(kept_images)} images and {len(data['annotations'])} annotations."
        )
        if missing_images:
            print(
                f"    Warning: skipped {len(missing_images)} missing images in {split_name}."
            )

    empty_splits = [name for name, count in split_counts.items() if count == 0]
    if empty_splits:
        raise RuntimeError(
            "One or more splits have zero images after dataset validation: "
            f"{', '.join(empty_splits)}. "
            "Please regenerate the COCO split files (train/val/test) so each split has valid image filenames."
        )

    return target_dir

def main():
    # ---------------------------------------------------------
    # 1. DOWNLOAD & FORMAT THE DATASET
    # ---------------------------------------------------------
    print("Fetching dataset from Kaggle cache...")
    source_dataset_path = kagglehub.dataset_download("duongnguyenquy/mosquitoes-compsci760")

    # Regenerate COCO split files from actual images to avoid stale/broken file names.
    convert_and_split_csv(source_dataset_path)
    
    # Build the required folder structure locally
    rf_dataset_dir = create_roboflow_structure(source_dataset_path)

    # ---------------------------------------------------------
    # 2. INITIALIZE THE MODEL
    # ---------------------------------------------------------
    print("Initializing RF-DETR model...")
    model = RFDETRSmall()

    # ---------------------------------------------------------
    # 3. TRAIN THE MODEL
    # ---------------------------------------------------------
    print("Starting training with validation...")
    
    # Because the folder structure is now perfect, RF-DETR handles the rest!
    model.train(
        dataset_dir=rf_dataset_dir,
        epochs=50,        
        lr=1e-4           
    )
    print("Training completed successfully!")

    # ---------------------------------------------------------
    # 4. TEST / INFERENCE ON AN UNSEEN IMAGE
    # ---------------------------------------------------------
    print("Running inference test on an UNSEEN image from the Test Set...")
    
    test_split_dir = os.path.join(rf_dataset_dir, "test")
    test_annotations = os.path.join(test_split_dir, "_annotations.coco.json")
    
    with open(test_annotations, 'r') as f:
        test_data = json.load(f)
        
    if test_data["images"]:
        test_image_filename = test_data["images"][0]["file_name"]
        test_image_path = os.path.join(test_split_dir, test_image_filename)
        
        print(f"Testing model on: {test_image_path}")
        image = Image.open(test_image_path)
        
        detections = model.predict(image, threshold=0.5)
        
        annotated_image = image.copy()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
        
        print(f"Found {len(detections)} mosquitoes in the image!")
        
        save_path = "final_test_prediction.jpg"
        annotated_image.save(save_path)
        print(f"Saved visualization to {save_path}.")

if __name__ == "__main__":
    main()