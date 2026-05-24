import argparse
import inspect
import os
import json
import shutil
from PIL import Image
import supervision as sv
from rfdetr import RFDETRSmall
from convert_to_coco import convert_and_split_csv
from dataset_images import build_image_index, resolve_image_path

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
    parser = argparse.ArgumentParser(
        description="Train RF-DETR on the mosquito dataset (COCO under dataset_root/labels/)."
    )
    parser.add_argument(
        "--dataset",
        metavar="PATH",
        default=None,
        help="Dataset root: directory with images (labels from manual_labels.csv in repo). "
        "If omitted, uses env MOSQUITOES_DATASET; if unset, downloads via kagglehub.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (early stopping may finish sooner).",
    )
    parser.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when validation mAP fails to improve (RF-DETR built-in; not raw train loss). "
        "Use --no-early-stopping to run all epochs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        metavar="N",
        help="Stop after N consecutive epochs without enough mAP improvement (with early stopping).",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.001,
        metavar="D",
        help="Minimum validation mAP increase to count as improvement (default 0.001 = 0.1%% mAP).",
    )
    parser.add_argument(
        "--early-stopping-use-ema",
        action="store_true",
        help="If set, early stopping compares the EMA weights' validation mAP.",
    )
    args = parser.parse_args()
    dataset_arg = os.path.abspath(os.path.expanduser(args.dataset)) if args.dataset else None

    # ---------------------------------------------------------
    # 1. FORMAT THE DATASET (and download only if no root was given)
    # ---------------------------------------------------------
    if dataset_arg:
        print(f"Using dataset root from --dataset: {dataset_arg}")
    elif os.environ.get("MOSQUITOES_DATASET", "").strip():
        print("Using dataset root from MOSQUITOES_DATASET.")
    else:
        print("No --dataset / MOSQUITOES_DATASET; fetching from Kaggle cache via kagglehub…")

    source_dataset_path = convert_and_split_csv(dataset_arg)
    
    # Build the required folder structure locally
    rf_dataset_dir = create_roboflow_structure(source_dataset_path)

    train_coco_path = os.path.join(source_dataset_path, "labels", "train_coco.json")
    with open(train_coco_path, "r", encoding="utf-8") as f:
        train_coco_meta = json.load(f)
    num_classes = len(train_coco_meta.get("categories", [{"id": 0}]))
    print(f"Training with num_classes={num_classes} (from COCO categories).")

    # ---------------------------------------------------------
    # 2. INITIALIZE THE MODEL
    # ---------------------------------------------------------
    print("Initializing RF-DETR model...")
    try:
        model = RFDETRSmall(num_classes=num_classes)
    except TypeError:
        model = RFDETRSmall()

    # ---------------------------------------------------------
    # 3. TRAIN THE MODEL
    # ---------------------------------------------------------
    print("Starting training with validation...")

    train_kw: dict = {
        "dataset_dir": rf_dataset_dir,
        "epochs": args.epochs,
        "lr": 1e-4,
    }
    if args.early_stopping:
        train_kw.update(
            {
                "early_stopping": True,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
                "early_stopping_use_ema": args.early_stopping_use_ema,
            }
        )
        print(
            "Early stopping on validation mAP: "
            f"patience={args.early_stopping_patience} epochs, "
            f"min_delta={args.early_stopping_min_delta}, "
            f"use_ema={args.early_stopping_use_ema}."
        )
    else:
        print("Early stopping disabled; training will run for all epochs unless interrupted.")

    sig = inspect.signature(model.train)
    params = sig.parameters
    allowed = set(params.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_varkw:
        forward = train_kw
    else:
        forward = {k: v for k, v in train_kw.items() if k in allowed}
        skipped = sorted(set(train_kw) - set(forward))
        if skipped:
            print(
                "Warning: this rfdetr build does not accept train() parameters "
                f"{skipped}; they will be ignored."
            )
        if args.early_stopping and "early_stopping" not in allowed:
            print(
                "Warning: upgrade rfdetr for early stopping "
                "(see https://rfdetr.roboflow.com/latest/learn/train/advanced/)."
            )

    model.train(**forward)
    print("Training finished (completed all epochs or stopped early).")

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