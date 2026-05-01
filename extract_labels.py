import json

# The 20 specific images from your logs
target_images = [
    # Worst 10 overlaps
    "train_00757.jpeg", "train_04136.jpeg", "train_05363.jpeg", "train_05413.jpeg",
    "train_07454.jpeg", "train_07478.jpeg", "train_07811.jpeg", "train_08090.jpeg",
    "train_08201.jpeg", "train_09316.jpeg",
    # Best 10 overlaps
    "train_02705.jpeg", "train_08494.jpeg", "train_07018.jpeg", "train_04462.jpeg",
    "train_00434.jpeg", "train_06939.jpeg", "train_00158.jpeg", "train_04425.jpeg",
    "train_01200.jpeg", "train_05168.jpeg"
]

def extract_labels(input_json="test_predictions.json", output_json="top20_predictions.json"):
    print(f"Loading {input_json}...")
    with open(input_json, 'r', encoding='utf-8') as f:
        all_predictions = json.load(f)

    # Filter for only the images in our target list
    filtered_preds = [
        item for item in all_predictions 
        if any(target in item.get('image_path', '') for target in target_images)
    ]

    # Save the results
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(filtered_preds, f, indent=2)

    print(f"Successfully found {len(filtered_preds)} out of 20 images.")
    print(f"Saved their prediction labels to: {output_json}")

if __name__ == "__main__":
    extract_labels()