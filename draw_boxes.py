import json
import os
from PIL import Image, ImageDraw

# We know these from your earlier terminal evaluation logs!
WORST_10 = [
    "train_00757.jpeg", "train_04136.jpeg", "train_05363.jpeg", "train_05413.jpeg",
    "train_07454.jpeg", "train_07478.jpeg", "train_07811.jpeg", "train_08090.jpeg",
    "train_08201.jpeg", "train_09316.jpeg"
]

BEST_10 = [
    "train_02705.jpeg", "train_08494.jpeg", "train_07018.jpeg", "train_04462.jpeg",
    "train_00434.jpeg", "train_06939.jpeg", "train_00158.jpeg", "train_04425.jpeg",
    "train_01200.jpeg", "train_05168.jpeg"
]

def draw_predictions_and_sort(json_file="top20_predictions.json"):
    # Create the sub-folders
    best_dir = os.path.join("annotated_images", "best_10")
    worst_dir = os.path.join("annotated_images", "worst_10")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)

    # Load the predictions
    with open(json_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    for item in predictions:
        # Get the bare filename (e.g., "train_00158.jpeg")
        filename = os.path.basename(item['image_path'])
        
        # Decide which folder this image belongs to
        if filename in BEST_10:
            out_dir = best_dir
        elif filename in WORST_10:
            out_dir = worst_dir
        else:
            continue # Skip if it's not one of our targeted 20
        
        # Look for the image. It handles both cases: whether you put them right 
        # next to the script, or if they are still inside rfdetr_dataset/test/
        if os.path.exists(filename):
            img_path_to_open = filename
        elif os.path.exists(item['image_path']):
            img_path_to_open = item['image_path']
        else:
            print(f"Skipping {filename} - Could not find the image file.")
            continue
            
        # Open the image
        img = Image.open(img_path_to_open).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Draw the boxes
        for det in item['detections']:
            x1, y1, x2, y2 = det['xyxy']
            score = det['score']
            
            # Red outline, 5 pixels thick
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            
            # Draw the confidence score
            label = f"Score: {score:.2f}"
            draw.text((x1, max(0, y1 - 15)), label, fill="red")
            
        # Save into the correct sub-folder
        save_path = os.path.join(out_dir, filename)
        img.save(save_path)
        print(f"Saved {filename} --> {out_dir}/")

if __name__ == "__main__":
    draw_predictions_and_sort()