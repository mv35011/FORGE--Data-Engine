import json
import os
from tqdm import tqdm

# --- CONFIG ---
# Paths to your JSON files
JSON_FILES = [
    "ppe_coco_format/ppe_coco_format/train/_annotations.coco.json",
    "ppe_coco_format/ppe_coco_format/valid/_annotations.coco.json",
    "ppe_coco_format/ppe_coco_format/test/_annotations.coco.json"
]


def denormalize_bboxes(json_path):
    if not os.path.exists(json_path): return

    print(f"Checking {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Map Image IDs to Dimensions
    # We need width/height to convert 0.5 -> 960px
    img_dims = {img['id']: (img['width'], img['height']) for img in data['images']}

    converted_count = 0

    for ann in tqdm(data['annotations']):
        bbox = ann['bbox']
        img_w, img_h = img_dims[ann['image_id']]

        # Check if Normalized (values all <= 1.0)
        # Note: We check if x+w <= 1.5 to be safe, as sometimes noise makes it 1.01
        if all(val <= 1.01 for val in bbox):
            # ASSUMPTION: If normalized, is it YOLO format (cx, cy, w, h) or COCO (x, y, w, h)?
            # Standard Roboflow export to COCO usually keeps (x, y, w, h) but normalizes them.
            # If your data came from YOLO txt, it might be center-based.
            # Standard COCO JSON logic is top-left (x, y, w, h).

            x_norm, y_norm, w_norm, h_norm = bbox

            # Convert to Pixels
            x = x_norm * img_w
            y = y_norm * img_h
            w = w_norm * img_w
            h = h_norm * img_h

            ann['bbox'] = [x, y, w, h]
            converted_count += 1

    if converted_count > 0:
        print(f"⚠️  Found and fixed {converted_count} normalized boxes.")
        # Create backup
        os.rename(json_path, json_path + ".bak")
        with open(json_path, 'w') as f:
            json.dump(data, f)
        print("✅ Saved fixed file.")
    else:
        print("✅ No normalized boxes found. Data is already in pixels.")


if __name__ == "__main__":
    for f in JSON_FILES:
        denormalize_bboxes(f)