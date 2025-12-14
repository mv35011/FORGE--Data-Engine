import os
import cv2
import json
import random
import shutil
import albumentations as A
import numpy as np
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================
DATASET_ROOT = "/workspace/dataset/ppe_coco_format"
TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
JSON_FILE = "_annotations.coco.json"
NUM_AUGS = 2

# ================================
# PIPELINE
# ================================
transform = A.Compose([
    A.LongestMaxSize(max_size=896, p=1),
    A.PadIfNeeded(min_height=896, min_width=896, border_mode=cv2.BORDER_CONSTANT, p=1),
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.MotionBlur(blur_limit=7),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
        # Using scale_limit for compatibility
        A.Downscale(scale_limit=0.5, p=1),
    ], p=0.6),
    A.OneOf([
        A.RandomBrightnessContrast(),
        A.RandomGamma(),
        A.CLAHE(),
    ], p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids'], min_area=1))  # reduced min_area


def validate_structure():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ FATAL: Directory not found: {TRAIN_DIR}")
        exit(1)
    print(f"✅ Working in: {TRAIN_DIR}")


# ================================
# SMART COORDINATE HANDLING
# ================================
def is_normalized(bbox):
    """Checks if a bbox [x,y,w,h] is likely normalized (all values <= 1.0)"""
    return all(v <= 1.01 for v in bbox)


def denormalize_bbox(bbox, width, height):
    """Converts normalized [0-1] to pixels [0-W]"""
    x, y, w, h = bbox
    return [x * width, y * height, w * width, h * height]


def clean_bbox(bbox, img_w, img_h):
    """Ensures bbox is within image bounds and valid"""
    x, y, w, h = bbox
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [x, y, w, h]


# ================================
# MAIN LOGIC
# ================================
def augment_in_place():
    print(f"\n🚀 Starting Smart Augmentation...")
    json_path = os.path.join(TRAIN_DIR, JSON_FILE)

    # Backup Logic
    if not os.path.exists(json_path + ".bak"):
        shutil.copy(json_path, json_path + ".bak")
        print("📦 Created new backup.")

    with open(json_path, "r") as f:
        coco = json.load(f)

    if any("aug_" in img["file_name"] for img in coco["images"]):
        print("⚠️  Dataset already augmented. Skipping.")
        return

    # Map annotations
    ann_map = {img["id"]: [] for img in coco["images"]}
    for ann in coco["annotations"]:
        ann_map[ann["image_id"]].append(ann)

    new_images, new_annotations = [], []

    # ID counters
    img_id_counter = max(img["id"] for img in coco["images"]) + 1 if coco["images"] else 1
    ann_id_counter = max(ann["id"] for ann in coco["annotations"]) + 1 if coco["annotations"] else 1

    original_images = list(coco["images"])

    for img_info in tqdm(original_images, desc="Augmenting"):
        img_path = os.path.join(TRAIN_DIR, img_info["file_name"])
        image = cv2.imread(img_path)

        if image is None: continue

        # Get Dimensions
        h_orig, w_orig = image.shape[:2]

        # Verify JSON matches Reality
        # (Sometimes JSON has wrong w/h, trusting actual image is safer for pixels)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        anns = ann_map.get(img_info["id"], [])

        bboxes = []
        cat_ids = []

        # === PRE-PROCESSING: Normalize Check ===
        for a in anns:
            box = a["bbox"]
            # If normalized, convert to pixels for Albumentations
            if is_normalized(box):
                box = denormalize_bbox(box, w_orig, h_orig)

            # Clamp to image boundaries
            box = clean_bbox(box, w_orig, h_orig)

            if box[2] > 1 and box[3] > 1:  # width/height > 1 pixel
                bboxes.append(box)
                cat_ids.append(a["category_id"])

        for i in range(NUM_AUGS):
            try:
                augmented = transform(image=image, bboxes=bboxes, category_ids=cat_ids)

                if len(bboxes) > 0 and len(augmented["bboxes"]) == 0: continue

                # Save Image
                aug_filename = f"aug_{i}_{img_info['file_name']}"
                aug_path = os.path.join(TRAIN_DIR, aug_filename)
                cv2.imwrite(aug_path, cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR))

                # New Image Record
                new_img = img_info.copy()
                new_img["id"] = img_id_counter
                new_img["file_name"] = aug_filename
                new_img["width"] = augmented["image"].shape[1]
                new_img["height"] = augmented["image"].shape[0]
                new_images.append(new_img)

                # Save Annotations (Always as Pixels for consistency)
                for box, cid in zip(augmented["bboxes"], augmented["category_ids"]):
                    new_annotations.append({
                        "id": ann_id_counter,
                        "image_id": img_id_counter,
                        "category_id": cid,
                        "bbox": list(box),  # Stored as pixels
                        "area": box[2] * box[3],
                        "iscrowd": 0,
                        "segmentation": []
                    })
                    ann_id_counter += 1
                img_id_counter += 1

            except Exception as e:
                pass

    coco["images"].extend(new_images)
    coco["annotations"].extend(new_annotations)

    with open(json_path, "w") as f:
        json.dump(coco, f)

    print(f"\n✅ Done. Added {len(new_images)} images.")


def generate_verification_grid():
    print("🔍 Generating verification grid...")
    json_path = os.path.join(TRAIN_DIR, JSON_FILE)
    with open(json_path, "r") as f:
        coco = json.load(f)

    aug_imgs = [i for i in coco["images"] if "aug_" in i["file_name"]]
    if not aug_imgs: return

    samples = random.sample(aug_imgs, min(9, len(aug_imgs)))
    ann_map = {img["id"]: [] for img in samples}

    # Filter only relevant annotations
    sample_ids = set(img["id"] for img in samples)
    for ann in coco["annotations"]:
        if ann["image_id"] in sample_ids:
            ann_map[ann["image_id"]].append(ann)

    grid_images = []
    for img_info in samples:
        path = os.path.join(TRAIN_DIR, img_info["file_name"])
        img = cv2.imread(path)
        if img is None: continue

        img_h, img_w = img.shape[:2]

        for ann in ann_map[img_info["id"]]:
            bbox = ann["bbox"]

            # VISUALIZATION FIX: Handle Normalized vs Pixel
            if is_normalized(bbox):
                x, y, w, h = denormalize_bbox(bbox, img_w, img_h)
            else:
                x, y, w, h = bbox

            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        grid_images.append(cv2.resize(img, (512, 512)))

    while len(grid_images) < 9: grid_images.append(np.zeros_like(grid_images[0]))

    row1 = np.hstack(grid_images[0:3])
    row2 = np.hstack(grid_images[3:6])
    row3 = np.hstack(grid_images[6:9])
    grid = np.vstack([row1, row2, row3])

    out = "/workspace/verification_grid.jpg"
    cv2.imwrite(out, grid)
    print(f"✅ Verified: {out}")


if __name__ == "__main__":
    validate_structure()
    augment_in_place()
    generate_verification_grid()