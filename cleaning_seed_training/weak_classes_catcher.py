import os
import shutil
import cv2
import torch
import numpy as np
import supervision as sv
from tqdm import tqdm
from rfdetr import RFDETRBase

# --- CONFIGURATION ---
DATASET_IMAGES = "/workspace/dataset/ppe_coco_format/train"  # Path to images
DATASET_LABELS = "/workspace/dataset/ppe_coco_format/train"  # Path to YOLO txt labels
CHECKPOINT_PATH = "/workspace/rfdetr_output/best_model.pth"  # Your trained model

OUTPUT_DIR = "/workspace/dataset_audit_results"

# Classes relevant to your PPE dataset
CLASS_NAMES = [
    'boots', 'gloves', 'goggles', 'helmet', 'no-boots',
    'no-gloves', 'no-goggles', 'no-helmet', 'no-vest', 'vest'
]

# Audit Settings
CONF_THRESHOLD = 0.50  # Only trust model if conf is higher than this
IOU_THRESHOLD = 0.6  # Intersection over Union to match GT with Pred
TARGET_WEAK_CLASSES = []  # Leave empty [] to check ALL classes.


# Set ['gloves', 'boots'] to only hunt for these missing labels.

def load_yolo_labels(label_path, image_shape):
    """Parses a YOLO txt file and returns xyxy coordinates and class_ids."""
    if not os.path.exists(label_path):
        return np.array([]), np.array([])

    with open(label_path, 'r') as f:
        lines = f.readlines()

    boxes = []
    class_ids = []
    h, w, _ = image_shape

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            boxes.append([x1, y1, x2, y2])
            class_ids.append(cls_id)

    return np.array(boxes), np.array(class_ids)


def main():
    # 1. Setup Model
    print("Loading Trained Model...")
    model = RFDETRBase(
        size="large",  # Must match your training config
        resolution=896,  # Must match your training config
        pretrain_weights=CHECKPOINT_PATH,
        num_classes=len(CLASS_NAMES)
    )
    model.optimize_for_inference()

    # 2. Setup Output Directories
    dirs = {
        "missing_label": os.path.join(OUTPUT_DIR, "1_potential_missing_labels"),
        "mislabel": os.path.join(OUTPUT_DIR, "2_potential_mislabeled_class"),
        "clean": os.path.join(OUTPUT_DIR, "0_clean_verified")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        os.makedirs(os.path.join(d, "visualizations"), exist_ok=True)

    # 3. List Images
    image_files = [f for f in os.listdir(DATASET_IMAGES) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Starting Audit on {len(image_files)} images...")

    # Annotators for visualization
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5)

    for img_file in tqdm(image_files):
        img_path = os.path.join(DATASET_IMAGES, img_file)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(DATASET_LABELS, label_file)

        image = cv2.imread(img_path)
        if image is None: continue

        # --- A. Ground Truth (GT) ---
        gt_boxes_xyxy, gt_class_ids = load_yolo_labels(label_path, image.shape)

        # Create supervision object for GT
        if len(gt_boxes_xyxy) > 0:
            gt_detections = sv.Detections(xyxy=gt_boxes_xyxy, class_id=gt_class_ids)
        else:
            gt_detections = sv.Detections.empty()

        # --- B. Model Prediction ---
        with torch.no_grad():
            # Standard inference. For 'ultra' CCTV, consider tiling logic here if predictions are poor.
            results = model.predict(image, threshold=CONF_THRESHOLD)

        # Filter by weak classes if requested
        if TARGET_WEAK_CLASSES:
            target_ids = [CLASS_NAMES.index(c) for c in TARGET_WEAK_CLASSES if c in CLASS_NAMES]
            mask = np.isin(results.class_id, target_ids)
            results = results[mask]

        # --- C. Compare (The Logic) ---
        has_issue = False
        issue_type = "clean"

        # Case 1: Missing Label (Model sees it, GT does not)
        # We check if any prediction has very low IoU with ALL GT boxes
        if len(results) > 0:
            if len(gt_detections) == 0:
                # Model found something, GT is empty -> DEFINITELY Missing Label
                issue_type = "missing_label"
                has_issue = True
            else:
                # Calculate IoU between all Preds and all GTs
                iou_matrix = sv.box_iou_batch(results.xyxy, gt_detections.xyxy)
                # Max IoU for each prediction (did this prediction match ANY GT?)
                max_ious = np.max(iou_matrix, axis=1)

                # If any prediction overlaps 0.0 (or very low) with GT, it's a new object
                if np.any(max_ious < 0.1):
                    issue_type = "missing_label"
                    has_issue = True

        # Case 2: Mislabel (Intersection is high, but Class ID is different)
        # Only check if we haven't already flagged it as missing
        if not has_issue and len(results) > 0 and len(gt_detections) > 0:
            iou_matrix = sv.box_iou_batch(results.xyxy, gt_detections.xyxy)

            # Iterate through matches
            for i, pred_class in enumerate(results.class_id):
                # Find best matching GT
                best_gt_idx = np.argmax(iou_matrix[i])
                best_iou = iou_matrix[i][best_gt_idx]

                if best_iou > IOU_THRESHOLD:
                    gt_class = gt_detections.class_id[best_gt_idx]
                    if pred_class != gt_class:
                        # Conflict! Model says Helmet, GT says No-Helmet
                        issue_type = "mislabel"
                        has_issue = True
                        break

        # --- D. Save Results ---
        dest_folder = dirs[issue_type]

        # Copy original image
        shutil.copy(img_path, os.path.join(dest_folder, img_file))

        # Create Visualization (Split View or Overlay)
        # We will draw GT in Green and Pred in Red
        annotated_img = image.copy()

        # Draw GT (Green)
        if len(gt_detections) > 0:
            annotated_img = box_annotator.annotate(annotated_img, gt_detections)
            labels_gt = [f"GT: {CLASS_NAMES[cid]}" for cid in gt_detections.class_id]
            annotated_img = label_annotator.annotate(annotated_img, gt_detections, labels=labels_gt)

        # Draw Pred (Red/Blue - supervision handles colors, but we label clearly)
        if len(results) > 0:
            annotated_img = box_annotator.annotate(annotated_img, results)
            labels_pred = [f"Pred: {CLASS_NAMES[cid]} {conf:.2f}" for cid, conf in
                           zip(results.class_id, results.confidence)]
            annotated_img = label_annotator.annotate(annotated_img, results, labels=labels_pred)

        cv2.imwrite(os.path.join(dest_folder, "visualizations", img_file), annotated_img)


if __name__ == "__main__":
    main()