import argparse
import os
import zipfile
import shutil
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# --- DEPENDENCIES ---
from ultralytics import YOLO
import torchvision.transforms as T

# Placeholder for RF-DETR imports
try:
    from models import build_model
    from util.misc import nested_tensor_from_tensor_list
except ImportError:
    print("⚠️ RF-DETR modules not found. Ensure you are running this from the repo root.")

# ================= CONFIGURATION =================
# RF-DETR Standard Transforms
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASSES = [
    'boots', 'gloves', 'goggles', 'helmet',
    'no-boots', 'no-gloves', 'no-goggles', 'no-helmet',
    'no-vest', 'vest'
]


def get_args():
    parser = argparse.ArgumentParser(description="Miner: Weighted Consensus Extraction")
    parser.add_argument("--input_zip", type=str, required=True, help="Path to zip file")
    parser.add_argument("--output_dir", type=str, default="outputs/mining_results", help="Output directory")
    parser.add_argument("--yolo_weights", type=str, required=True, help="Path to YOLO .pt file")
    parser.add_argument("--rfdetr_weights", type=str, required=True, help="Path to RF-DETR .pth file")
    parser.add_argument("--sample_rate", type=int, default=30, help="Process every Nth frame")

    # Selection Hyperparameters
    parser.add_argument("--min_score", type=float, default=0.5, help="Minimum Combined Score to keep frame")
    parser.add_argument("--yolo_weight", type=float, default=0.7, help="Weight for YOLO confidence (0.0 - 1.0)")

    return parser.parse_args()


def load_rfdetr(weights_path, device):
    """Load RF-DETR model"""

    class Args:
        backbone = 'dinov2_windowed_small'  # <--- UPDATE THIS if using ResNet
        masks = False
        dilation = False
        position_embedding = 'sine'
        hidden_dim = 256
        enc_layers = 6
        dec_layers = 6
        dim_feedforward = 2048
        dropout = 0.1
        nheads = 8
        num_queries = 300
        pre_norm = False
        aux_loss = True
        set_cost_class = 1
        set_cost_bbox = 5
        set_cost_giou = 2
        lr_backbone = 1e-5
        device = device

    args = Args()
    try:
        model, _, _ = build_model(args)
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Failed to load RF-DETR: {e}")
        return None


def infer_rfdetr(model, image, device):
    """Run RF-DETR and return max confidence and detections"""
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    # Get probabilities
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]

    # 1. Get Max Confidence for the entire frame (for scoring)
    # We take the highest confidence score of any object in the frame
    if probas.numel() > 0:
        max_rf_conf = probas.max().item()
    else:
        max_rf_conf = 0.0

    # 2. Get Annotations (only keep decent ones for JSON)
    keep = probas.max(-1).values > 0.4
    bboxes_scaled = outputs['pred_boxes'][0, keep]

    results = []
    for p, b in zip(probas[keep], bboxes_scaled):
        cl = p.argmax()
        conf = p[cl]
        results.append({
            "label": CLASSES[cl.item()] if cl.item() < len(CLASSES) else "unknown",
            "score": float(conf.item()),
            "bbox": b.tolist()
        })

    return max_rf_conf, results


def process_video(video_path, yolo_model, rfdetr_model, args, output_frames_dir, proposals_data):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    video_name = Path(video_path).stem
    device = next(rfdetr_model.parameters()).device

    # Normalize weights
    w_yolo = args.yolo_weight
    w_rf = 1.0 - w_yolo

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % args.sample_rate != 0:
            continue

        # --- STEP 1: YOLO OPINION ---
        yolo_results = yolo_model(frame, verbose=False)[0]
        if len(yolo_results.boxes) > 0:
            yolo_conf = yolo_results.boxes.conf.max().item()
        else:
            yolo_conf = 0.0

        # Optimization: If YOLO is effectively 0, RF-DETR alone needs to be VERY confident to pass.
        # If combined score relies heavily on YOLO, we might skip RF-DETR if YOLO is 0 to save time.
        # But per your request, we check both.

        # --- STEP 2: RF-DETR OPINION ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        rf_conf, rf_detections = infer_rfdetr(rfdetr_model, pil_img, device)

        # --- STEP 3: THE CONSENSUS ---
        combined_score = (yolo_conf * w_yolo) + (rf_conf * w_rf)

        # Debug print (optional)
        # print(f"Frame {frame_count}: Y={yolo_conf:.2f} R={rf_conf:.2f} -> Final={combined_score:.2f}")

        if combined_score < args.min_score:
            continue  # REJECTED

        # --- STEP 4: ACCEPT & STORE ---
        filename = f"{video_name}_f{frame_count:06d}.jpg"
        save_path = output_frames_dir / filename

        cv2.imwrite(str(save_path), frame)

        proposals_data[filename] = {
            "image_height": frame.shape[0],
            "image_width": frame.shape[1],
            "scores": {
                "combined": combined_score,
                "yolo": yolo_conf,
                "rfdetr": rf_conf
            },
            "annotations": rf_detections  # We still only save RF-DETR boxes as "Pre-annotations"
        }

    cap.release()


def main():
    args = get_args()

    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    temp_extract_dir = out_dir / "temp_extracted"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Weighted Miner starting on {device}")
    print(f"   Policy: {args.yolo_weight}*YOLO + {1.0 - args.yolo_weight}*RF-DETR >= {args.min_score}")

    # Load Models
    print("🔹 Loading YOLO...")
    yolo_model = YOLO(args.yolo_weights)

    print("🔹 Loading RF-DETR...")
    rfdetr_model = load_rfdetr(args.rfdetr_weights, device)
    if rfdetr_model is None: return

    # Extract Zip
    if not temp_extract_dir.exists():
        print(f"📦 Extracting {args.input_zip}...")
        with zipfile.ZipFile(args.input_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)
    else:
        print("⚠️ Temp dir exists, skipping extraction (clean it if needed)")

    video_files = list(temp_extract_dir.glob("**/*.mp4"))

    proposals_data = {}

    for video in tqdm(video_files, desc="Mining Videos"):
        process_video(video, yolo_model, rfdetr_model, args, frames_dir, proposals_data)

    # Save
    json_path = out_dir / "proposals.json"
    with open(json_path, 'w') as f:
        json.dump(proposals_data, f, indent=2)

    # Cleanup
    if temp_extract_dir.exists():
        shutil.rmtree(temp_extract_dir)

    print(f"✅ Mining Complete. Saved to {frames_dir}")


if __name__ == "__main__":
    main()