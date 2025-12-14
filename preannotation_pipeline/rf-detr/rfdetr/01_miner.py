import os
import sys
import cv2
import json
import zipfile
import shutil
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

# ================= PATH FIXES =================
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from rfdetr import RFDETRBase

# ================= HARDCODED PATHS =================
YOLO_PATH = r"D:\Compressed\pycharm projects\fire_ppe_personID_IndustrialSystem\models\ppe_detection_10class_10epochs.pt"
RFDETR_PATH = r"D:\Compressed\pycharm projects\fire_ppe_personID_IndustrialSystem\models\checkpoint_best_refdetr_ppe.pth"
INPUT_ZIP_PATH = r"D:\Compressed\pycharm projects\active_learning_objectdetection_annotation_pipeline\preannotation_pipeline\videos\rawdata.zip"

# ================= CONFIG =================
CLASSES = [
    'boots', 'gloves', 'goggles', 'helmet',
    'no-boots', 'no-gloves', 'no-goggles', 'no-helmet',
    'no-vest', 'vest'
]


def get_args():
    parser = argparse.ArgumentParser("RF-DETR + YOLO Miner")
    parser.add_argument("--output_dir", default="outputs/mining_results")
    parser.add_argument("--sample_rate", type=int, default=30)
    parser.add_argument("--yolo_thresh", type=float, default=0.6)
    parser.add_argument("--rfdetr_thresh", type=float, default=0.4)
    return parser.parse_args()


# ================= LOAD MODELS =================
def load_rfdetr(weights_path):
    print(f"🔹 Loading RF-DETR from {weights_path}")
    model = RFDETRBase(
        num_classes=10,
        size="base",
        resolution=672,
        pretrain_weights=weights_path
    )
    return model



# ================= RF-DETR INFERENCE =================
def infer_rfdetr(model, pil_image):
    with torch.no_grad():
        detections = model.predict(pil_image)

    # RF-DETR Detections API (NOT YOLO, NOT torchvision)
    boxes = detections.xyxy
    scores = detections.confidence
    labels = detections.class_id

    if boxes is None or len(scores) == 0:
        return 0.0, []

    max_conf = float(scores.max().item())

    results = []
    for box, score, label in zip(boxes, scores, labels):
        results.append({
            "label": CLASSES[int(label)] if int(label) < len(CLASSES) else "unknown",
            "score": float(score.item()),
            "bbox": box.tolist()  # xyxy absolute pixels
        })

    return max_conf, results

# ================= VIDEO PROCESS =================
def process_video(
    video_path,
    yolo_model,
    rfdetr_model,
    args,
    frames_dir,
    proposals
):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    video_name = Path(video_path).stem

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % args.sample_rate != 0:
            continue

        # ---------- YOLO ----------
        yolo_res = yolo_model(frame, verbose=False)[0]
        yolo_conf = float(yolo_res.boxes.conf.max()) if len(yolo_res.boxes) else 0.0

        if yolo_conf < args.yolo_thresh:
            continue

        # ---------- RF-DETR ----------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        rf_conf, rf_dets = infer_rfdetr(rfdetr_model, pil_img)

        if rf_conf < args.rfdetr_thresh:
            continue

        # ---------- SAVE ----------
        fname = f"{video_name}_f{frame_idx:06d}.jpg"
        out_path = frames_dir / fname
        cv2.imwrite(str(out_path), frame)

        proposals[fname] = {
            "image_height": frame.shape[0],
            "image_width": frame.shape[1],
            "scores": {
                "yolo": yolo_conf,
                "rfdetr": rf_conf
            },
            "annotations": rf_dets
        }

    cap.release()


# ================= MAIN =================
def main():
    args = get_args()

    print("🚀 Starting Miner")

    if not os.path.exists(YOLO_PATH):
        raise FileNotFoundError("YOLO weights not found")
    if not os.path.exists(RFDETR_PATH):
        raise FileNotFoundError("RF-DETR weights not found")
    if not os.path.exists(INPUT_ZIP_PATH):
        raise FileNotFoundError("Input zip not found")

    yolo_model = YOLO(YOLO_PATH)
    rfdetr_model = load_rfdetr(RFDETR_PATH)

    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    temp_dir = out_dir / "temp"

    frames_dir.mkdir(parents=True, exist_ok=True)

    if not temp_dir.exists():
        with zipfile.ZipFile(INPUT_ZIP_PATH, "r") as z:
            z.extractall(temp_dir)

    videos = list(temp_dir.glob("**/*.mp4"))
    print(f"📼 Found {len(videos)} videos")

    proposals = {}

    for vid in tqdm(videos, desc="Mining"):
        process_video(
            vid,
            yolo_model,
            rfdetr_model,
            args,
            frames_dir,
            proposals
        )

    json_path = out_dir / "proposals.json"
    with open(json_path, "w") as f:
        json.dump(proposals, f, indent=2)

    shutil.rmtree(temp_dir)

    print(f"✅ Done. Saved {len(proposals)} frames")
    print(f"📄 Proposals → {json_path}")


if __name__ == "__main__":
    main()
