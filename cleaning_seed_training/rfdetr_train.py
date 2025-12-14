import torch
import os
import random
import glob
import numpy as np
from rfdetr import RFDETRBase

DATASET_DIR = "/workspace/dataset/ppe_coco_format"
# Ensure this is on the persistent volume
OUTPUT_DIR = "/workspace/rfdetr_output"


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # For deterministic behavior on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_dataset_structure():
    req = ["train", "valid"]
    for split in req:
        p = os.path.join(DATASET_DIR, split)
        if not os.path.exists(p):
            raise FileNotFoundError(f"❌ Missing dataset split: {p}")
    print("✅ Dataset structure valid.")


def find_latest_checkpoint(output_dir):
    """
    Scans output directory for the latest checkpoint to auto-resume.
    Assumes checkpoints are named like 'checkpoint_epoch_*.pt' or 'last.pt'
    """
    if not os.path.exists(output_dir):
        return None

    # Standard RF-DETR/YOLO usually saves a 'last.pt' or 'last.pth'
    last_ckpt = os.path.join(output_dir, "weights", "last.pt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    # If not, look for epoch based checkpoints
    checkpoints = glob.glob(os.path.join(output_dir, "*.pt"))
    if not checkpoints:
        return None

    # Sort by modification time to get the newest one
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    return latest_ckpt


def main():
    set_seed()
    validate_dataset_structure()

    device_name = torch.cuda.get_device_name(0)
    print(f"🔥 Starting Training on {device_name}")
    print(f"📂 Output Directory: {OUTPUT_DIR}")

    # Initialize Output Dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Check for Auto-Resume
    resume_path = find_latest_checkpoint(OUTPUT_DIR)
    if resume_path:
        print(f"♻️  Found existing checkpoint! Resuming from: {resume_path}")
    else:
        print("🆕 No checkpoint found. Starting fresh training.")

    model = RFDETRBase(
        size="large",
        resolution=896,
        gradient_checkpointing=True
    )

    try:
        history = model.train(
            dataset_dir=DATASET_DIR,
            output_dir=OUTPUT_DIR,
            epochs=25,
            batch_size=4,  # WARNING: 896px is heavy. If OOM, reduce to 2.
            grad_accum_steps=48,  # Effective batch = 4 * 48 = 192
            lr=2e-4,
            warmup_epochs=2,
            lr_scheduler="cosine",
            weight_decay=1e-4,
            optimizer="adamw",
            clip_grad_norm=1.0,
            fp16=True,
            num_workers=8,
            checkpoint_interval=1,
            save_best_model=True,
            early_stopping=False,
            resume=resume_path  # Pass the auto-detected path here
        )
        print("🏆 Training Complete.")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ CUDA OUT OF MEMORY ERROR ❌")
            print("Your resolution (896) or Batch Size (4) is too high.")
            print("Try reducing batch_size to 2 or resolution to 640.")
        else:
            raise e


if __name__ == "__main__":
    main()