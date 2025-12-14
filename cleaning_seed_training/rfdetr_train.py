import argparse
import datetime
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import util.misc as utils
from datasets import build_dataset
from engine import train_one_epoch, evaluate
from models import build_model


# ----------------------------
# ARGUMENTS (REAL Namespace)
# ----------------------------
def get_args():
    parser = argparse.ArgumentParser("RF-DETR Finetuning (RunPod Safe)")

    # Paths
    parser.add_argument("--coco_path", default="dataset/ppe_coco_format")
    parser.add_argument("--output_dir", default="rfdetr_output")
    parser.add_argument("--resume", default="")

    # Training
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_max_norm", type=float, default=0.1)

    # Model
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--backbone", default="dinov2_windowed_small")
    parser.add_argument("--num_queries", type=int, default=300)

    # System
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)  # IMPORTANT for RunPod

    return parser.parse_args()


def main():
    args = get_args()
    utils.init_distributed_mode(args)

    print(f"🚀 Starting RF-DETR Finetuning on {args.device}")
    device = torch.device(args.device)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ----------------------------
    # Model
    # ----------------------------
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Trainable params: {n_params}")

    # ----------------------------
    # Optimizer
    # ----------------------------
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": args.lr * 0.1,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.1
    )

    # ----------------------------
    # Dataset
    # ----------------------------
    dataset_train = build_dataset("train", args)
    dataset_val = build_dataset("val", args)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        pin_memory=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=utils.collate_fn,
        pin_memory=True,
    )

    # ----------------------------
    # Resume (SAFE)
    # ----------------------------
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f"📥 Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")

        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1

        print(f"✅ Resume OK. Starting from epoch {start_epoch}")

    # ----------------------------
    # Training Loop
    # ----------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, args.clip_max_norm
        )

        lr_scheduler.step()

        # Save checkpoint every epoch (RunPod-safe)
        ckpt_path = output_dir / f"checkpoint_{epoch:04}.pth"
        utils.save_on_master({
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }, ckpt_path)

        evaluate(
            model, criterion, postprocessors,
            data_loader_val, dataset_val,
            device, output_dir
        )

        print(f"✅ Epoch {epoch} complete")

    total_time = time.time() - start_time
    print(f"🏁 Training finished in {datetime.timedelta(seconds=int(total_time))}")


if __name__ == "__main__":
    main()
