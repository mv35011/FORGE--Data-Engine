import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

# --- IMPORTS FROM YOUR EXISTING PROJECT STRUCTURE ---
# We assume standard RF-DETR folder structure exists
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import train_one_epoch, evaluate
from models import build_model


# --- HARDCODED CONFIG (Based on your logs) ---
class Config:
    def __init__(self):
        # Paths
        self.output_dir = "rfdetr_output"
        self.coco_path = "dataset/ppe_coco_format"
        self.dataset_file = "coco"
        self.resume = "rfdetr_output/checkpoint0006.pth"  # <--- YOUR SPECIFIC CHECKPOINT

        # Hardware
        self.device = "cuda"
        self.seed = 42
        self.num_workers = 2
        self.world_size = 1
        self.dist_url = "env://"
        self.distributed = False

        # Model Params (From your logs)
        self.num_classes = 10
        self.backbone = "dinov2_windowed_small"
        self.num_queries = 300
        self.pretrain_weights = None  # We resume, so no pretrain needed

        # Hyperparameters
        self.lr = 0.0002
        self.batch_size = 4
        self.weight_decay = 0.0001
        self.epochs = 25  # Total target epochs
        self.start_epoch = 0  # Will be overwritten by checkpoint
        self.clip_max_norm = 0.1

        # Loss Coefficients
        self.mask_loss_coef = 1
        self.dice_loss_coef = 1
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.1
        self.cls_loss_coef = 1.0

        # Model toggles (Standard RF-DETR defaults)
        self.masks = False
        self.aux_loss = True
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.dim_feedforward = 2048
        self.enc_layers = 6
        self.dec_layers = 6
        self.pre_norm = False

        # DINOv2 Specifics (Likely defaults)
        self.use_checkpoint = False


def main():
    args = Config()
    utils.init_distributed_mode(args)
    print(f"🚀 Starting Emergency Fine-Tune Resume on {args.device}")

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 1. Build Model
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model loaded. Number of params: {n_parameters}")

    # 2. Build Optimizer (AdamW standard)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr * 0.1,  # Backbone usually has lower LR
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Simple LR Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 3. Build Dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)

    data_loader_val = DataLoader(dataset_val, args.batch_size, drop_last=False,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # 4. LOAD CHECKPOINT (The Critical Part)
    print(f"📥 Loading Checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')

    model_without_ddp.load_state_dict(checkpoint['model'])

    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Successfully resumed optimizer and scheduler. Starting from Epoch {args.start_epoch}")

    # 5. Training Loop
    print("🔥 Beginning Training Loop...")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm
        )

        lr_scheduler.step()

        # Save Checkpoint
        checkpoint_path = Path(args.output_dir) / f'checkpoint{epoch:04}.pth'
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

        # Evaluate
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, dataset_val, device, args.output_dir
        )

        print(f"Epoch {epoch} complete. Stats: {train_stats}")

    total_time = time.time() - start_time
    print(f"🏁 Training complete in {str(datetime.timedelta(seconds=int(total_time)))}")


if __name__ == '__main__':
    main()