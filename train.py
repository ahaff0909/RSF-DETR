#!/usr/bin/env python3
"""Unified training entry for RSF-DETR.

Defaults follow the recorded run settings:
  AdamW, lr0=1e-4, weight_decay=1e-4, epochs=300, batch=16,
  imgsz=640, 2000 warm-up iterations, mosaic=0, patience=30.

Pass --weights weights/rtdetr-r18.pt for the pretrained initialization used
by the full-model experiments. Without --weights the model trains from scratch.
"""

from __future__ import annotations

import argparse

from ultralytics import RSFDETR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RSF-DETR and its paper ablations.")
    p.add_argument(
        "--config",
        default="configs/rsf-detr.yaml",
        help="Model YAML path.",
    )
    p.add_argument("--data", default="dataset/PipeSafeDataset.yaml", help="Dataset YAML path.")
    p.add_argument("--project", default="runs/train", help="Project directory for logs/weights.")
    p.add_argument("--name", default="rsf_detr", help="Run name.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0, help="Random seed (paper uses a fixed seed).")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    p.add_argument("--cache", action="store_true", help="Cache images in RAM/disk.")
    p.add_argument(
        "--weights",
        default="",
        help="Initialization checkpoint. Use weights/rtdetr-r18.pt for paper runs; empty trains from scratch.",
    )
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=0.0001)
    p.add_argument("--lrf", type=float, default=1.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument(
        "--warmup-iters",
        type=int,
        default=2000,
        help="Number of optimizer warm-up iterations.",
    )
    p.add_argument("--mosaic", type=float, default=0.0, help="Mosaic probability used by the trainer.")
    p.add_argument("--cos-lr", action="store_true", help="Enable cosine LR.")
    p.add_argument("--device", default="", help="CUDA device, e.g. 0 or 0,1. Empty = auto.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = RSFDETR(args.config)
    if args.weights:
        model.load(args.weights)
    else:
        print("WARNING: no initialization checkpoint supplied; training from scratch.")

    train_kw = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        amp=args.amp,
        cache=args.cache,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        mosaic=args.mosaic,
        cos_lr=args.cos_lr,
        seed=args.seed,
    )
    if args.device != "":
        train_kw["device"] = args.device

    model.train(**train_kw)


if __name__ == "__main__":
    main()
