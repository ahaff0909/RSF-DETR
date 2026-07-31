#!/usr/bin/env python3
"""Validation / metric reporting entry for RSF-DETR checkpoints."""

from __future__ import annotations

import argparse
import os

from ultralytics import RSFDETR
from ultralytics.utils.torch_utils import model_info


def get_weight_size_mb(path: str) -> str:
    return f"{os.stat(path).st_size / 1024 / 1024:.1f}"


def print_table(title, headers, rows):
    """Print a compact dependency-free table."""
    rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    line = "  ".join(f"{{:<{width}}}" for width in widths)
    print(f"\n{title}")
    print(line.format(*headers))
    print(line.format(*("-" * width for width in widths)))
    for row in rows:
        print(line.format(*row))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate an RSF-DETR / RT-DETR checkpoint.")
    p.add_argument("--weights", required=True, help="Path to best.pt / last.pt.")
    p.add_argument("--data", default="dataset/PipeSafeDataset.yaml", help="Dataset YAML.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--project", default="runs/val")
    p.add_argument("--name", default="exp")
    p.add_argument("--device", default="", help="CUDA device id; empty = auto.")
    p.add_argument("--conf", type=float, default=0.001, help="Confidence threshold (paper: 0.001).")
    p.add_argument("--save-json", action="store_true", help="Save COCO-format predictions when supported.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = RSFDETR(args.weights)
    val_kw = dict(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        conf=args.conf,
        save_json=args.save_json,
        exist_ok=True,
    )
    if args.device != "":
        val_kw["device"] = args.device

    result = model.val(**val_kw)

    if model.task != "detect":
        return

    n_l, n_p, n_g, flops = model_info(model.model)
    prep = result.speed["preprocess"]
    infer = result.speed["inference"]
    post = result.speed["postprocess"]
    all_t = prep + infer + post

    info_headers = [
        "GFLOPs",
        "Parameters",
        "Pre(ms)",
        "Infer(ms)",
        "Post(ms)",
        "FPS(all)",
        "FPS(infer)",
        "Weight(MB)",
    ]
    info_rows = [[
            f"{flops:.1f}",
            f"{n_p:,}",
            f"{prep:.2f}",
            f"{infer:.2f}",
            f"{post:.2f}",
            f"{1000 / all_t:.2f}" if all_t > 0 else "n/a",
            f"{1000 / infer:.2f}" if infer > 0 else "n/a",
            get_weight_size_mb(args.weights),
        ]]
    print_table("Model Info", info_headers, info_rows)

    metric_headers = ["Class", "P", "R", "F1", "mAP50", "mAP75", "mAP50-95"]
    metric_rows = []
    names = result.names
    class_ids = [int(x) for x in result.box.ap_class_index]
    for metric_i, class_id in enumerate(class_ids):
        p_i, r_i = float(result.box.p[metric_i]), float(result.box.r[metric_i])
        f1 = 2 * p_i * r_i / (p_i + r_i + 1e-16)
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        metric_rows.append(
            [
                class_name,
                f"{p_i:.4f}",
                f"{r_i:.4f}",
                f"{f1:.4f}",
                f"{float(result.box.all_ap[metric_i, 0]):.4f}",
                f"{float(result.box.all_ap[metric_i, 5]):.4f}",
                f"{float(result.box.ap[metric_i]):.4f}",
            ]
        )
    # Prefer aggregate if available
    try:
        print(
            f"mAP50={float(result.box.map50):.4f}  "
            f"mAP75={float(result.box.map75):.4f}  "
            f"mAP50-95={float(result.box.map):.4f}  "
            f"AP_S/M/L require COCO area-range evaluation and are not emitted by this table"
        )
    except Exception:
        pass
    print_table("Detection Metrics", metric_headers, metric_rows)


if __name__ == "__main__":
    main()
