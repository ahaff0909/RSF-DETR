#!/usr/bin/env python3
"""Print model parameter count / GFLOPs for a paper YAML."""

from ultralytics import RSFDETR

if __name__ == "__main__":
    # Default: full RSF-DETR config
    cfg = "configs/rsf-detr.yaml"
    model = RSFDETR(cfg)
    model.info(detailed=False)
