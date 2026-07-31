"""Configuration helpers for the RSF-DETR train, validation and prediction API."""

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union

from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    RANK,
    ROOT,
    RUNS_DIR,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    LOGGER,
    colorstr,
    yaml_load,
)

TASK2DATA = {"detect": "dataset/PipeSafeDataset.yaml"}

CFG_FLOAT_KEYS = ("box", "cls", "dfl", "degrees", "shear")
CFG_FRACTION_KEYS = (
    "dropout",
    "iou",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "label_smoothing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "conf",
    "fraction",
)
CFG_INT_KEYS = (
    "epochs",
    "patience",
    "batch",
    "workers",
    "seed",
    "close_mosaic",
    "max_det",
    "vid_stride",
    "line_width",
    "nbs",
    "save_period",
    "warmup_iters",
)
CFG_BOOL_KEYS = (
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "boxes",
)


def cfg2dict(cfg):
    """Convert a path or namespace to a configuration dictionary."""
    if isinstance(cfg, (str, Path)):
        return yaml_load(cfg)
    if isinstance(cfg, SimpleNamespace):
        return vars(cfg)
    return cfg


def check_dict_alignment(base: Dict, custom: Dict):
    """Reject configuration keys outside the shipped RSF-DETR schema."""
    mismatched = sorted(set(custom) - set(base))
    if mismatched:
        invalid = ", ".join(colorstr("red", "bold", key) for key in mismatched)
        raise SyntaxError(f"Unsupported RSF-DETR configuration argument(s): {invalid}")


def get_cfg(
    cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT,
    overrides: Dict = None,
):
    """Merge and validate RSF-DETR runtime arguments."""
    cfg = dict(cfg2dict(cfg))
    if overrides:
        overrides = dict(cfg2dict(overrides))
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)
        check_dict_alignment(cfg, overrides)
        cfg.update(overrides)

    for key in ("project", "name"):
        if key in cfg and isinstance(cfg[key], (int, float)):
            cfg[key] = str(cfg[key])
    if cfg.get("name") == "model":
        cfg["name"] = str(cfg.get("model", "")).split(".")[0]
        LOGGER.warning(f"Updating run name to '{cfg['name']}'.")

    for key, value in cfg.items():
        if value is None:
            continue
        if key in CFG_FLOAT_KEYS and not isinstance(value, (int, float)):
            raise TypeError(f"'{key}' must be numeric.")
        if key in CFG_FRACTION_KEYS:
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"'{key}' must be a number between 0 and 1.")
        if key in CFG_INT_KEYS and not isinstance(value, int):
            raise TypeError(f"'{key}' must be an integer.")
        if key in CFG_BOOL_KEYS and not isinstance(value, bool):
            raise TypeError(f"'{key}' must be a boolean.")
    return IterableSimpleNamespace(**cfg)


def get_save_dir(args, name=None):
    """Resolve the output directory for a train, validation or prediction run."""
    if getattr(args, "save_dir", None):
        return Path(args.save_dir)

    from ultralytics.utils.files import increment_path

    project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
    run_name = name or args.name or args.mode
    return Path(increment_path(Path(project) / run_name, exist_ok=args.exist_ok if RANK in (-1, 0) else True))
