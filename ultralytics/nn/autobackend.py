"""PyTorch inference backend used by RSF-DETR."""

from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.utils import yaml_load


def default_class_names(data=None):
    """Return class names from a dataset YAML or a numerical fallback."""
    if data:
        try:
            names = yaml_load(data)["names"]
            return dict(enumerate(names)) if isinstance(names, list) else names
        except (FileNotFoundError, KeyError, TypeError):
            pass
    return {i: f"class{i}" for i in range(999)}


def check_class_names(names):
    """Normalize a class-name list or dictionary."""
    if isinstance(names, list):
        names = dict(enumerate(names))
    if not isinstance(names, dict) or not names:
        raise TypeError("Class names must be a non-empty list or dictionary.")
    names = {int(key): str(value) for key, value in names.items()}
    if min(names) < 0 or max(names) >= len(names):
        raise KeyError(f"Class indices must be contiguous from 0 to {len(names) - 1}.")
    return names


class AutoBackend(nn.Module):
    """Run RSF-DETR inference from an in-memory model or a PyTorch checkpoint."""

    @torch.no_grad()
    def __init__(
        self,
        weights,
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        fuse=True,
        verbose=True,
    ):
        super().__init__()
        if dnn:
            raise ValueError("RSF-DETR only ships the PyTorch inference backend.")

        if isinstance(weights, nn.Module):
            model = weights.to(device)
            if fuse and hasattr(model, "fuse"):
                model = model.fuse(verbose=verbose)
        else:
            weight_path = Path(weights[0] if isinstance(weights, list) else weights)
            if weight_path.suffix.lower() != ".pt":
                raise ValueError(
                    f"Unsupported checkpoint '{weight_path}'. RSF-DETR accepts PyTorch .pt files only."
                )
            from ultralytics.nn.tasks import attempt_load_weights

            model = attempt_load_weights(weights, device=device, inplace=True, fuse=fuse)

        model.half() if fp16 else model.float()
        model.eval()

        self.model = model
        self.device = device
        self.fp16 = fp16
        self.names = check_class_names(
            getattr(model.module if hasattr(model, "module") else model, "names", default_class_names(data))
        )
        stride = getattr(model, "stride", torch.tensor([32]))
        self.stride = max(int(stride.max()), 32)
        self.batch_size = 1

        # Attributes used by the retained predictor and validator.
        self.pt = True
        self.jit = False
        self.engine = False
        self.triton = False

    def forward(self, im, augment=False, visualize=False):
        """Run a PyTorch forward pass."""
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        return self.model(im, augment=augment, visualize=visualize)

    def from_numpy(self, value):
        """Convert a NumPy value to a tensor on the configured device."""
        return torch.tensor(value).to(self.device) if not isinstance(value, torch.Tensor) else value

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Warm up CUDA inference with an empty tensor."""
        if self.device.type != "cpu":
            im = torch.empty(*imgsz, dtype=torch.float16 if self.fp16 else torch.float32, device=self.device)
            self.forward(im)
