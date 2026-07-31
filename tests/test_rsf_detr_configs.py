from pathlib import Path

import pytest
import yaml

from ultralytics import RSFDETR
from ultralytics.cfg import get_cfg
from ultralytics.nn.rsf_detr import CERB, MESA, SFFN
from ultralytics.nn.rsf_detr.parser import parse_rsf_detr_model


PAPER_CONFIGS = [
    "configs/rsf-detr-baseline.yaml",
    "configs/rsf-detr-cerb.yaml",
    "configs/rsf-detr-mesa.yaml",
    "configs/rsf-detr-sffn.yaml",
    "configs/rsf-detr-cerb-mesa.yaml",
    "configs/rsf-detr-cerb-sffn.yaml",
    "configs/rsf-detr-mesa-sffn.yaml",
    "configs/rsf-detr.yaml",
]


@pytest.mark.parametrize("config", PAPER_CONFIGS)
def test_paper_config_builds(config):
    model = RSFDETR(config)

    assert model.model is not None


def test_public_module_names_match_the_paper():
    assert RSFDETR.__name__ == "RSFDETR"
    assert CERB.__name__ == "CERB"
    assert MESA.__name__ == "MESA"
    assert SFFN.__name__ == "SFFN"


def test_configs_only_reference_supported_modules():
    allowed = {
        "AIFI",
        "BasicBlock",
        "Blocks",
        "CERB",
        "Concat",
        "Conv",
        "ConvNormLayer",
        "MESA",
        "RepC3",
        "RTDETRDecoder",
        "SFFN",
    }
    for config_path in PAPER_CONFIGS:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        modules = {layer[2] for layer in config["backbone"] + config["head"]}
        assert all(module in allowed or module.startswith("nn.") for module in modules)


def test_unrelated_module_is_rejected():
    config = yaml.safe_load(Path("configs/rsf-detr.yaml").read_text(encoding="utf-8"))
    config["backbone"][6][2] = "UnrelatedModule"

    with pytest.raises(ValueError, match="Unsupported module 'UnrelatedModule'"):
        parse_rsf_detr_model(config, channels=3, verbose=False)


def test_warmup_name_and_unit_are_explicit():
    args = get_cfg(overrides={"warmup_iters": 2000})

    assert args.warmup_iters == 2000
    assert not hasattr(args, "warmup_epochs")
