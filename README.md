# RSF-DETR

Official implementation of **RSF-DETR: Routing–Structure–Frequency Enhanced
Detection Transformer for Small-Object Detection under Class Imbalance in
Construction-Site Monitoring**.

RSF-DETR adds the three modules defined in the paper to an RT-DETR-R18
detection core:

| Paper symbol | Full name | Implementation |
|---|---|---|
| `CERB` | Conditional Expert Routing Block | `ultralytics/nn/rsf_detr/cerb.py` |
| `MESA` | Multi-Scale Edge Enhancement Aggregation | `ultralytics/nn/rsf_detr/mesa.py` |
| `SFFN` | Spectral-Enhanced Feed-Forward Network | `ultralytics/nn/rsf_detr/sffn.py` |

This repository focuses on the modules, configurations, and evaluation tools
needed to reproduce the experiments described in the paper.

## Repository layout

```text
RSF-DETR/
├── configs/                   # full model and seven paper ablations
├── dataset/                   # portable dataset YAMLs and smoke-test samples
├── docs/                      # data protocol, reproduction, code map
├── tests/                     # config/name boundary tests
├── tools/                     # FPS and model-complexity helpers
├── ultralytics/
│   ├── nn/rsf_detr/           # CERB, MESA, SFFN and strict config parser
│   └── models/rtdetr/         # RT-DETR training and validation core
├── train.py
└── val.py
```

## Quick start

```bash
pip install -r requirements.txt

# One-epoch pipeline check using the bundled sample subset
python train.py --epochs 1 --batch 2 --workers 2 --name smoke

# Full RSF-DETR training
python train.py \
  --config configs/rsf-detr.yaml \
  --data dataset/PipeSafeDataset.yaml \
  --weights weights/rtdetr-r18.pt \
  --project runs/psd --name rsf_detr --amp

# Validation
python val.py \
  --weights runs/psd/rsf_detr/weights/best.pt \
  --data dataset/PipeSafeDataset.yaml --conf 0.001
```

`rtdetr-r18.pt` is only the published upstream initialization checkpoint name;
all RSF-DETR source symbols and paper configurations use the paper terminology.

## Paper configurations

| Setting | Config |
|---|---|
| Baseline | `configs/rsf-detr-baseline.yaml` |
| CERB | `configs/rsf-detr-cerb.yaml` |
| MESA | `configs/rsf-detr-mesa.yaml` |
| SFFN | `configs/rsf-detr-sffn.yaml` |
| CERB + MESA | `configs/rsf-detr-cerb-mesa.yaml` |
| CERB + SFFN | `configs/rsf-detr-cerb-sffn.yaml` |
| MESA + SFFN | `configs/rsf-detr-mesa-sffn.yaml` |
| RSF-DETR | `configs/rsf-detr.yaml` |

Detailed training and evaluation settings are provided in
[docs/REPRODUCE.md](docs/REPRODUCE.md). The exact paper-to-code map is available
in [docs/MODULE_MAP.md](docs/MODULE_MAP.md).

## Data and checkpoints

The full PSD dataset is not included because its raw construction-site imagery
is subject to privacy and data-sharing restrictions. The repository contains a
small pipeline-test subset and portable YAML templates. VisDrone and MOCS must
be downloaded from their public sources. See
[docs/DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md).

Store released `.pt` checkpoints under `weights/` and publish large files
through GitHub Releases instead of Git history.

## License

This repository is distributed under AGPL-3.0 because it contains a reduced
Ultralytics RT-DETR fork. See `LICENSE` and `NOTICE`.
