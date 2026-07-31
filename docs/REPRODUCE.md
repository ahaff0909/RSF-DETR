# Reproduction guide

## Environment

```bash
cd RSF-DETR
pip install -r requirements.txt
```

Python 3.8 or later is required. Install the PyTorch build matching the local
CUDA environment before training.

## Config mapping

| Paper setting | YAML |
|---|---|
| Baseline | `configs/rsf-detr-baseline.yaml` |
| CERB | `configs/rsf-detr-cerb.yaml` |
| MESA | `configs/rsf-detr-mesa.yaml` |
| SFFN | `configs/rsf-detr-sffn.yaml` |
| CERB + MESA | `configs/rsf-detr-cerb-mesa.yaml` |
| CERB + SFFN | `configs/rsf-detr-cerb-sffn.yaml` |
| MESA + SFFN | `configs/rsf-detr-mesa-sffn.yaml` |
| RSF-DETR | `configs/rsf-detr.yaml` |

The default class count is 5 for PSD. Training rebuilds the decoder for the
class count declared in the selected dataset YAML.

## Reference training protocol

| Key | Value |
|---|---:|
| epochs | 300 |
| batch | 16 |
| image size | 640 |
| optimizer | AdamW |
| initial learning rate | 1e-4 |
| weight decay | 1e-4 |
| momentum / beta1 | 0.9 |
| warm-up iterations | 2000 |
| Mosaic | 0.0 |
| cosine schedule | disabled |
| patience | 30 |
| classification loss | Varifocal Loss |
| initialization | `weights/rtdetr-r18.pt` |

The available experiment records and the paper description differ in several
settings. PSD run metadata enables AMP, while MOCS and VisDrone run metadata
disable it; the paper describes all runs as FP32. The released configuration
also disables Mosaic, uses Varifocal Loss, and implements separate CERB
load-balancing and router-Z terms, whereas the paper describes Mosaic, Focal
Loss, and a single CERB coefficient of 0.05. Reproduction reports should state
which settings were used when comparing results.

## Portable data paths

| YAML | Default `path` |
|---|---|
| `dataset/PipeSafeDataset.yaml` | `./dataset/samples` |
| `dataset/VisDroneDataset.yaml` | `./datasets/VisDrone2019` |
| `dataset/MOCSDataset.yaml` | `./datasets/MOCS_yolo` |

Do not commit machine-specific absolute paths. The bundled PSD samples are only
for pipeline checks and must not be used to report paper metrics.

## Commands

```bash
# Pipeline check
python train.py \
  --config configs/rsf-detr.yaml \
  --data dataset/PipeSafeDataset.yaml \
  --epochs 1 --batch 2 --workers 2 --name smoke_rsf

# Full PSD setup
python train.py \
  --config configs/rsf-detr.yaml \
  --data dataset/PipeSafeDataset.yaml \
  --weights weights/rtdetr-r18.pt \
  --project runs/psd --name rsf_detr_full \
  --epochs 300 --batch 16 --workers 8 --seed 0 --amp

# Validation
python val.py \
  --weights runs/psd/rsf_detr_full/weights/best.pt \
  --data dataset/PipeSafeDataset.yaml \
  --split val --conf 0.001

# Parameters and FLOPs
python tools/main_profile.py
python tools/get_all_yaml_param_and_flops.py

# Runtime test
python tools/get_FPS.py \
  --weights weights/rsf-detr-psd-best.pt \
  --batch 1 --warmup 100 --testtime 500
```

Publish large checkpoints through GitHub Releases and place downloaded copies
under the ignored local `weights/` directory.
