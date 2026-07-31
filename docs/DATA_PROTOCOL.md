# Data protocol (PSD / MOCS / VisDrone)

This document describes the class definitions, annotation and split protocol,
and data-access conditions used in the paper. Full PSD raw images are **not**
distributed with this repository.

## 1. PipeSafeDataset (PSD) — primary benchmark

### 1.1 Scope

- Domain: pipeline construction-site monitoring (fixed masts / UAV-style views).
- Task: multi-class object detection (YOLO-format boxes).
- Classes (`nc = 5`):

| ID | Name | Description (EN) |
|----|------|------------------|
| 0 | `pedestrian` | Workers / pedestrians on site |
| 1 | `car` | Cars / light vehicles |
| 2 | `truck` | Trucks / transport vehicles |
| 3 | `non_motor_vehicle` | Non-motor vehicles |
| 4 | `construction_machinery` | Special construction machinery |

### 1.2 Statistics (full restricted dataset, as used in the paper)

| Item | Value |
|------|------:|
| Images (annotated) | 12,312 |
| Instances | 32,038 |
| Train images | 11,199 (~91%) |
| Val images | 1,113 (~9%) |
| Train instances | 29,635 |
| Val instances | 2,403 |
| Independent test set | None; **val is the official evaluation split** |

Instance frequencies (head → tail):

| Class | Instances | Share | Frequency group (paper) |
|-------|----------:|------:|-------------------------|
| pedestrian | 12,978 | 40.51% | Head |
| car | 7,385 | 23.05% | Head |
| truck | 4,678 | 14.60% | Medium |
| non_motor_vehicle | 4,050 | 12.64% | Medium |
| construction_machinery | 2,947 | 9.20% | Tail |

### 1.3 Split protocol

- **Fixed image-level index split** (not random re-split per run).
- All boxes of one image stay in the same subset (no box-level leakage).
- Physical layout:

```text
PipeSafeDataset/
  images/train/   images/val/
  labels/train/   labels/val/    # YOLO txt: class cx cy w h (normalized)
```

- Training and hyper-parameter selection use **train only**; reported metrics use **val only**.

### 1.4 Why raw images are restricted

Raw site imagery may contain identifiable workers, equipment serials, or site
layout details. Because of privacy and data-sharing restrictions, **full PSD
images are not published** with this code release.

### 1.5 What this repo provides

| Asset | Location |
|-------|----------|
| Dataset YAML template | `dataset/PipeSafeDataset.yaml` (`path: ./dataset/samples` by default) |
| Tiny smoke-test subset | `dataset/samples/` (20 images, 26 class-0 boxes; do **not** use for paper metrics) |
| This protocol | `docs/DATA_PROTOCOL.md` |

The full annotations and the 12,312-image split index are not included because
they remain linked to restricted raw imagery.

**Not provided:** full PSD images and labels.

### 1.6 Local layout when you have full PSD

Use a **relative** path from the repository root (example only):

```text
./datasets/PSD_yolo/
  images/train/   images/val/
  labels/train/   labels/val/
```

```yaml
# dataset/PipeSafeDataset.yaml
path: ./datasets/PSD_yolo
train: images/train
val: images/val
```

Never commit absolute host paths into shared configs.

### 1.7 Requesting full PSD access

Where data-use permissions allow, researchers may request access for
non-commercial academic reproduction by contacting the corresponding author
listed in the paper with:

1. Name, affiliation, and intended use
2. Agreement not to redistribute raw images
3. Compliance with local privacy regulations

Subject to the applicable permissions, class lists, annotation guidelines,
evaluation configs, and additional anonymized examples may also be available.

---

## 2. VisDrone2019-DET (public)

- Official project: <https://github.com/VisDrone/VisDrone-Dataset>
- Config template: `dataset/VisDroneDataset.yaml`
- Set `path` to your local root after download.
- Paper uses the unified protocol (imgsz 640, conf 0.001, same training recipe unless noted).

## 3. MOCS (public)

- Construction-site detection benchmark (13 classes).
- Config template: `dataset/MOCSDataset.yaml`
- Convert official labels to YOLO layout if needed, then set `path`.
- Some table entries marked † in the paper are taken from prior reports and are
  **not** claimed as re-implemented with this codebase.

## 4. Evaluation protocol (all three datasets)

| Setting | Value |
|---------|-------|
| Input | Aspect-preserving resize + pad to 640×640 |
| Metric | COCO-style AP50, mAP50:95, AP_S / AP_M / AP_L |
| Confidence threshold | 0.001 |
| Best checkpoint | Highest validation mAP50:95 |
| FPS (paper tables) | FP32, batch=1, 100 warm-up + 500 timed runs |

See `docs/REPRODUCE.md` for exact commands.
