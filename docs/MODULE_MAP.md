# Paper-to-code map

The public implementation has one canonical Python symbol for each proposed
paper module.

| Paper module | Canonical class | Source |
|---|---|---|
| Conditional Expert Routing Block | `CERB` | `ultralytics/nn/rsf_detr/cerb.py` |
| Multi-Scale Edge Selection Aggregation | `MESA` | `ultralytics/nn/rsf_detr/mesa.py` |
| Spectral-Enhanced Feed-Forward Network | `SFFN` | `ultralytics/nn/rsf_detr/sffn.py` |
| Strict paper-config parser | `parse_rsf_detr_model` | `ultralytics/nn/rsf_detr/parser.py` |
| Full model | — | `configs/rsf-detr.yaml` |

The parser exposes the three paper modules and the RT-DETR building blocks used
by the eight provided configurations. Other module names are rejected so that
configuration behavior remains explicit.

## Full-model path

```text
Input
  → ResNet-18 stages
  → CERB after P3, P4 and P5
  → MESA on the three backbone outputs
  → SFFN in the hybrid encoder
  → RT-DETR decoder and detection head
```

## Canonical YAML symbols

```yaml
backbone:
  - [-1, 1, CERB, [128, 4, 2]]
  - [-1, 1, CERB, [256, 8, 2]]
  - [-1, 1, CERB, [512, 16, 2]]

head:
  - [10, 1, MESA, [512]]
  - [8, 1, MESA, [256]]
  - [6, 1, MESA, [128]]
  - [-1, 1, SFFN, [1024, 8]]
```
