# Smoke-test samples only

Tiny YOLO-format subset (`images/{train,val}`, `labels/{train,val}`) for
**pipeline checks**. Not for reporting paper metrics.

| Split | Images | Labels |
|-------|-------:|-------:|
| train | 16 | 16 |
| val   | 4 | 4 |

The current sample contains 26 valid boxes, all with class ID `0`
(`pedestrian`). It verifies loading, training, and validation plumbing only; it
does not exercise all five PSD classes.

The full PSD dataset is not included because its raw construction-site imagery
is subject to privacy and data-sharing restrictions. See
[the data protocol](../../docs/DATA_PROTOCOL.md).

Default `dataset/PipeSafeDataset.yaml` uses `path: ./dataset/samples` so that:

```bash
python train.py --epochs 1 --batch 2
```

runs without external data. For training on the full dataset, change `path` to
your local PSD YOLO root (relative path recommended), e.g.
`./datasets/PSD_yolo`.
