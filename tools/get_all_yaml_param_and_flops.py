import argparse
import glob

import tqdm

from ultralytics import RSFDETR
from ultralytics.utils.torch_utils import model_info


def parse_args():
    parser = argparse.ArgumentParser(description="Profile paper model YAML files.")
    parser.add_argument("--glob", default="configs/*.yaml", help="RSF-DETR configuration glob.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    flops_dict = {}
    failures = []
    for yaml_path in tqdm.tqdm(sorted(glob.glob(args.glob, recursive=True))):
        try:
            model = RSFDETR(yaml_path)
            model.fuse()
            n_l, n_p, n_g, flops = model_info(model.model)
            flops_dict[yaml_path] = [flops, n_p]
        except Exception as exc:
            failures.append((yaml_path, exc))

    sorted_items = sorted(flops_dict.items(), key=lambda x: x[1][0])
    for key, value in sorted_items:
        print(f"{key}: {value[0]:.2f} GFLOPs {value[1]:,} Params")
    for path, exc in failures:
        print(f"FAILED {path}: {exc}")
