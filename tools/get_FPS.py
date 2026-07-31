import argparse
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from ultralytics import RSFDETR
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import select_device

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='trained weights path')
    parser.add_argument('--batch', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--imgs', nargs='+', type=int, default=[640, 640], help='[height, width] image sizes')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # Paper protocol: 100 warm-up iterations, then average over 500 runs (FP32, batch=1).
    parser.add_argument('--warmup', default=100, type=int, help='warmup iterations (paper: 100)')
    parser.add_argument('--testtime', default=500, type=int, help='timed iterations (paper: 500)')
    parser.add_argument('--half', action='store_true', default=False, help='fp16 mode.')
    opt = parser.parse_args()

    device = select_device(opt.device, batch=opt.batch)

    # Model
    weights = opt.weights
    if weights.endswith('.pt'):
        model = attempt_load_weights(weights=weights, device=device)
        print(f'Loaded {weights}')  # report
    else:
        model = RSFDETR(weights).model

    model = model.to(device)
    model.fuse()
    model.eval()
    example_inputs = torch.randn((opt.batch, 3, *opt.imgs)).to(device)

    if opt.half:
        model = model.half()
        example_inputs = example_inputs.half()

    print('begin warmup...')
    with torch.inference_mode():
        for _ in tqdm(range(opt.warmup), desc='warmup'):
            model(example_inputs)

    print('begin test latency...')
    time_arr = []

    with torch.inference_mode():
        for _ in tqdm(range(opt.testtime), desc='test latency'):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            model(example_inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            time_arr.append(time.perf_counter() - start_time)

    std_time = np.std(time_arr)
    infer_time_per_image = np.sum(time_arr) / (opt.testtime * opt.batch)

    if weights.endswith('.pt'):
        print(
            f'model weights:{opt.weights} size:{get_weight_size(opt.weights)}M '
            f'(bs:{opt.batch}) latency:{infer_time_per_image:.5f}s '
            f'+- {std_time:.5f}s fps:{1 / infer_time_per_image:.1f}'
        )
    else:
        print(
            f'model yaml:{opt.weights} (bs:{opt.batch}) '
            f'latency:{infer_time_per_image:.5f}s +- {std_time:.5f}s '
            f'fps:{1 / infer_time_per_image:.1f}'
        )
