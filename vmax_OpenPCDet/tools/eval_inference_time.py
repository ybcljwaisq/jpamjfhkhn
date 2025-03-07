import _init_path
import argparse
import datetime
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import tqdm

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models import load_data_to_gpu


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)
    return args, cfg


def evaluate_inference_speed(model, test_loader, time_meter=None):    
    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(test_loader), leave=True, desc='eval', dynamic_ncols=True)

    start_time = time.time()
    for _, batch_dict in enumerate(test_loader):
        load_data_to_gpu(batch_dict)

        start_time = time.time()
        with torch.no_grad():
            _ = model(batch_dict)
        inference_time = time.time() - start_time

        if time_meter:
            time_meter.update(inference_time * 1000)

        if cfg.LOCAL_RANK == 0:
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()



def main():
    args, cfg = parse_config()

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    dist_test = False
    total_gpus = 1
    args.batch_size = 1 # use to match "inference"

    import logging
    logger = logging.getLogger("OpenPCDet")
    logger.addHandler(logging.NullHandler())

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda().eval()

    with torch.no_grad():
        evaluate_inference_speed(model, test_loader, None) # One epoch for GPU warmup
        avg_time = []
        for _ in range(5):
            infer_time_meter = common_utils.AverageMeter()
            evaluate_inference_speed(model, test_loader, infer_time_meter)
            print("AVG TIME")
            print(f"{infer_time_meter.avg:.2f}")
            avg_time.append(infer_time_meter.avg)

        mean_time = np.mean(avg_time)
        stddev_time = np.std(avg_time, ddof=1)
        print(f"Mean Inference Time: {mean_time:.2f}")
        print(f"Standard Deviation: {stddev_time:.2f}")


if __name__ == '__main__':
    main()
