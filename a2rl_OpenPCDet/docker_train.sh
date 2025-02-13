#!/bin/bash

docker run \
    -v $PWD/../data/v1.0-mini/:/opt/ar_dataset/a2rl_OpenPCDet/data/nuscenes/v1.0-mini \
    -v $PWD/output:/opencv/a2rl_OpenPCDet/output \
    --gpus all -it ar_dataset


    # python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos     --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_a2rl.yaml     --version v1.0-mini
    # python3 train.py --cfg_file cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_a2rl.yaml
    # python3 test.py --cfg_file cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_a2rl.yaml --eval_all --ckpt_dir ../output/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_a2rl/default/ckpt