#!/bin/bash

docker run \
    -v /home/marvin/dev/dataset/data/v1.0-mini/:/opt/ar_dataset/a2rl_OpenPCDet/data/nuscenes/v1.0-mini \
    -v /home/marvin/dev/lidar_object_detection/OpenPCDet/.vscode:/opt/ar_dataset/a2rl_OpenPCDet/.vscode \
    -v $PWD/output:/opt/ar_dataset/a2rl_OpenPCDet/output \
    --gpus all -it ar_dataset


    # python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos     --cfg_file tools/cfgs/dataset_configs/a2rl_dataset.yaml     --version v1.0-mini
    # python3 train.py --cfg_file cfgs/a2rl_models/cbgs_pillar0200_res2d_centerpoint_a2rl.yaml
    # python3 test.py --cfg_file cfgs/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_a2rl.yaml --eval_all --ckpt_dir ../output/nuscenes_models/cbgs_pillar0075_res2d_centerpoint_a2rl/default/ckpt