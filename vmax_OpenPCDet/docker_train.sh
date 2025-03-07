#!/bin/bash

docker run \
    -v $PWD/../data/v1.0-mini/:/opt/vmax_dataset/vmax_OpenPCDet/data/nuscenes/v1.0-mini:ro \
    -v $PWD/../data/hilbert/:/opt/vmax_dataset/vmax_OpenPCDet/data/hilbert \
    -v $PWD/tools/cfgs:/opt/vmax_dataset/vmax_OpenPCDet/tools/cfgs \
    -v $PWD/output:/opt/vmax_dataset/vmax_OpenPCDet/output \
    --gpus all -it vmax_dataset

# docker run \
#     -v /home/marvin/dev/dataset/data/v1.0-mini/:/opt/vmax_dataset/vmax_OpenPCDet/data/nuscenes/v1.0-mini \
#     -v /home/marvin/dev/lidar_object_detection/OpenPCDet/.vscode:/opt/vmax_dataset/vmax_OpenPCDet/.vscode \
#     -v $PWD/tools/cfgs:/opt/vmax_dataset/vmax_OpenPCDet/tools/cfgs \
#     -v $PWD/output:/opt/vmax_dataset/vmax_OpenPCDet/output \
#     --gpus all -it vmax_dataset

# docker run \
#     -v $PWD/../data/v1.0-mini/:/opt/ar_dataset/a2rl_OpenPCDet/data/nuscenes/v1.0-mini \
#     -v $PWD/../data/hilbert/:/opt/ar_dataset/a2rl_OpenPCDet/data/hilbert \
#     -v $PWD/tools/cfgs:/opt/ar_dataset/a2rl_OpenPCDet/tools/cfgs \
#     -v $PWD/output:/opt/ar_dataset/a2rl_OpenPCDet/output \
#     --gpus all -it ar_dataset