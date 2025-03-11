#!/bin/bash

docker run \
    -v $PWD/../data/v1.0-mini/:/opt/vmax_dataset/vmax_OpenPCDet/data/nuscenes/v1.0-mini:ro \
    -v $PWD/../data/hilbert/:/opt/vmax_dataset/vmax_OpenPCDet/data/hilbert \
    -v $PWD/tools/cfgs:/opt/vmax_dataset/vmax_OpenPCDet/tools/cfgs \
    -v $PWD/output:/opt/vmax_dataset/vmax_OpenPCDet/output \
    --gpus all -it vmax_dataset
