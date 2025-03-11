#!/bin/bash
python3 evaluate_tracking.py
python3 -m nuscenes.eval.tracking.evaluate --config_path /opt/vmax_dataset/vmax_tracking/tracking_cfg.json --dataroot /opt/vmax_dataset/vmax_OpenPCDet/data/v1.0-mini --version v1.0-mini  --eval_set mini_all /opt/vmax_dataset/vmax_tracking/prediction_tracks_mini_all.json