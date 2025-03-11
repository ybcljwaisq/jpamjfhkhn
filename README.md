# V<sub>max</sub>: The autonomous racing dataset for long-range, high-speed perception and multi-vehicle interaction

Welcome to the V<sub>max</sub> dataset!

If you use our dataset, or want to cite it, please use the following citation:
```
```

## Changelog

## Install

#### Step 1: Download the Data

The dataset is available for download through the following mirrors:

| Mirror  | URL  |
|---------|------|
| Google Drive (Available shortly) | [Google Drive Download](#) |


#### Step 2: Extract the Data
After downloading the data, please extract it in the `data` directory.
```bash
mv /path/to/vmax_dataset.tar.xz data/
cd data/
tar -xf vmax_dataset.tar.xz
``` 

#### Step 3: Setup the Docker container
We provide a dockerfile that install all nessesary libraries for detection and tracking tasks.
```
docker build -t vmax_dataset -f Dockerfile .
```

Note: VoxelMamba and LION conflict in their install. Hence, we provide a separate branch for LION and VoxelMamba. If you want to train the LION model, please run `git checkout voxelmamba` or `git checkout lion` before building the docker container.

After building the docker, you need to create the dataset info file for OpenPCDet once.
```bash
cd vmax_OpenPCDet && ./docker_train.sh

cd .. && python3 -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos --cfg_file tools/cfgs/dataset_configs/vmax_dataset.yaml --version v1.0-mini
```
## Detection

For detection, we rely on OpenPCDet. We made several changes to support detections up to 300 meters, as well as the range-based evaluation.
<br>
Detection configurations are stored in `vmax_OpenPCDet/tools/cfgs/vmax_models`

#### Training a model
```bash
# ./docker_train.sh contains nessesary mounts so the training works out of the box
# if you extracted the data somewhere else, please adjust the docker_train.sh accordingly
cd vmax_OpenPCDet && ./docker_train.sh

# For example: to train SECOND, change the config to the desired model
python3 train.py --cfg_file cfgs/vmax_models/second.yaml
```

#### Evaluating a model

```bash
# ./docker_train.sh contains nessesary mounts so the training works out of the box
cd vmax_OpenPCDet && ./docker_train.sh

# For example: to train SECOND, change the config to the desired model
python3 test.py --cfg_file cfgs/vmax_models/second_a2rl.yaml --ckpt ../output/vmax_dataset/<path_to_ckpt>
```

If you have further questions, please refer to the official OpenPCDet repository.

#### Evaluating the runtime of a model

```bash
# ./docker_train.sh contains nessesary mounts so the training works out of the box
cd vmax_OpenPCDet && ./docker_train.sh

# For example: to train SECOND, change the config to the desired model
python3 eval_inference_time.py --cfg_file cfgs/vmax_models/second_a2rl.yaml --ckpt ../output/vmax_dataset/<path_to_ckpt>
```

## Tracking

Right now we cannot provide the code to run tracking as the license prohibits publishing it in any modified way.
Once we resolved the situation, we will upload the modified code. If that is not possible in the foreseeable future, we will provide instructions how to modify the code.

If you eant to run your own tracker, your tracker API should be able to run on `[[h,w,l,x,y,z,theta]]`. See `vmax_tracking/evaluate_tracking.py` for details. Alternatively, you can modify `vmax_tracking/evaluate_tracking.py` to fit your trackers API.

Before running the tracker, please make sure you adjust paths to the dataset in `./docker_track.sh` as well as the `evaluate_tracking.sh` in `vmax_tracking`.
```bash
cd vmax_tracking && ./docker_track.sh
```

## Acknowledgement
