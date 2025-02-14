from glob import glob
import json 
import uuid
from itertools import groupby
from operator import itemgetter
import os

import open3d as o3d
from pypcd_imp.pypcd import PointCloud
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from pyquaternion import Quaternion


def extract_info(filename):
    path = filename.split("/")
    filename = path[-1]
    team = path[-2]
    log = path[-3]
    scene = filename.split(".")[0]
    return team, log, scene


def detection_json_to_map(detections, filename):
    team, log, scene = extract_info(filename)

    with open(file=filename, mode="r") as fp:
        datumaro = json.load(fp)

    if log not in detections:
        detections[log] = {}
    if team not in detections[log]:
        detections[log][team] = {}
    if scene not in detections[log][team]:
        detections[log][team][scene] = {}
    
    for frame in datumaro["items"]:
        detections[log][team][scene][frame["id"]] = [{"id": annotation["id"], "position": annotation["position"], "size": annotation["scale"], "yaw": annotation["rotation"][2], "visible": not annotation["attributes"]["occluded"]} for annotation in frame["annotations"]]

    return detections


def generate_unique_token():
    return uuid.uuid4().hex

def generate_log(log_name, team):
    """ Generate one log per session
    """
    return {
        'token': generate_unique_token(),
        "logfile": "", #@TODO
        "vehicle": team,
        "date_captured": log_name, # @TODO use this as specification when it was?
        "location": f"Yas Marina Circuit",
    }

def generate_scene(scene, log, team, nbr_samples):
    return {
        'token': generate_unique_token(),
        'log_token': "",
        'nbr_samples': nbr_samples,
        'first_sample_token': "",
        'last_sample_token': "",
        'name': scene,
        'description': f"{scene} of {log} for team {team}",
    }

def generate_sample(scene_token, timestamp):
    return {
        "token": generate_unique_token(),
        "scene_token": scene_token,
        "timestamp": int(timestamp),
        "next": "",
        "prev": ""
    }

def generate_sample_annotation(sample_token, annotation, instance_map):
    return {
        "token": generate_unique_token(),
        "sample_token": sample_token,
        "instance_token": get_instance_token(instance_map, annotation["id"]),
        "annotation_id": annotation["id"], # this is a custom field. It is required to match instances to sample_annotations
        "prev": "",
        "next": "",
        "attribute_tokens": [],
        "visibility_token": "0", # We do not annotate this
        "translation": annotation["position"],
        "size": [annotation["size"][1], annotation["size"][0], annotation["size"][2]], # flip annotation sizes to match NUS format
        "rotation": Quaternion(axis=[0, 0, 1], angle=annotation["yaw"]).q.tolist(),
        "num_lidar_pts": 1, # OpenPCDet requires to have at least 1 lidar_pts in the box. @todo: calculate the number points and store here
        "num_radar_pts": 1
    }


def generate_sample_data(sample_token, team_name, ego_pose_token, calibrated_sensor_token, timestamp, type="LIDAR_TOP"):
    if type == "LIDAR_TOP":
        file_format = ".pcd.bin"
    else:
        file_format = ".pcd"

    return {
        "token": generate_unique_token(),
        "sample_token": sample_token,
        "ego_pose_token": ego_pose_token,
        "calibrated_sensor_token": calibrated_sensor_token,
        "timestamp": int(timestamp),
        "fileformat": "pcd",
        "is_key_frame": True,
        "height": 0,
        "width": 0,
        "filename": f"samples/{type}/{type}_{team_name}_{timestamp}{file_format}",
        "prev": "",
        "next": ""
    }

def generate_ego_pose(timestamp, translation, rotation):
    return {
        "token": generate_unique_token(),
        "timestamp": int(timestamp),
        "rotation": rotation,
        "translation": translation
    }


def generate_calibrated_sensor(sensor_token, translation, rotation, camera_intrinsic):
    return {
        "token": generate_unique_token(),
        "sensor_token": sensor_token,
        "translation": translation,
        "rotation": rotation,
        "camera_intrinsic": camera_intrinsic
    }


def get_instance_token(map, id):
    if id not in map:
        map[id] = generate_unique_token()

    return map[id]


def find_last_sample_annotation(samples_annotations, token):
    for sample_annotation in reversed(samples_annotations):
        if sample_annotation['instance_token'] == token:
            return sample_annotation
    
    return None


def generate_instances(samples_annotations, racecar_category_token, instance_map):
    samples_annotations.sort(key=itemgetter('instance_token'))
    grouped_objects = {k: list(v) for k, v in groupby(samples_annotations, key=itemgetter('instance_token'))}
    instances = []
    for _, group in grouped_objects.items():
        instances.append({
            "token": get_instance_token(instance_map, group[0]["annotation_id"]),
            "category_token": racecar_category_token,
            "nbr_annotations": len(group),
            "first_annotation_token": group[0]["token"],
            "last_annotation_token": group[-1]["token"]
        })

    return instances


def save_lidar_data(path_data, path_output, log, scene, team, filename):
    # @todo: assume timestamp data is correct here
    
    original_filename = filename.split(os.path.sep)[-1].split("_")[-1][:-4] # remove .bin at the end as pointclouds are in .pcd originally
    original_path = os.path.join(path_data, "lidar", log, team, scene, original_filename)
    if not os.path.exists(original_path):
        raise IOError(f"Pointcloud {original_path} not found")
    
    pcl = o3d.t.io.read_point_cloud(original_path)
    positions = pcl.point["positions"].numpy()
    # quick fix
    if "intensity" in pcl.point:
        intensities = pcl.point["intensity"].numpy()
    else:
        intensities = pcl.point["intensities"].numpy()

    pcl_nus = np.zeros([positions.shape[0], 5], dtype=np.float32)
    pcl_nus[:, :3] = positions

    # Different teams have different intensity scales
    if INTENSITY_SCALE == 1:
        if intensities[:, 0].max() > 1.0 + 1e06:
            intensities = intensities / 255
    if INTENSITY_SCALE == 255:
        if intensities[:, 0].max() < 1.0 + 1e06:
            intensities = intensities * 255
            intensities = np.round(intensities)
    pcl_nus[:, 3] = intensities[:, 0]
    
    file_path = os.path.join(path_output, f"{filename}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    with open(file_path, "wb") as fp:
       pcl_nus.tofile(fp)

radar_cache = {}

def cache_radar_files(path_data, log, team, scene):
    """Caches radar file timestamps and filenames for fast lookup."""
    radar_path = os.path.join(path_data, "radar", log, team, scene)
    radar_files = glob(os.path.join(radar_path, "*.pcd"))

    timestamps = []
    filenames = []

    for file in radar_files:
        filename = os.path.basename(file)
        file_timestamp_str, _ = os.path.splitext(filename)
        try:
            timestamps.append(int(file_timestamp_str))
            filenames.append(file)
        except ValueError:
            continue 

    if timestamps:
        # Sort for efficient searching
        sorted_indices = np.argsort(timestamps)
        timestamps = np.array(timestamps)[sorted_indices]
        filenames = np.array(filenames)[sorted_indices]

    radar_cache[(log, team, scene)] = (timestamps, filenames)


def find_closest_radar_file(timestamp_lidar, log, team, scene):
    """Finds the closest radar file using NumPy search."""
    if (log, team, scene) not in radar_cache:
        return None

    timestamps, filenames = radar_cache[(log, team, scene)]
    if len(timestamps) == 0:
        return None

    idx = np.searchsorted(timestamps, timestamp_lidar)
    if idx == 0:
        return filenames[0]
    elif idx == len(timestamps):
        return filenames[-1]
    
    # Compare the two closest options
    before = timestamps[idx - 1]
    after = timestamps[idx]
    if abs(before - timestamp_lidar) < abs(after - timestamp_lidar):
        return filenames[idx - 1]
    else:
        return filenames[idx]
    

def radar_sample_data_for_lidar_scene_sample_data(path_data, path_output, scd, log, scene, team, calibrated_sensor_token):
    # find closes radar file to lidar file

    if (log, team, scene) not in radar_cache:
        cache_radar_files(path_data, log, team, scene)

    # Find the closest radar file
    timestamp_lidar = scd["timestamp"]
    closest_file = find_closest_radar_file(timestamp_lidar, log, team, scene)

    if closest_file is None:
        return None

    timestamp_radar = os.path.basename(closest_file)[:-4]
    pcl = o3d.t.io.read_point_cloud(closest_file)
    positions = pcl.point["positions"].numpy()
    intensity = pcl.point["intensity"].numpy()
    velocity = pcl.point["vel_var"].numpy()

    num_points = positions.shape[0]
    # save radar pointcloud in nuscenes format
    # x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    # As the radar data we get differs, we can only populate a few of the fields    
    pcl_nus = np.zeros(num_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('dyn_prop', 'i1'), ('id', 'i2'), ('rcs', 'f4'),
        ('vx', 'f4'), ('vy', 'f4'), ('vx_comp', 'f4'), ('vy_comp', 'f4'),
        ('is_quality_valid', 'i1'), ('ambig_state', 'i1'),
        ('x_rms', 'i1'), ('y_rms', 'i1'),
        ('invalid_state', 'i1'), ('pdh0', 'i1'),
        ('vx_rms', 'i1'), ('vy_rms', 'i1')
    ])

    # Assign known values
    pcl_nus['x'] = positions[:, 0]
    pcl_nus['y'] = positions[:, 1]
    pcl_nus['z'] = positions[:, 2]
    pcl_nus['rcs'] = intensity[:, 0]  # Radar Cross Section
    pcl_nus['vx'] = velocity[:, 0]  # Velocity X (assuming radial velocity)
    
    # Default values for other fields (since we don't have real data for them)
    pcl_nus['dyn_prop'] = 0
    pcl_nus['id'] = 0
    pcl_nus['vy'] = 0
    pcl_nus['vx_comp'] = 0
    pcl_nus['vy_comp'] = 0
    pcl_nus['is_quality_valid'] = 0
    pcl_nus['ambig_state'] = 3
    pcl_nus['x_rms'] = 0
    pcl_nus['y_rms'] = 0
    pcl_nus['invalid_state'] = 0
    pcl_nus['pdh0'] = 1
    pcl_nus['vx_rms'] = 0
    pcl_nus['vy_rms'] = 0

    # PCD metadata definition (matches the required header)
    metadata = {
        'version': 0.7,
        'fields': ['x', 'y', 'z', 'dyn_prop', 'id', 'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp',
                   'is_quality_valid', 'ambig_state', 'x_rms', 'y_rms',
                   'invalid_state', 'pdh0', 'vx_rms', 'vy_rms'],
        'size': [4, 4, 4, 1, 2, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1],  # Match field sizes
        'type': ['F', 'F', 'F', 'I', 'I', 'F', 'F', 'F', 'F', 'F',
                 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'I'],  # Float (F) & Integer (I)
        'count': [1] * 18,  # One value per field
        'width': num_points,
        'height': 1,
        'viewpoint': [0, 0, 0, 1, 0, 0, 0],  # Viewpoint is required for pypcd
        'points': num_points,
        'data': 'binary'
    }

    # Create a PCD object using the PointCloud constructor
    pc = PointCloud(metadata, pcl_nus)
    
    file_path = os.path.join(path_output, f"samples/RADAR_TOP/RADAR_TOP_{team}_{timestamp_radar}.pcd")
    os.makedirs(os.path.dirname(file_path), exist_ok=True) 
    pc.save_pcd(file_path, compression='binary')
    # Nuscenes requires first line to be a comment and pypcd doesnt write the comment
    comment = "# .PCD v0.7 - Point Cloud Data file format\n"
    with open(file_path, "rb") as f_in:
        pcd_content = f_in.read()
    with open(file_path, "wb") as f_out:
        f_out.write(comment.encode())
        f_out.write(pcd_content)
        f_out.write("\n".encode()) # NuS expects the new line at EOF

    return generate_sample_data(
        scd["sample_token"],
        team,
        scd["ego_pose_token"],
        calibrated_sensor_token,
        timestamp_radar,
        type="RADAR_TOP"
    )

def detections_to_nuscenes(detections, path_json, path_data, path_output):
    attributes = []
    calibrated_sensors = []
    categories = [{
        "token": generate_unique_token(),
        "name": "vehicle.car",
        "description": "A car, in this domain mostly a racecar."
    }]
    ego_poses = []
    instances = []
    logs = []
    maps = [{
        "category": "semantic_prior",
        "token": "53992ee3023e5494b90c316c183be829",
        "filename": "maps/53992ee3023e5494b90c316c183be829.png", # @todo: generate image from gnss path from TUM
        "log_tokens": []
    },]
    samples = []
    samples_annotations = []
    samples_data = []
    scenes = []
    vehicle_translation = [0,0,0]
    vehicle_rotation = [1,0,0,0]
    sensor_lidar_vehicle_token = generate_unique_token()
    sensor_radar_vehicle_token = generate_unique_token()
    sensors = [
        {
            "token": sensor_lidar_vehicle_token,
            "channel": "LIDAR_TOP",
            "modality": "lidar"
        },
        {
            "token": sensor_radar_vehicle_token,
            "channel": "RADAR_TOP",
            "modality": "radar"
        },
    ]
    visibility = [{
        "token": "0",
        "description": "not implemented",
        "level": "0"
    }]


    for log_name, teams in detections.items():
        print(f"Processing {log_name}")
        for team, log_scenes in teams.items():
            # just for development
            # if team != "uni":
            #     continue

            print(f"Processing team {team}")
            log = generate_log(log_name, team)
            logs.append(log)
            for scene, frames in tqdm(log_scenes.items()):
                instance_map = {}
                scene = generate_scene(scene, log["date_captured"], team, len(frames))

                scene_samples = []
                scene_sample_annotations = []
                scene_sample_data = []
                for sample_idx, (timestamp, frame) in enumerate(frames.items()):
                    # There is some weird bug with the constructor data
                    # In the beginning it sometimes has a gap of 1 plc
                    if sample_idx == 0:
                        ts_0 = int(list(frames.keys())[0].replace("_", ""))
                        ts_1 = int(list(frames.keys())[1].replace("_", ""))
                        if ts_1 - ts_0 >= 200000000:
                            # skip this iteration, it would not affect detection, but it would affect tracking
                            print(f"Initial skip detected for: {scene['name']}")
                            continue
   
                    #
                    # Furthermore, there is another minor weird problem. Sometimes, one pcl is missing at the end
                    # if that is the case, we end the scene without this sample.
                    # This is mainly in sections there were shorter than 10s
                    # this is a little bit ugly codewise, but is does its job
                    #
                    original_path = os.path.join(path_data, "lidar", log["date_captured"], team, scene["name"], f"{timestamp.replace('_', '')}.pcd")
                    if not os.path.exists(original_path):
                        print(f"After sample idx {sample_idx} pointcloud {original_path} not found")
                        break

                    # only required if timestamp is split by _
                    timestamp = timestamp.replace("_", "")
                    
                    #
                    # Generate Samples
                    #
                    sample = generate_sample(scene["token"], timestamp)
                    if len(scene_samples) > 0:
                        sample["prev"] = scene_samples[-1]["token"]
                        scene_samples[-1]["next"] = sample["token"]
                    if len(scene_samples) == 0:
                        scene["first_sample_token"] = sample["token"]
                    scene_samples.append(sample)

                    #
                    # Generate Annotations
                    #
                    for annotation in frame:
                        if not annotation["visible"]:
                            continue

                        sample_annotation = generate_sample_annotation(sample["token"], annotation, instance_map)
                        last_sample_annotation = find_last_sample_annotation(scene_sample_annotations, sample_annotation["instance_token"])
                        if last_sample_annotation:
                            sample_annotation["prev"] = last_sample_annotation["token"]
                            last_sample_annotation["next"] = sample_annotation["token"]
                        
                        scene_sample_annotations.append(sample_annotation)

                    #
                    # Generate Sample Data
                    #
                    # We define all LiDAR frames as key frames. Radar / Camera are associated to closest
                    #
                    ego_pose = generate_ego_pose(timestamp, translation=vehicle_translation, rotation=vehicle_rotation) # @TODO: gnss values?
                    ego_poses.append(ego_pose)

                    # LiDAR pointclouds and detections are already in vehicle frame
                    calibrated_sensor = generate_calibrated_sensor(sensor_lidar_vehicle_token, translation=vehicle_translation, rotation=vehicle_rotation, camera_intrinsic=[])
                    calibrated_sensors.append(calibrated_sensor)

                    sample_data = generate_sample_data(
                        sample["token"],
                        team,
                        ego_pose_token=ego_pose["token"],
                        calibrated_sensor_token=calibrated_sensor["token"],
                        timestamp=timestamp,
                        type="LIDAR_TOP"
                    )
                    if len(scene_sample_data) > 0:
                        sample_data["prev"] = scene_sample_data[-1]["token"]
                        scene_sample_data[-1]["next"] = sample_data["token"]
                    scene_sample_data.append(sample_data)

                    # Move LiDAR data
                    save_lidar_data(path_data, path_output, log["date_captured"], scene["name"], team, sample_data["filename"])

                scene["last_sample_token"] = sample["token"]

                #
                # Alle Lidar generieren
                # Alle Radar generieren
                # da Lidar key frames sind zu key frames radar frames finden
                #
                radar_sample_datas = []
                radar_calibrated_sensors = []
                for scd in scene_sample_data:
                    radar_calibrated_sensor = generate_calibrated_sensor(sensor_radar_vehicle_token, translation=vehicle_translation, rotation=vehicle_rotation, camera_intrinsic=[])
                    radar_sample_data = radar_sample_data_for_lidar_scene_sample_data(
                        path_data, 
                        path_output, 
                        scd, 
                        log["date_captured"], 
                        scene["name"], 
                        team,
                        radar_calibrated_sensor["token"]
                    )
                    if radar_sample_data is None:
                        continue
                    radar_sample_datas.append(radar_sample_data)
                    radar_calibrated_sensors.append(radar_calibrated_sensor)
                if len(radar_sample_datas) >= 0:
                    scene_sample_data.extend(radar_sample_datas)
                    calibrated_sensors.extend(radar_calibrated_sensors)

                instances.extend(generate_instances(scene_sample_annotations, categories[0]["token"], instance_map))
                scenes.append(scene)
                samples.extend(scene_samples)
                samples_annotations.extend(scene_sample_annotations)
                samples_data.extend(scene_sample_data)
    
    maps[0]["log_tokens"] = [log["token"] for log in logs]

    #
    # Dump NuScenes format
    #
    with open(os.path.join(path_json, "attribute.json"), 'w') as json_file:
        json.dump(attributes, json_file, indent=2)    
    with open(os.path.join(path_json, "calibrated_sensor.json"), 'w') as json_file:
        json.dump(calibrated_sensors, json_file, indent=2)
    with open(os.path.join(path_json, "category.json"), 'w') as json_file:
        json.dump(categories, json_file, indent=2)
    with open(os.path.join(path_json, "ego_pose.json"), 'w') as json_file:
        json.dump(ego_poses, json_file, indent=2)
    with open(os.path.join(path_json, "instance.json"), 'w') as json_file:
        json.dump(instances, json_file, indent=2)
    with open(os.path.join(path_json, "log.json"), 'w') as json_file:
        json.dump(logs, json_file, indent=2)
    with open(os.path.join(path_json, "map.json"), 'w') as json_file:
        json.dump(maps, json_file, indent=2)
    with open(os.path.join(path_json, "sample.json"), 'w') as json_file:
        json.dump(samples, json_file, indent=2)
    with open(os.path.join(path_json, "sample_annotation.json"), 'w') as json_file:
        json.dump(samples_annotations, json_file, indent=2)
    with open(os.path.join(path_json, "sample_data.json"), 'w') as json_file:
        json.dump(samples_data, json_file, indent=2)
    with open(os.path.join(path_json, "scene.json"), 'w') as json_file:
        json.dump(scenes, json_file, indent=2)
    with open(os.path.join(path_json, "sensor.json"), 'w') as json_file:
        json.dump(sensors, json_file, indent=2)
    with open(os.path.join(path_json, "visibility.json"), 'w') as json_file:
        json.dump(visibility, json_file, indent=2)

    #
    #
    #
    print("------------------- Summary ---------------------")
    print(f"Logs: {len(logs)}")
    print(f"Scenes: {len(scenes)}")
    print(f"Frames: {len(samples)}")
    print(f"Tracks: {len(instances)}")
    print(f"Annotations: {len(samples_annotations)}")
if __name__ == "__main__":
    import random
    random.seed(42)

    annotations_path = "./data/scene"
    path_data = "./data/sensors"
    path_output = "./data/v1.0-mini"

    INTENSITY_SCALE = 1
    detections = {
    }
    for datumaro_filepath in sorted(glob(f"{annotations_path}/*/*/*.json")):
        detection_json_to_map(detections, datumaro_filepath)

    os.makedirs(path_output, exist_ok=True)
    os.makedirs(os.path.join(path_output, "maps"), exist_ok=True)
    # from pathlib import Path
    # Path(os.path.join(path_output, "maps").join("53992ee3023e5494b90c316c183be829.png")).touch() # quickfix for now (not tested yet)
    path_json = os.path.join(path_output, "v1.0-mini")
    os.makedirs(path_json, exist_ok=True)
    os.makedirs(os.path.join(path_output, "samples"), exist_ok=True)
    os.makedirs(os.path.join(path_output, "sweeps"), exist_ok=True)
    detections_to_nuscenes(detections, path_json, path_data, path_output)
