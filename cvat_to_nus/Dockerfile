FROM ubuntu:22.04

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt update && apt install -y python3 python3-pip

WORKDIR /opt
COPY convert_cvat_to_nus.py .

RUN pip install open3d pypcd_imp scipy tqdm pyquaternion