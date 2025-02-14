FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV NVENCODE_CFLAGS "-I/usr/local/cuda/include"
ENV DEBIAN_FRONTEND=noninteractive

# Get all dependencies
RUN apt-get update && apt-get install -y \
    git zip unzip libssl-dev libcairo2-dev lsb-release libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev software-properties-common \
    build-essential cmake pkg-config libapr1-dev autoconf automake libtool curl libc6 libboost-all-dev debconf libomp5 libstdc++6 \
    libqt5core5a libqt5xml5 libqt5gui5 libqt5widgets5 libqt5concurrent5 libqt5opengl5 libcap2 libusb-1.0-0 libatk-adaptor neovim \
    python3-pip python3-tornado python3-dev python3-numpy python3-virtualenv libpcl-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev \
    libsuitesparse-dev python3-pcl pcl-tools libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev libtbb2 libtbb-dev libjpeg-dev \
    libpng-dev libtiff-dev libdc1394-dev xfce4-terminal &&\
    rm -rf /var/lib/apt/lists/*

# PyTorch for CUDA 11.8
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.7 8.9+PTX"
    
# OpenPCDet
# Fiexed Kornia version for bug
RUN pip3 install numpy==1.23.0 llvmlite numba tensorboardX easydict pyyaml scikit-image tqdm SharedArray open3d mayavi av2 kornia pyquaternion opencv-python==4.8.0.76
RUN pip3 install spconv-cu118

WORKDIR /opt

RUN git clone https://github.com/ybcljwaisq/jpamjfhkhn.git /opt/ar_dataset ####

WORKDIR /opt/ar_dataset/a2rl_OpenPCDet
RUN python3 setup.py develop
RUN pip install kornia==0.5.8
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

WORKDIR /opt/ar_dataset/a2rl_nuscenes_devkit
RUN cd setup && pip install -e .

WORKDIR /opt/ar_dataset
