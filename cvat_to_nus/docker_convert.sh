#!/bin/bash

docker run \
    -v $PWD/../data:/opt/data \
    -it cvat_to_nus
