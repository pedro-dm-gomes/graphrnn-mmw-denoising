#!/bin/bash

IMAGE=millinoise:tf2
docker run --rm -it --privileged --net=host --ipc=host \
    --runtime=nvidia \
    --gpus all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev:/dev \
    -e DISPLAY=$DISPLAY \
    -v $HOME/.Xauthority:/home/$(id -un)/.Xauthority \
    -e XAUTHORITY=/home/$(id -un)/.Xauthority \
    -v ./:/root/ws \
    -w /root/ws \
    $IMAGE bash