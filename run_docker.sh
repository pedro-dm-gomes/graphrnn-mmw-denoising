#!/bin/bash

IMAGE=millinoise:tf1
docker run --gpus all --rm -it --privileged --net=host --ipc=host \
	--device=/dev/dri:/dev/dri \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-e DISPLAY=$DISPLAY \
	-v $HOME/.Xauthority:/home/$(id -un)/.Xauthority \
	-e XAUTHORITY=/home/$(id -un)/.Xauthority \
    -v ./:/root/ws \
    -w /root/ws \
	$IMAGE bash