#!/bin/bash

xhost +
docker run --rm -it --privileged \
	--net host \
	--ipc host \
	--runtime nvidia \
	--gpus all \
	--device=/dev/dri:/dev/dri \
	-e DISPLAY=$DISPLAY \
        -e XAUTHORITY=/root/.Xauthority \
	-v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v ~/.Xauthority:/root/.Xauthority \
	-v ./:/root/ws/ \
	gnnmmwave bash
