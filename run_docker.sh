#!/bin/bash

IMAGE=millinoise:tf1_suflur
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

# rm /usr/lib/x86_64-linux-gnu/libcuda.so.1
# ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.384.130 /usr/lib/x86_64-linux-gnu/libcuda.so.1
# export PATH="/usr/local/cuda-9.0/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH"
# find /usr/lib -iname "*nvidia*560*"


#python3 
#rm /usr/lib/x86_64-linux-gnu/libcuda.so.1
#ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.384.183 /usr/lib/x86_64-linux-gnu/libcuda.so.1
#python3 
#nvidia-smi
#python3 
#nvcc -V
#export CUDA_HOME=/usr/local/cuda-9.0/
#python3 
#export PATH="/usr/local/cuda-9.0/bin:$PATH"
#export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH"
#python3 
#echo $CUDA_HOME/
#echo $CUDA_PKG_VERSION 
#echo $CUDA_VERSION 
#echo $CUDNN_VERSION 
#pip3 install -U tensorflow-gpu==1.12.0
#pip3 install -U tensorflow==1.12.0
#python3
#dpkg -l | grep nvrm
#dpkg -l | grep nvr
#dpkg -l | grep nv
#dpkg --purge nvidia-465
#dpkg --purge nvidia-modprobe
#dpkg --purge nvidia-opencl-icd-465
#dpkg --purge nvidia-settings
#python3
#nvidia-smi
#rm /usr/lib/x86_64-linux-gnu/libcuda.so.1
#ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.384.130 /usr/lib/x86_64-linux-gnu/libcuda.so.1
#export PATH="/usr/local/cuda-9.0/bin:$PATH"
#export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH"
#nvidia-smi
#rm /usr/lib/x86_64-linux-gnu/libcuda.so.1
#ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.384.183 /usr/lib/x86_64-linux-gnu/libcuda.so.1
#nvidia-smi
#python3
#dpkg -l | grep 560
#dpkg -l | grep 56
#dpkg -l | grep 5
#dpkg -l | grep 384
#dpkg --get-selections | grep nvidia
#apt search nvidia
#apt search nvidia | grep 384
#apt install nvidia-libopencl1-384
#python3
#nvidia-smi
#rm /usr/lib/x86_64-linux-gnu/libcuda.so.1
#ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.384.183 /usr/lib/x86_64-linux-gnu/libcuda.so.1
#printenv | grep CUDA
#printenv | grep CUD
#export PATH="/usr/local/cuda-9.0/bin:$PATH"
#export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH"
#nvidia-smi
#python3
#dmesg 
#nvidia-cuda-mps-server 
#dmesg 
#dpkg -l | grep nvidida
#dpkg -l | grep nvidia
#find /usr/lib -iname "*nvidia*530*"
#find /usr/lib -iname "*nvidia*560*"
#ls -lha /usr/lib/x86_64-linux-gnu/libnvidia*
#rm /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
#ln -s /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.384.183 /usr/lib/x86_64-linux-gnu/libnvidia-opencl.so.1
#nvidia-smi
#python3
#find /usr/lib -iname "*nvidia*560*"
#apt search libnvidia-nv
#ls -lha /usr/lib/x86_64-linux-gnu/libnvidia*
#ls -lha /usr/lib/x86_64-linux-gnu/cuda*
#ls -lha /usr/lib/x86_64-linux-gnu/*cuda*
#ls -lha /usr/lib/x86_64-linux-gnu/*cudnn*
#dpkg -l | grep cudnn
#dpkg --purge libcudnn7
#dpkg --purge libcudnn7-dev
#dpkg --purge libcudnn7
#cd /tmp && wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
#dpkg -i /tmp/libcudnn7_7.0.3.11-1+cuda9.0_amd64.deb
#dpkg -l | grep cudnn
#nvidia-smi
#rm /usr/lib/x86_64-linux-gnu/libcuda.so.1
#ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.384.183 /usr/lib/x86_64-linux-gnu/libcuda.so.1
#nvidia-smi
#python3
#printenv | grep cud
#printenv | grep -i cud
#nvcc -V
#export CUDNN_VERSION=7.0.3.11
#python3
#find /usr/lib -iname "*nvidia*560*"
#find / -iname "*nvidia*560*"
#find / -iname "*560*"
#ls -lha /lib/firmware/nvidia/