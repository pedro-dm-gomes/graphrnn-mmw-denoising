#FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04
FROM tensorflow/tensorflow:2.10.0-gpu

FROM ros:noetic AS ros_base


RUN apt update && apt upgrade -y && apt install -y python3 python3-pip

#RUN DEBIAN_FRONTEND=nonintercative python3 -m pip install tensorflow==2.12.0
RUN DEBIAN_FRONTEND=nonintercative python3 -m pip install pandas matplotlib 
RUN DEBIAN_FRONTEND=nonintercative python3 -m pip install protobuf==3.20.0

RUN DEBIAN_FRONTEND=noninteractive apt install -y wget gpg vim nano
RUN DEBIAN_FRONTEND=nonintercative python3 -m pip install cuda-python

RUN apt install -y git

RUN pip3 install matplotlib scipy jupyter numpy==1.23.0