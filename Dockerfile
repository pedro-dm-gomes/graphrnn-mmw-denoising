FROM nvidia/cuda:11.6.1-runtime-ubuntu20.04

RUN apt update && apt install -y python3 python3-pip

RUN DEBIAN_FRONTEND=nonintercative python3 -m pip install tensorflow==2.12.0
RUN DEBIAN_FRONTEND=nonintercative python3 -m pip install numpy pandas matplotlib 
RUN DEBIAN_FRONTEND=nonintercative python3 -m pip install protobuf==3.20.0

RUN DEBIAN_FRONTEND=noninteractive apt install -y wget gpg vim nano
RUN DEBIAN_FRONTEND=nonintercative python3 -m pip install cuda-python

