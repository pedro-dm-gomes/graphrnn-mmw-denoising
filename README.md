# Graph-rnn for radar MMW denosing

Tensorflow Implementation of Graph-RNN for mmW (point cloud) denoising
We propose a new neural network with Graph-RNN cells, for point cloud sequence denoising

### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7.21
Compile the code. You will need to select the correct CUDA version and Tensorflow instaled in your computer. For that edit the Makefiles to the paths of your Cuda and Tensorflow directories.
The Makefiles to compile the code are in `modules/tf_ops`

### Datasets
The models were trained  evaluated with the following datasets:
1. [Labelled mmW Point Cloud (9 sequences, 1000 points)](https://drive.google.com/drive/folders/1dHwhJ9NzrVlGN91MHuaodX62HMzftcN8)
The dataset contains the sequences in original position (Not_Rotated) and tranlated acording to robot position  (Rotated).

## Usage

To train a model

    python train-mmnist-GraphRNN.py

To evaluate the model

    python eval-mmnist.py


#### Models
The models were evaluated with the following datasets:

To create the Human Bodies dataset follow the instruction in the Dataset folder.

### Graph-RNN modules

## Visual Results



#### Acknowledgement
The parts of this codebase is borrowed from Related Repos:
#### Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
3. Dynamic Graph CNN for Learning on Point Clouds https://github.com/WangYueFt/dgcnn
4. Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks https://github.com/jelmr/pc_temporal_interpolation

