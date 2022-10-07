# Graph-rnn for mmw denosing

Tensorflow Implementation of Graph-RNN for mmW (point cloud) denoising


We propose a new neural network with Graph-RNN cells, for point cloud sequence denoising


[[Project]](https://github.com/pedro-dm-gomes/Graph-RNN) [[Paper]](https://arxiv.org/abs/2102.07482)     


## Overview
<img src="https://github.com/pedro-dm-gomes/Graph-RNN/blob/main/full_scheme.png" scale="0.2">

## Citation
Please cite this paper if you want to use it in your work,

	@article{gomes2021spatiotemporal,
	  title={Spatio-temporal Graph-RNN for Point Cloud Prediction},
	  author={Pedro Gomes and Silvia Rossi and Laura Toni},
	  year={2021},
	  eprint={2102.07482},
	  archivePrefix={arXiv},
	  }
### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7.21

Compile the code. You will need to select the correct CUDA version and Tensorflow instaled in your computer. For that edit the Makefiles to the paths of your Cuda and Tensorflow directories.
The Makefiles to compile the code are in `modules/tf_ops`

### Usage
#### MNIST
To train a model to long-term prediction using the GraphRNN

    python train-mmnist-GraphRNN.py

To evaluate the model

    python eval-mmnist.py

#### Human Bodies 
to train the model without color or with color

    python train-bodies-GraphRNN.py
    python train-bodies-GraphRNN_color.py

to evaluate

    python eval-bodies.py
    python eval-bodies_color.py

### Datasets
The models were evaluated with the following datasets:

To create the Human Bodies dataset follow the instruction in the Dataset folder.

## Visual Results



## Acknowledgement
The parts of this codebase is borrowed from Related Repos:

### Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
3. Dynamic Graph CNN for Learning on Point Clouds https://github.com/WangYueFt/dgcnn
4. Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks https://github.com/jelmr/pc_temporal_interpolation

