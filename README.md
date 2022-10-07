# Graph-rnn for radar MMW denosing

Tensorflow Implementation of Graph-RNN for mmW (point cloud) denoising
We propose a new neural network with Graph-RNN cells, for point cloud sequence denoising

### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7.21
Compile the code. You will need to select the correct CUDA version and Tensorflow instaled in your computer. For that edit the Makefiles to the paths of your Cuda and Tensorflow directories.
The Makefiles to compile the code are in `modules/tf_ops`

### Datasets
The models were evaluated with the following datasets:
1. [Moving MNIST Point Cloud (1 digit)](https://drive.google.com/open?id=17RpNwMLDcR5fLr0DJkRxmC5WgFn3RwK_) &emsp; 
2. [Moving MNIST Point Cloud (2 digits)](https://drive.google.com/open?id=11EkVsE5fmgU5D5GsOATQ6XN17gmn7IvF) &emsp; 
3. [JPEG Dynamic Human Bodies (4000 points)](https://drive.google.com/file/d/1hbB1EPKq3UVlXUL5m81M1E6_s5lWmoB-/view)

To create the Human Bodies dataset follow the instruction in the Dataset folder.

## Usage




#### MNIST
To train a model

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

