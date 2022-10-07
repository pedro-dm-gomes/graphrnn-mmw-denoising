# Graph-RNN for Point Cloud Millimetre-wave (mmW) Denoising

Tensorflow Implementation of Graph-RNN for mmW (point cloud) denoising
We propose a new neural network with Graph-RNN cells, for point cloud sequence denoising

### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7.21
Compile the code. You will need to select the correct CUDA version and Tensorflow instaled on your computer. For that edit the Makefiles to the paths of your Cuda and Tensorflow directories.
The Makefiles to compile the code are in `modules/tf_ops`

### Datasets
The models were trained and evaluated with the following datasets:
1. [Labelled mmW Point Cloud (9 sequences, 1000 points)](https://drive.google.com/drive/folders/1dHwhJ9NzrVlGN91MHuaodX62HMzftcN8)
The dataset contains the sequences in the original position (Not_Rotated) and translated according to the robot position  (Rotated).

## Usage

To train a model

    python train-eval-mmW-GraphRNN.py 

To train a specific model and specific graph-rnn module

    python train-eval-mmW-GraphRNN.py --model GraphRNN_cls --graph_module Simple_GraphRNNCell --version r1  --data-dir /Datasets/Labelled_mmW/Rotated_dataset 

To evaluate the model

    python eval-mmW  --model GraphRNN_cls --graph_module Simple_GraphRNNCell --version r1 

### Models
The following  models are provided
1. **GraphRNN_cls**: Each Graph-RNN cell learns dynamic features from a Spatio-temporal graph build between two frames (t) and (t-1) 
3. **Stacked_GraphRNN_cls**: Each Graph-RNN cell learns dynamic features from a Spatio-temporal graph build between stacking frames: from frame (t, t-1, ... T_s)
 
### Graph-RNN Cells/Modules
The following graph-rnn cells are provided:
1. **Simple_GraphRNNCell**: Simple implementation of Graph-RNN cell (implemeted using tf.layers.conv2d)
2. **Simple_GraphRNNCell_bn**:Simple implementation of Graph-RNN cell with Batch normalization (implemeted using tf_util lib from [1])
3. **Displacement_GraphRNNCell_bn**: Equal to Simple_GraphRNNCell_bn, but does not considers point cordinates $(x_i,y_i,z_i)$ during the convolution, only the dipalcementes $(x_i-x_j, y_i-y_j, z_i - z_j)$

## Visual Results



#### Acknowledgement
The parts of this codebase are borrowed from Related Repos:
#### Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
3. Dynamic Graph CNN for Learning on Point Clouds https://github.com/WangYueFt/dgcnn
4. Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks https://github.com/jelmr/pc_temporal_interpolation

