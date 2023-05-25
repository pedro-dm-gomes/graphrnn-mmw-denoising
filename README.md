# Graph-RNN for Point Cloud Millimetre-wave (mmW) Denoising

Tensorflow Implementation of Graph-RNN for mmW (point cloud) denoising

### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7.21
Compile the code. You will need to select the correct CUDA version and Tensorflow instaled on your computer. For that edit the Makefiles to the paths of your Cuda and Tensorflow directories.
The Makefiles to compile the code are in `modules/tf_ops`

### Datasets
The models were trained and evaluated with the following datasets:
1. [Labelled mmW Point Cloud (50 sequences, 200 points per frame)][(https://drive.google.com/drive/folders/1dHwhJ9NzrVlGN91MHuaodX62HMzftcN8](https://drive.google.com/drive/folders/1rmNyCtOBnE2CfpLzESUh8q6zjPIi_DfS?usp=sharing))
The path to this folder should be given as input to the network.

Option 1\\
Place the dataset in a directory as such:"/Pc/Deskotp/Dataset/200/all_runs_final"
Give our path as input to train.py as  --data-dir argument example: "/Pc/Deskotp/Dataset/"

This option requires you to have "200/all_runs_final" diretory structure.


Option 2\\
You can change the python files in the dataset diretory to match you directory.

## Usage

To train and evaluate specific model and specific graph-rnn module

    # PointNET Baseline
    python  train-eval-split-mmW-GraphRNN.py   --version pointnet_v1 --bn_flag 1  --gpu 0 --graph_module PointNet --learning-rate 1e-5 --data-dir /home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset --batch-size 64 --out_channels 64 --num-points 200 --seq-length 1 --num-samples 8 --model PointNet_cls --context-frames 0 --restore-training 0

    # GraphRNN Model
    python train-eval-split-mmW-GraphRNN.py   --version v2 --bn_flag 0  --gpu 2 --graph_module Simple_GraphRNNCell  --learning-rate 1e-5 --data-dir /home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset --batch-size 8 --out_channels 64 --num-points 200 --seq-length 100 --num-samples 8 --model GraphRNN_cls  --down-points1 2 --down-points2 4 --down-points3 8 --drop_rate 0.4  --restore-training 0
    

### Models
The following  models are provided
1. **GraphRNN_cls**: Each Graph-RNN cell learns dynamic features from a Spatio-temporal graph build between two frames (t) and (t-1) 
3. **Stacked_GraphRNN_cls**: Each Graph-RNN cell learns dynamic features from a Spatio-temporal graph build between stacking frames: from frame (t, t-1, ... T_s)
 
### Graph-RNN Modules
The following graph-rnn cells are provided:
1. **Simple_GraphRNNCell**: Simple implementation of Graph-RNN cell (implemeted using tf.layers.conv2d)
2. **Simple_GraphRNNCell_util**:Simple implementation of Graph-RNN cell (implemeted using tf_util lib from [1]), There is a problem with batch normalization

## Visual Results



#### Acknowledgement
The parts of this codebase are borrowed from Related Repos:
#### Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
3. Dynamic Graph CNN for Learning on Point Clouds https://github.com/WangYueFt/dgcnn
4. Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks https://github.com/jelmr/pc_temporal_interpolation

