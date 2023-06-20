# Graph-RNN for Point Cloud Millimetre-wave (mmW) Denoising

Tensorflow Implementation of Graph-RNN for mmW (point cloud) denoising

### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 3.6, TensorFlow 1.12.0, CUDA 9.0 and cuDNN 7.21
Compile the code. You will need to select the correct CUDA version and Tensorflow instaled on your computer. For that edit the Makefiles to the paths of your Cuda and Tensorflow directories.
The Makefiles to compile the code are in `modules/tf_ops`

### Datasets
The models were trained and evaluated with the following datasets:
1. [Labelled mmW Point Cloud (50 sequences, 200 points per frame)][(https://drive.google.com/drive/folders/1dHwhJ9NzrVlGN91MHuaodX62HMzftcN8](https://drive.google.com/drive/folders/1rmNyCtOBnE2CfpLzESUh8q6zjPIi_DfS?usp=sharing))


Option 1<br />
Place the dataset in a directory structured as such ":/PC/Deskotp/Dataset/200/all_runs_final"
Give your path as input to train.py as  --data-dir argument example: "/Pc/Deskotp/Dataset/"
This option requires you to have "/200/all_runs_final" diretory structure.

Option 2<br />
You can change the python files in the dataset diretory to match you directory.

## Usage

To train and evaluate specific model and specific graph-rnn module

    # PointNET Model
    python train-eval-split-mmW-bari.py  --data-split 11  --version v1 --bn_flag 1  --gpu 0 --graph_module PointNet --learning-rate 1e-5 --data-dir /home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset --batch-size 32 --num-points 200 --seq-length 6 --num-samples 8 --model Stacked_PointNet_cls   --restore-training 0

    # DGCNN Model
    python train-eval-split-mmW-bari.py  --data-split 11  --version v2 --bn_flag 1  --gpu 0 --graph_module DGCNN ---learning-rate 1e-5 --data-dir /home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset --batch-size 32 --num-points 200 --seq-length 6 --num-samples 8 --model Stacked_DGCNN_cls    --restore-training 0


    # K-Hop GNN Model
    python train-eval-split-mmW-bari.py --data-split 11  --version v2 --bn_flag 0  --gpu 0 --graph_module GNN  --learning-rate 1e-5 --data-dir /home/uceepdg/profile.V6/Desktop/Datasets/Labelled_mmW/Not_Rotated_dataset --batch-size 32 --num-points 200 --seq-length 6 --num-samples 8 --model Stacked_Basic_GNN_cls   --restore-training 0

    # To evaluate
    python eval-debug=spli-mmW-bari.py -version v2  --bn_flag 1 --gpu 0   --graph_module GNN   --data-dir /scratch/uceepdg/Labelled_mmW/Not_Rotated_dataset --batch-size 1 --num-points 200 --seq-length  12 --num-samples 8 --model  Stacked_Basic_GNN_cls  --context-frames 0  --manual-restore 2    --data-split 11
    



#### Acknowledgement
The parts of this codebase are borrowed from Related Repos:
#### Related Repos
1. PointRNN TensorFlow implementation: https://github.com/hehefan/PointRNN
2. PointNet++ TensorFlow implementation: https://github.com/charlesq34/pointnet2
3. Dynamic Graph CNN for Learning on Point Clouds https://github.com/WangYueFt/dgcnn
4. Temporal Interpolation of Dynamic Point Clouds using Convolutional Neural Networks https://github.com/jelmr/pc_temporal_interpolation

