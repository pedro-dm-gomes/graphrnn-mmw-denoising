import os
import sys
import tensorflow as tf
import numpy as np
"""
Implement a DGCNN Architecture 
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))

from pointnet2_color_feat_states import *
import graph_rnn_modules as modules
import tf_util
import tf_util
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, seq_length, num_points):
  
 
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,seq_length, num_points, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size, seq_length, num_points,1 ))
  
  return pointclouds_pl, labels_pl
  
def get_model(point_cloud, is_training, model_params):
  
  """ Classification DGCNNNet, input is BxNx3, output BxNx2 """
  # Get model parameters
  batch_size = point_cloud.get_shape()[0].value
  seq_length =point_cloud.get_shape()[1].value
  num_points = point_cloud.get_shape()[2].value
  sampled_points = num_points
  num_samples = model_params['num_samples']
  context_frames = model_params['context_frames']
  sampled_points_down1 = model_params['sampled_points_down1'] #not used
  sampled_points_down2 = model_params['sampled_points_down2'] #not used
  
  sampled_points_down3 = model_params['sampled_points_down3'] #not used
  BN_FLAG = model_params['BN_FLAG']
  bn_decay = model_params['bn_decay']
  out_channels = model_params['out_channels'] #use?
  drop_rate = model_params['drop_rate']
  graph_module_name = model_params['graph_module']
  end_points = {}
  
  
  print("[Load Module]: DGCNN") # DGCNN
  print("point_cloud.shape", point_cloud.shape)
  point_cloud = tf.reshape(point_cloud, (batch_size*seq_length,num_points, 3) )
  print("[new] point_cloud.shape", point_cloud.shape)
  
  #Single Frame processing 
  context_frames = 0
  num_point = num_points
  k =   num_samples
  weight_decay = bn_decay
  input_image = tf.expand_dims(point_cloud, -1) 
  
  adj = tf_util.pairwise_distance(point_cloud[:, :, :])
  nn_idx = tf_util.knn(adj, k=k) # (batch, num_points, k)
  edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

  out1 = tf_util.conv2d(edge_feature, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
  out2 = tf_util.conv2d(out1, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training, weight_decay=weight_decay,
                        scope='adj_conv2', bn_decay=bn_decay, is_dist=True)

  net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)


  adj = tf_util.pairwise_distance(net_1)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

  out3 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=BN_FLAG, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv3', bn_decay=bn_decay, is_dist=True)

  out4 = tf_util.conv2d(out3, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=BN_FLAG, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv4', bn_decay=bn_decay, is_dist=True)
  
  net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)
  
  
  adj = tf_util.pairwise_distance(net_2)
  nn_idx = tf_util.knn(adj, k=k)
  edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

  out5 = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=BN_FLAG, is_training=is_training, weight_decay=weight_decay,
                       scope='adj_conv5', bn_decay=bn_decay, is_dist=True)


  net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)



  out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=BN_FLAG, is_training=is_training,
                       scope='adj_conv7', bn_decay=bn_decay, is_dist=True)

  out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')


  expand = tf.tile(out_max, [1, num_point, 1, 1])

  concat = tf.concat(axis=3, values=[expand, 
                                     net_1,
                                     net_2,
                                     net_3])

  # CONV 
  net = tf_util.conv2d(concat, 128, [1,1], padding='VALID', stride=[1,1],
             bn=BN_FLAG, is_training=is_training, scope='seg/conv1', is_dist=True)
  net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
             bn=BN_FLAG, is_training=is_training, scope='seg/conv2', is_dist=True)
  net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
  net = tf_util.conv2d(net, 2, [1,1], padding='VALID', stride=[1,1],
             activation_fn=None, scope='seg/conv3', is_dist=True)
  net = tf.squeeze(net, [2])
  
  print("net:", net)

  predicted_labels = tf.reshape(net, (batch_size,seq_length,num_points, 2) )
  print("predicted_labels", predicted_labels, "\n")

  return predicted_labels, end_points
       

def get_loss(predicted_labels, ground_truth_labels, context_frames):

  """ Calculate loss 
   inputs: predicted labels : 
   	   ground_truth_labels: (batch, seq_length, num_points, 1)
   	   predicted_labels : (batch,seq_length, num_points, 2
  """
  batch_size = ground_truth_labels.get_shape()[0].value
  seq_length = ground_truth_labels.get_shape()[1].value
  num_points = ground_truth_labels.get_shape()[2].value
  
  # Convert labels to a list - This can be improved but it works
  ground_truth_labels = tf.split(value = ground_truth_labels , num_or_size_splits=seq_length, axis=1)
  ground_truth_labels = [tf.squeeze(input=label, axis=[1]) for label in ground_truth_labels] 
  
  predicted_labels = tf.split(value = predicted_labels , num_or_size_splits=seq_length, axis=1) 
  predicted_labels = [tf.squeeze(input=label, axis=[1]) for label in predicted_labels]  
  
  sequence_loss = 0
  #Calculate loss frame by frame
  for frame in range(context_frames,seq_length ):
    logits = predicted_labels[frame]
    labels = ground_truth_labels[frame]

    logits = tf.reshape(logits, [batch_size * num_points , 2])
    labels = tf.reshape(labels, [batch_size*num_points ,])
    labels = tf.cast(labels, tf.int32)

    frame_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    frame_loss = tf.reduce_mean(frame_loss)
    sequence_loss = sequence_loss + frame_loss  	
  	
  sequence_loss = sequence_loss/(seq_length *batch_size )
  return sequence_loss 
  

 
