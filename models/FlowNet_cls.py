import os
import sys
import tensorflow as tf
import numpy as np
"""
Implement a FLowNet Architecture like the paper

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
from flow_net_modules import flow_embedding_module, set_upconv_module
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, seq_length, num_points):
  
 
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,seq_length, num_points, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size, seq_length, num_points,1 ))
  
  return pointclouds_pl, labels_pl
  
def get_model(point_cloud, is_training, model_params):
  
  """ Classification FlowNet, input is BxFxNx3, output BxFxNx2 """
  # Get model parameters
  batch_size = point_cloud.get_shape()[0].value
  seq_length =point_cloud.get_shape()[1].value
  num_points = point_cloud.get_shape()[2].value
  sampled_points = num_points
  num_samples = model_params['num_samples']
  context_frames = model_params['context_frames']
  sampled_points_down1 = model_params['sampled_points_down1'] 
  sampled_points_down2 = model_params['sampled_points_down2'] 
  sampled_points_down3 = model_params['sampled_points_down3']
  BN_FLAG = model_params['BN_FLAG']
  bn_decay = model_params['bn_decay']
  out_channels = model_params['out_channels'] #use?
  drop_rate = model_params['drop_rate']
  graph_module_name = model_params['graph_module']
  end_points = {}
  
  #Single Frame processing there ar
  context_frames = 0
  graph_module_name =  "Flow Net"
  print("[Load Module]: ",graph_module_name)
  
  l0_xyz_f1 = point_cloud[:,0, : ,:] #frame 1
  l0_xyz_f2 = point_cloud[:,1, : ,:] #frame 2
  l0_points_f1 = l0_xyz_f1
  l0_points_f2 = l0_xyz_f2
  predicted_labels = tf.zeros(( point_cloud.shape[0], point_cloud.shape[1], point_cloud.shape[2], 2))
  print("l0_xyz_f1:", l0_xyz_f1)
  print("l0_xyz_f2", l0_xyz_f2)
  
  # PointNet++ Layers
  with tf.variable_scope('sa1', reuse=tf.AUTO_REUSE) as scope:
    
    # Frame 1, 
    l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1, npoint=sampled_points_down1, radius=0.2, nsample=num_samples,  mlp=[64,64,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz_f1, l2_points_f1, l1_indices_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=sampled_points_down2, radius=0.2, nsample=num_samples,  mlp=[64,64,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    
    # Frame 2, Layer 1
    l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=sampled_points_down1, radius=0.2, nsample=num_samples,  mlp=[64,64,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz_f2, l2_points_f2, l1_indices_f2 = pointnet_sa_module(l1_xyz_f2, l1_points_f2, npoint=sampled_points_down2, radius=0.2, nsample=num_samples,  mlp=[64,64,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer2')

  
  print("\n")
  # Flow Embeddign module
  _, l2_points_f1_new = flow_embedding_module(l2_xyz_f1, l2_xyz_f2, l2_points_f1, l2_points_f2, radius=3.0, nsample=num_samples * 2 , mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding_f1', bn=True, pooling='max', knn=True, corr_func='concat')
  print("l2_points_f1_new:", l2_points_f1_new)  # batch, points, frames, 
  
  # Flow Embeddign module
  _, l2_points_f2_new = flow_embedding_module(l2_xyz_f2, l2_xyz_f1, l2_points_f2, l2_points_f1_new, radius=3.0, nsample=num_samples * 2 , mlp=[128,128,128], is_training=is_training, bn_decay=bn_decay, scope='flow_embedding_f2', bn=True, pooling='max', knn=True, corr_func='concat')
  print("l2_points_f2_new:", l2_points_f2_new)  # batch, points, frames, 
  
  l3_feat_f1 = set_upconv_module(l1_xyz_f1, l2_xyz_f1, l1_points_f1, l2_points_f1_new, nsample=8, radius=2.4, mlp=[], mlp2=[128,128,64], scope='up_sa_layer_f1', is_training=is_training, bn_decay=bn_decay, knn=True)
  l3_feat_f2 = set_upconv_module(l1_xyz_f1, l2_xyz_f2, l1_points_f1, l2_points_f2_new, nsample=8, radius=2.4, mlp=[], mlp2=[128,128,64], scope='up_sa_layer_f2', is_training=is_training, bn_decay=bn_decay, knn=True)
  print("l3_feat_f1:", l3_feat_f1)  # batch, points, frames, 
  print("l3_feat_f2:", l3_feat_f2)  # batch, points, frames, 
  
  # FC layers
  with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
    net1 = tf_util.conv1d(l3_feat_f1, 2, 1, padding='VALID', bn=False, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net2 = tf_util.conv1d(l3_feat_f2, 2, 1, padding='VALID', bn=False, is_training=is_training, activation_fn=None, scope='fc1', bn_decay=bn_decay)
  
  print("net1", net1)
  print("net2", net2)
  net1 = tf.expand_dims(net1, axis=1)   
  net2 = tf.expand_dims(net2, axis=1)   
  
  print("\nnet1", net1)
  print("net2", net2)

  # Reshape vector
  predicted_labels = tf.concat( (net1, net2), axis =1 )
        
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
  	
  sequence_loss = sequence_loss/(seq_length)
  return sequence_loss 
  

 
