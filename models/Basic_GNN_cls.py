import os
import sys
import tensorflow as tf
import numpy as np
"""
Implement a Basic GNN Architecture 

"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/grouping'))

from pointnet2_color_feat_states import *
import graph_rnn_modules as modules
import tf_util
from transform_nets import input_transform_net, feature_transform_net
from tf_grouping import query_ball_point, group_point, knn_point, knn_feat

def placeholder_inputs(batch_size, seq_length, num_points):
  
 
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,seq_length, num_points, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size, seq_length, num_points,1 ))
  
  return pointclouds_pl, labels_pl
  
def get_model(point_cloud, is_training, model_params):
  
  """ Classification PointNet, input is BxNx3, output BxNx2 """
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
  
  #Single Frame processing
  context_frames = 0
  
  print("[Load Module]: ",graph_module_name) # GNN
  
  print("point_cloud.shape", point_cloud.shape)
  point_cloud = tf.reshape(point_cloud, (batch_size, seq_length * num_points, 3) )
  print("point_cloud.shape", point_cloud.shape)
  
  #### PointNET ++ 
  #l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
  l0_xyz = point_cloud
  l0_points = l0_xyz #tf.zeros(l0_xyz.shape) 
  l0_feats = l0_points
  print("l0_xyz", l0_xyz)
  print("l0_points", l0_points)
  
  
  # Downsampling Layer 1
  l1_xyz, l1_color, l1_feat, l1_states, _, _ = sample_and_group(int(num_points/sampled_points_down1), 
                                                                radius=1.0+1e-8, 
                                                                nsample= 1, 
                                                                xyz=l0_xyz,  
                                                                color=l0_xyz, 
                                                                features=l0_points, 
                                                                states = l0_points, 
                                                                knn=True, 
                                                                use_xyz=False) 
  print("l1_xyz: ", l1_xyz)
  print("l1_feat: ", l1_feat)
  print("l1_states: ", l1_states)
  
  # Create adjacent matrix on cordinate space
  l1_adj_matrix = tf_util.pairwise_distance(l1_xyz)
  l1_nn_idx = tf_util.knn(l1_adj_matrix, k= num_samples) 
  print("l1_adj_matrix:", l1_adj_matrix)
  print("l1_nn_idx", l1_nn_idx)
  
  # Group Points
  l1_xyx_grouped = group_point(l1_xyz, l1_nn_idx)  
  print("l1_xyx_grouped:", l1_xyx_grouped)
  #l1_feat_grouped = group_point(l1_feat, l1_nn_idx)     
  #print("l1_feat_grouped:", l1_feat_grouped)
  
  # Calculate displacements
  l1_xyz_expanded = tf.expand_dims(l1_xyz, 2)
  l1_displacement = l1_xyx_grouped - l1_xyz_expanded   
  print("l1_displacement:", l1_displacement)
  print("l1_xyx_grouped:", l1_xyx_grouped)
  
  # Concatenate Message passing
  concatenation = tf.concat([l1_xyx_grouped, l1_displacement], axis=3)   
  print("concatenation", concatenation)
  
  # MLP message -passing
  with tf.variable_scope('GNN_1') as sc:
    l1_feats = tf_util.conv2d(concatenation, 
                              64, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l1', 
                              bn_decay=bn_decay)
    l1_feats = tf.reduce_max(l1_feats, axis=[2], keepdims=False)
        
  print("l1_feats", l1_feats)

  # Group Points
  l1_feats_gropued = group_point(l1_feats, l1_nn_idx)  
  
  # Concatenate Message passing
  concatenation = tf.concat([l1_xyx_grouped, l1_displacement, l1_feats_gropued], axis=3)   
  print("concatenation", concatenation)
  
  # MLP message -passing
  with tf.variable_scope('GNN_1') as sc:
    l1_feats = tf_util.conv2d(concatenation, 
                              128, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l2', 
                              bn_decay=bn_decay)
    l1_feats = tf.reduce_max(l1_feats, axis=[2], keepdims=False)
        
  print("l1_feats", l1_feats)
  
  

  # Downsampling Layer 2
  l2_xyz, l2_color, l2_feats, l2_states, _, _ = sample_and_group(int(num_points/sampled_points_down2), 
                                                                radius =10,
                                                                nsample= 1, 
                                                                xyz=l1_xyz,  
                                                                color=l1_xyz, 
                                                                features=l1_feats, 
                                                                states = l1_feats, 
                                                                knn=True, use_xyz=False) 
  l2_feats = tf.reduce_max(l2_feats, axis=[2], keepdims=False, name='maxpool')
  print("\n layer 2")
  print("l2_xyz: ", l2_xyz)
  print("l2_feat: ", l2_feats)
  
  # Create adjacent matrix on cordinate space
  l2_adj_matrix = tf_util.pairwise_distance(l2_xyz)
  l2_nn_idx = tf_util.knn(l2_adj_matrix, k= num_samples) 

  # Group Points
  l2_xyx_grouped = group_point(l2_xyz, l2_nn_idx) 
  l2_feats_grouped =   group_point(l2_feats, l2_nn_idx) 
  # Calculate displacements
  l2_xyz_expanded = tf.expand_dims(l2_xyz, 2)
  l2_displacement = l2_xyx_grouped - l2_xyz_expanded   
  
  print("l2_feats_grouped", l2_feats_grouped)
  print("l2_displacement", l2_displacement)
  print("l2_xyz_expanded", l2_xyz_expanded)
  
  # Concatenate Message passing
  concatenation = tf.concat([l2_xyx_grouped, l2_displacement,l2_feats_grouped ], axis=3)   
  
  # MLP message -passing
  with tf.variable_scope('GNN_2') as sc:
    l2_feats = tf_util.conv2d(concatenation, 
                              128, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l1', 
                              bn_decay=bn_decay)
    l2_feats = tf.reduce_max(l2_feats, axis=[2], keepdims=False)

  print("l2_feats", l2_feats)
  l2_feats_grouped =   group_point(l2_feats, l2_nn_idx) 

  # Concatenate Message passing
  concatenation = tf.concat([l2_xyx_grouped, l2_displacement,l2_feats_grouped ], axis=3)   
  
  # MLP message -passing
  with tf.variable_scope('GNN_2') as sc:
    l2_feats = tf_util.conv2d(concatenation, 
                              256, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l2', 
                              bn_decay=bn_decay)
    l2_feats = tf.reduce_max(l2_feats, axis=[2], keepdims=False)
      
  print("l2_feats", l2_feats)
        
  # Downsampling Layer 3
  l3_xyz, l3_color, l3_feats, l3_states, _, _ = sample_and_group(int(num_points/sampled_points_down3), 
                                                                radius =10,
                                                                nsample= 1, 
                                                                xyz=l2_xyz,  
                                                                color=l2_xyz, 
                                                                features=l2_feats, 
                                                                states = l2_feats, 
                                                                knn=True, use_xyz=False) 
  l3_feats = tf.reduce_max(l3_feats, axis=[2], keepdims=False, name='maxpool')
  print("\n layer 3")
  print("l3_xyz: ", l3_xyz)
  print("l3_feat: ", l3_feats)
  
  # Create adjacent matrix on cordinate space
  l3_adj_matrix = tf_util.pairwise_distance(l3_xyz)
  l3_nn_idx = tf_util.knn(l3_adj_matrix, k= num_samples) 

  # Group Points
  l3_xyx_grouped = group_point(l3_xyz, l3_nn_idx) 
  l3_feats_grouped =   group_point(l3_feats, l3_nn_idx) 
  # Calculate displacements
  l3_xyz_expanded = tf.expand_dims(l3_xyz, 2)
  l3_displacement = l3_xyx_grouped - l3_xyz_expanded   
  
  print("l3_feats_grouped", l3_feats_grouped)
  print("l3_displacement", l3_displacement)
  print("l3_xyz_expanded", l3_xyz_expanded)
  
  # Concatenate Message passing
  concatenation = tf.concat([l3_xyx_grouped, l3_displacement,l3_feats_grouped ], axis=3)   
  
  # MLP message -passing
  with tf.variable_scope('GNN_3') as sc:
    l3_feats = tf_util.conv2d(concatenation, 
                              256, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l1', 
                              bn_decay=bn_decay)
    l3_feats = tf.reduce_max(l3_feats, axis=[2], keepdims=False)
    
  print("l3_feats", l3_feats)          

  l3_feats_grouped =   group_point(l3_feats, l3_nn_idx) 
  concatenation = tf.concat([l3_xyx_grouped, l3_displacement,l3_feats_grouped ], axis=3)  
  # MLP message -passing
  with tf.variable_scope('GNN_3') as sc:
    l3_feats = tf_util.conv2d(concatenation, 
                              256, [1,1], 
                              padding='VALID', 
                              stride=[1,1], bn=BN_FLAG,  
                              is_training=is_training, 
                              activation_fn= tf.nn.relu,
                              scope = 'l2', 
                              bn_decay=bn_decay)
    l3_feats = tf.reduce_max(l3_feats, axis=[2], keepdims=False)
    
    
  # Downsampling Layer 4
  l4_xyz, l3_color, l4_feats, l3_states, _, _ = sample_and_group(int(20), 
                                                                radius =10,
                                                                nsample= 2, 
                                                                xyz=l3_xyz,  
                                                                color=l3_xyz, 
                                                                features=l3_feats, 
                                                                states = l3_feats, 
                                                                knn=True, use_xyz=False) 
  l4_feats = tf.reduce_max(l4_feats, axis=[2], keepdims=False, name='maxpool')
  print("\n layer 4")
  print("l4_xyz: ", l4_xyz)
  print("l4_feats: ", l4_feats)
  
  # Create global vector
  global_feat = tf.reshape(l4_feats, ( l4_feats.shape[0], 1, l4_feats.shape[1]  * l4_feats.shape[2] ))
  global_feat = tf_util.conv1d(global_feat, 1024, 1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='global_1', bn_decay=bn_decay)
  print("global_feat", global_feat)
  
  """ ---- Feature Propagation  --- """
  
  # Concatenate global features with 20 last frames
  global_feat_repeat = tf.tile(global_feat, [1, l4_feats.shape[1], 1])
  print("global_feat_repeat", global_feat_repeat)
  
  l4_feats_and_global = tf.concat(( l4_feats,global_feat_repeat ), axis = 2 )
  
  with tf.variable_scope('fp', reuse=tf.AUTO_REUSE) as scope:
    l2_prop_feat = pointnet_fp_module_original(l3_xyz,
                                          l4_xyz,
                                          l3_feats,
                                          l4_feats_and_global,
                                          mlp=[256,128],
                                          last_mlp_activation=True,
                                          scope='fp2') 
    
    print("l2_prop_feat", l2_prop_feat)
    l1_prop_feat = pointnet_fp_module_original(l1_xyz,
                                          l2_xyz,
                                          l1_feats,
                                          l2_prop_feat,
                                          mlp=[128],
                                          last_mlp_activation=True,
                                          scope='fp1') 
    
    print("l1_prop_feat", l1_prop_feat)
    l0_prop_feat = pointnet_fp_module_original(l0_xyz,
                                          l0_xyz,
                                          l0_feats,
                                          l1_prop_feat,
                                          mlp=[128],
                                          last_mlp_activation=True,
                                          scope='fp0') 
    print("l0_prop_feat", l0_prop_feat)
        
          
  # FC layers
  net = tf_util.conv1d(l0_prop_feat, 128, 1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='fc1', bn_decay=bn_decay)
  net = tf_util.conv1d(net, 64, 1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='fc2', bn_decay=bn_decay)
  net_last = net
  #net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp1')
  net = tf_util.conv1d(net, 2, 1, padding='VALID', activation_fn=None, scope='fc3')

  print("net_last", net_last.shape)
  net_last = tf.reshape(net_last, (batch_size, seq_length, num_points,64 ))
  end_points['last_d_feat'] = net_last 
    
  print("net", net)
  end_points['feats'] = net 
  
  predicted_labels = tf.reshape(net, (batch_size,seq_length,num_points, 2) )
  print("predicted_labels", predicted_labels)


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
  	
  sequence_loss = sequence_loss/(seq_length )
  return sequence_loss 
  

def get_balanced_loss(predicted_labels, ground_truth_labels, context_frames):
  """ Calculate balanced loss 
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

    #print("--- Normal Classification ---")
    frame_loss =0
    frame_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    frame_loss = tf.cast(frame_loss, tf.float32)
    
    labels = tf.cast(labels, tf.float32)
    print("---Weighted Classification ---")
    mask_0 = tf.where(labels < 0.5, tf.ones_like(labels), tf.zeros_like(labels))  # 0 -Noise points
    mask_1 = tf.where(labels > 0.5, tf.ones_like(labels), tf.zeros_like(labels))  # 1 -Clean points
    mask_0 = tf.cast(mask_0, tf.float32)
    mask_1 = tf.cast(mask_1, tf.float32)
    
    print("mask_0", mask_0)
    print("frame_loss", frame_loss)
    frame_loss_0 = frame_loss * mask_0 * 0.7 # worth less
    frame_loss_1 = frame_loss * mask_1 * 1.3 # worth more
    frame_loss = frame_loss_0 + frame_loss_1

    frame_loss = tf.reduce_mean(frame_loss)
    sequence_loss = sequence_loss + frame_loss  	
  	
  sequence_loss = sequence_loss/(seq_length)
  return sequence_loss  

