import os
import sys
import tensorflow as tf
import numpy as np
"""
Implement a PointNet Architecture like the paper

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
from transform_nets import input_transform_net, feature_transform_net

def placeholder_inputs(batch_size, seq_length, num_points):
  
 
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,seq_length, num_points, 3))
  rotated_pointcloud_pl = tf.placeholder(tf.float32, shape=(batch_size,seq_length, num_points, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size, seq_length, num_points,1 ))
  
  return pointclouds_pl, rotated_pointcloud_pl,  labels_pl
  
def get_model(point_cloud,  is_training, model_params):
  
  """ Classification Stacked PointNET++, input is BxNx3, output BxNx2 """
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
  

  #Stacked Frame processing
  context_frames = 0
  original_num_points =num_points
  
  print("[Load Module]: ",graph_module_name) # PointNET++
  
  
  # Add relative time-stamp to to point cloud
  timestep_tensor = tf.zeros( (batch_size,1,original_num_points,1) )
  for f in range(1, seq_length):
    frame_tensor = tf.ones( (batch_size,1,original_num_points,1) ) * f
    timestep_tensor = tf.concat( (timestep_tensor, frame_tensor) , axis = 1 )
    
      
  num_points = original_num_points *seq_length
  #point_cloud = tf.reshape(point_cloud, (batch_size, seq_length * original_num_points, 3) )
  #timestep_tensor = tf.reshape(timestep_tensor, (batch_size,seq_length *original_num_points, 1) )
  print("point_cloud.shape", point_cloud.shape)
  print("timestep_tensor.shape", timestep_tensor.shape)
  
  #### PointNET ++ 
  #l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
  l0_xyz_f1 = point_cloud[:,0,:,:]
  l0_rot_xyz_f1 = rotated_pointcloud[:,0,:,:]
  l0_points_f1 = timestep_tensor[:,0,:,:] #tf.zeros(l0_xyz.shape) 
  l0_xyz_f2 =  point_cloud[:,1,:,:]
  l0_rot_xyz_f2 = rotated_pointcloud[:,1,:,:]
  l0_points_f2 = timestep_tensor[:,1,:,:] #tf.zeros(l0_xyz.shape)   
  l0_xyz_f3 = point_cloud[:,2,:,:]
  l0_rot_xyz_f3 = rotated_pointcloud[:,2,:,:]
  l0_points_f3 = timestep_tensor[:,2,:,:] #tf.zeros(l0_xyz.shape)   
  
  print("\n")
  print("l0_xyz_f1", l0_xyz_f1)
  print("l0_rot_xyz_f1", l0_rot_xyz_f1)
  print("l0_points", l0_points_f1)

  
    
  # PointNet++ Layers
  with tf.variable_scope('sa1', reuse=tf.AUTO_REUSE) as scope:
    
    # Frame 1, 
    l1_xyz_f1, l1_points_f1, l1_indices_f1 = pointnet_sa_module(l0_xyz_f1, l0_points_f1,npoint=int(num_points/sampled_points_down1), radius=0.2, nsample=num_samples,  mlp=[64,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz_f1, l2_points_f1, l2_indices_f1 = pointnet_sa_module(l1_xyz_f1, l1_points_f1, npoint=int(num_points/sampled_points_down2), radius=0.2, nsample=num_samples,  mlp=[128,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz_f1, l3_points_f1, l3_indices_f1 = pointnet_sa_module(l2_xyz_f1, l2_points_f1, npoint=int(num_points/sampled_points_down3), radius=0.2, nsample=num_samples,  mlp=[128,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    
    scope.reuse_variables()
    # Frame 2
    l1_xyz_f2, l1_points_f2, l1_indices_f2 = pointnet_sa_module(l0_xyz_f2, l0_points_f2, npoint=int(num_points/sampled_points_down1), radius=0.2, nsample=num_samples,  mlp=[64,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz_f2, l2_points_f2, l1_indices_f2 = pointnet_sa_module(l1_xyz_f2, l1_points_f2,  npoint=int(num_points/sampled_points_down2), radius=0.2, nsample=num_samples,  mlp=[128,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz_f2, l3_points_f2, l3_indices_f2 = pointnet_sa_module(l2_xyz_f2, l2_points_f2, npoint=int(num_points/sampled_points_down3), radius=0.2, nsample=num_samples,  mlp=[128,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    
    scope.reuse_variables()
    # Frame 3
    l1_xyz_f3, l1_points_f3, l1_indices_f3 = pointnet_sa_module(l0_xyz_f3, l0_points_f3, npoint=int(num_points/sampled_points_down1), radius=0.2, nsample=num_samples,  mlp=[64,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz_f3, l2_points_f3, l1_indices_f3 = pointnet_sa_module(l1_xyz_f3, l1_points_f3, npoint=int(num_points/sampled_points_down2), radius=0.2, nsample=num_samples,  mlp=[128,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz_f3, l3_points_f3, l3_indices_f3 = pointnet_sa_module(l2_xyz_f3, l2_points_f3, npoint=int(num_points/sampled_points_down3), radius=0.2, nsample=num_samples,  mlp=[128,128], mlp2=None, group_all=False, knn= True, is_training=is_training, bn_decay=bn_decay, scope='layer3')
            

  print("l3_xyz_f1", l3_xyz_f1)
  print("l3_xyz_f2", l3_xyz_f2)
  print("l3_xyz_f3", l3_xyz_f3)
  
  # Merge all frames together
  l3_xyz_f1 = tf.expand_dims(l3_xyz_f1, axis=1)
  l3_xyz_f2 = tf.expand_dims(l3_xyz_f2, axis=1)
  l3_xyz_f3 = tf.expand_dims(l3_xyz_f3, axis=1)
  l2_xyz_f1 = tf.expand_dims(l2_xyz_f1, axis=1)
  l2_xyz_f2 = tf.expand_dims(l2_xyz_f2, axis=1)
  l2_xyz_f3 = tf.expand_dims(l2_xyz_f3, axis=1)
  l1_xyz_f1 = tf.expand_dims(l1_xyz_f1, axis=1)
  l1_xyz_f2 = tf.expand_dims(l1_xyz_f2, axis=1)
  l1_xyz_f3 = tf.expand_dims(l1_xyz_f3, axis=1)

  print("\nl3_xyz_f1", l3_xyz_f1)
  print("l3_xyz_f2", l3_xyz_f2)
  print("l3_xyz_f3", l3_xyz_f3)
  
  merged_xyz_3 = tf.concat( (l3_xyz_f1,l3_xyz_f2, l3_xyz_f3), axis =1)
  l3_xyz = tf.reshape(merged_xyz_3 ,  (merged_xyz_3.shape[0], merged_xyz_3.shape[1]* merged_xyz_3.shape[2], merged_xyz_3.shape[3] ))
  print("merged_xyz_3.shape", merged_xyz_3)
  merged_xyz_2 = tf.concat( (l2_xyz_f1,l2_xyz_f2, l2_xyz_f3), axis =1)
  l2_xyz = tf.reshape(merged_xyz_2 ,  (merged_xyz_2.shape[0], merged_xyz_2.shape[1]* merged_xyz_2.shape[2], merged_xyz_2.shape[3] ))
  print("merged_xyz_2.shape", merged_xyz_2)
  merged_xyz_1 = tf.concat( (l1_xyz_f1,l1_xyz_f2, l1_xyz_f3), axis =1)
  l1_xyz = tf.reshape(merged_xyz_1 ,  (merged_xyz_1.shape[0], merged_xyz_1.shape[1]* merged_xyz_1.shape[2], merged_xyz_1.shape[3] ))
  print("merged_xyz_1.shape", merged_xyz_1)
  l0_xyz = tf.concat( (l0_xyz_f1,l0_xyz_f1,l0_xyz_f3), axis =1 )
  print("l0_xyz.shape", l0_xyz)
  
  l0_points = tf.concat( (l0_points_f1,l0_points_f2,l0_points_f3), axis =1 )
  l1_points = tf.concat( (l1_points_f1,l1_points_f2,l1_points_f3), axis =1 )
  l2_points = tf.concat( (l2_points_f1,l2_points_f2,l2_points_f3), axis =1 )
  l3_points = tf.concat( (l3_points_f1,l3_points_f2,l3_points_f3), axis =1 )
  
  # Set Abstraction layers
  # no downsampleing
  l1_xyz, l1_points, l1_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=int(int(l1_points.shape[1])), knn= True, radius=0.2, nsample=num_samples,  mlp=[128,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
  l2_xyz, l2_points, l2_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=int(l2_points.shape[1]), knn= True, radius=0.4, nsample=num_samples, mlp=[128,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
  l3_xyz, l3_points, l3_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=int(l3_points.shape[1]), knn= True, radius=None, nsample=num_samples, mlp=[128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')  

  # Feature Propagation layers
  l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256], is_training, bn_decay, scope='fa_layer1')
  l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [128], is_training, bn_decay, scope='fa_layer2')
  l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128], is_training, bn_decay, scope='fa_layer3')

  
  print("l2_points", l2_points)
  print("l1_points", l1_points)
  print("l0_points", l0_points)
  

  
  # FC layers
  net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='fc1', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
  net = tf_util.conv1d(net, 2, 1, padding='VALID', activation_fn=None, scope='fc3')

  
  print("net", net)
  end_points['feats'] = net 
  predicted_labels = tf.reshape(net, (batch_size,seq_length,original_num_points, 2) )
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
  

 
