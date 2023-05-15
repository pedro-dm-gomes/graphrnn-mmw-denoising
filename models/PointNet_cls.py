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
import tf_util
from transform_nets import input_transform_net, feature_transform_net

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
  
  #Single Frame processing there ar
  context_frames = 0
  
  print("[Load Module]: ",graph_module_name) # PointNET
  
  print("point_cloud.shape", point_cloud.shape)
  point_cloud = tf.reshape(point_cloud, (batch_size,seq_length*num_points, 3) )
  print("point_cloud.shape", point_cloud.shape)
  
  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
  point_cloud_transformed = tf.matmul(point_cloud, transform)
  input_image = tf.expand_dims(point_cloud_transformed, -1)
  print("input_image", input_image)

  
  net = tf_util.conv2d(input_image, 64, [1,3],
                      padding='VALID', stride=[1,1],
                      bn=BN_FLAG, is_training=is_training,
                      scope='conv1', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 64, [1,1],
                      padding='VALID', stride=[1,1],
                      bn=BN_FLAG, is_training=is_training,
                      scope='conv2', bn_decay=bn_decay)
  
  with tf.variable_scope('transform_net2') as sc:
      transform = feature_transform_net(net, is_training, bn_decay, K=64)
  end_points['transform'] = transform
  net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
  point_feat = tf.expand_dims(net_transformed, [2])
  print("point_feat",point_feat)

  net = tf_util.conv2d(point_feat, 64, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv3', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 128, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv4', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 1024, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv5', bn_decay=bn_decay)  
  
  global_feat = tf_util.max_pool2d(net, [num_points*seq_length,1],
                                    padding='VALID', scope='maxpool')
  print("global_feat", global_feat)

  global_feat_expand = tf.tile(global_feat, [1, num_points*seq_length, 1, 1])
  #concat_feat = tf.concat(3, [point_feat, global_feat_expand])
  concat_feat = tf.concat( (point_feat, global_feat_expand), axis =3 )
  print("concat_feat", concat_feat)

  net = tf_util.conv2d(concat_feat, 512, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv6', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 256, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv7', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 128, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv8', bn_decay=bn_decay)
  net_last = tf_util.conv2d(net, 50, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=BN_FLAG, is_training=is_training,
                        scope='conv9', bn_decay=bn_decay)

  net = tf_util.conv2d(net_last, 2, [1,1],
                        padding='VALID', stride=[1,1], activation_fn=None,
                         is_training=is_training, scope='conv10')
  
  net = tf.squeeze(net, [2]) # BxNxC

  net_last =  tf.reshape(net_last, (batch_size, seq_length, num_points,50 ))
  print("net_last", net_last) # (batch, frames, npoints, dims)
  
  end_points['last_d_feat'] = net_last 
  end_points['points'] = point_cloud 
  
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
  	
  sequence_loss = sequence_loss/(seq_length)
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
    frame_loss_0 = frame_loss * mask_0 * 0.5 # worth less
    frame_loss_1 = frame_loss * mask_1 * 1.5 # worth more
    frame_loss = frame_loss_0 + frame_loss_1

    frame_loss = tf.reduce_mean(frame_loss)
    sequence_loss = sequence_loss + frame_loss  	
  	
  sequence_loss = sequence_loss/(seq_length)
  return sequence_loss  

def get_MSE_loss(predicted_labels, ground_truth_labels, context_frames = 0 ): 
  """
  Gets predicted labels and gdt labels (numpy) and returns accuracy
  In : predicted_labels : Tensor (batch , seq_length, num_points, 2)
       ground_truth_labels : Tensor(batch , seq_length, num_points, 1) 
  Out: accuracy (1)
  """
  print("MSE loss")

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
    
    
    logits = tf.nn.softmax(logits)
    #logits = tf.nn.sigmoid(logits)

    
    logits = logits[:,:,0]
    
    logits = tf.reshape( logits,  [batch_size * num_points , 1] )
    labels = tf.reshape( labels,  [batch_size * num_points , 1] )
    
    print("logits", logits) # Tensor("Reshape_5:0", shape=(12800, 2), dtype=float32)
  
    binary_output = logits
    #binary_output = tf.cast(tf.greater(logits, 0.5), dtype=tf.float32)
    
    
    labels = tf.cast(labels, tf.float32)
    
    print("binary_output", binary_output)
    print("labels", labels)
    
    mse_loss = tf.losses.mean_squared_error(labels, binary_output)


    
    sequence_loss = sequence_loss +   mse_loss
    
    print("sequence_loss", mse_loss)
    
  
  sequence_loss = sequence_loss/(seq_length)
  
  return sequence_loss  


def get_acurracy_loss_2(predicted_labels, ground_truth_labels, context_frames = 0 ): 
  """
  Gets predicted labels and gdt labels (numpy) and returns accuracy
  In : predicted_labels : Tensor (batch , seq_length, num_points, 2)
       ground_truth_labels : Tensor(batch , seq_length, num_points, 1) 
  Out: accuracy (1)
  """
  print("MSE loss")

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
    
    logits = tf.nn.softmax(logits)
    
    logits = tf.reshape(logits, [-1, 2]) 
    labels = tf.reshape(labels, [-1, 1])
    print("logits", logits) # Tensor("Reshape_5:0", shape=(12800, 2), dtype=float32)
    print("labels", labels) # labels Tensor("Reshape_6:0", shape=(12800, 1),
    labels = tf.cast(labels, tf.float32)
    
    print("labels", labels) # labels Tensor("Reshape_6:0", shape=(12800, 1),
    
    mse_loss = tf.reduce_mean(tf.square(logits - labels))
    print("mse_loss", mse_loss)
    
    sequence_loss = sequence_loss +   mse_loss
  
  sequence_loss = sequence_loss/(seq_length)
  
  return sequence_loss  

