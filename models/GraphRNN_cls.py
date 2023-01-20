import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))

from pointnet2_color_feat_states import *
import graph_rnn_modules as modules
#from graph_rnn_modules import *
import tf_nndistance
import tf_approxmatch
import tf_util

def placeholder_inputs(batch_size, seq_length, num_points):
  
 
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,seq_length, num_points, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size, seq_length, num_points,1 ))
  
  return pointclouds_pl, labels_pl
  
def get_model(point_cloud, is_training, model_params):
  
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
  out_channels = model_params['out_channels']
  drop_rate = model_params['drop_rate']
  graph_module_name = model_params['graph_module']
  end_points = {}
  
   
  print("BN_FLAG:",BN_FLAG)
  print("[Load Graph Module]: ",graph_module_name)
  graph_module = getattr(modules, graph_module_name)
  
  graph_cell1 = graph_module(radius= 0.1, nsample=num_samples, out_channels=out_channels, knn= True, pooling='max', BN_FLAG = BN_FLAG)
  graph_cell2 = graph_module(radius= 0.1, nsample=num_samples, out_channels=out_channels, knn= True, pooling='max', BN_FLAG = BN_FLAG)
  graph_cell3 = graph_module(radius= 0.1, nsample=num_samples, out_channels=out_channels, knn= True, pooling='max', BN_FLAG = BN_FLAG)
  
 
  frames = tf.split(value = point_cloud , num_or_size_splits=seq_length, axis=1)
  frames = [tf.squeeze(input=frame, axis=[1]) for frame in frames]
  
    
  # Initialize Variables
  global_state1 = None
  global_state2 = None
  global_state3 = None
  predicted_labels = []
  ground_truth_labels = []


  
  for i in range(int(seq_length) ):  #Interate for all the Frames
  
  	input_frame = frames[i]
  	input_frame_color = input_frame
  	xyz0 = input_frame
  	
  	""" ===  Dynamic Extraction Phase """
  	
  	""" Downsampling Layer 1 """
  	xyz1, color1, feat1, states1, _, _ = sample_and_group(int(sampled_points_down1), radius=1.0+1e-8, nsample= 4, xyz=xyz0,  color=input_frame_color, features=None, states = None, knn=True, use_xyz=False) 
  	feat1 = xyz1 # tf.reduce_max(feat1, axis=[2], keepdims=False, name='maxpool')
  	states1 = tf.reduce_max(states1, axis=[2], keepdims=False, name='maxpool')                       
  	time = tf.fill( (xyz1.shape[0],xyz1.shape[1],1), (i/1.0))
  	
  	""" Graph-rnn cell 1 """
  	with tf.variable_scope('graphrnn_1', reuse=tf.AUTO_REUSE) as scope:
  		global_state1 = graph_cell1( (xyz1, None, feat1, None, time, bn_decay, is_training), global_state1)
  		s_xyz1, s_color1, s_feat1, s_states1, time  = global_state1
  		
  	"""  Downsampling Layer 2 """
  	xyz2, color2, feat2, states2, _, _ = sample_and_group(int(sampled_points_down2), radius=1.0+1e-20, nsample= 4 , xyz=s_xyz1,  color=s_xyz1, features=s_feat1, states = s_states1, knn=True, use_xyz=False)
  	feat2 = xyz2# tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
  	states2 = tf.reduce_max(states2, axis=[2], keepdims=False, name='maxpool')
  	time = tf.fill( (xyz2.shape[0],xyz2.shape[1],1), (i/1.0))
  	
  	""" Graph-rnn cell 2 """
  	with tf.variable_scope('graphrnn_2', reuse=tf.AUTO_REUSE) as scope:
  		global_state2 = graph_cell2( (xyz2, None, feat2, states2, time, bn_decay, is_training), global_state2)
  		s_xyz2, s_color2, s_feat2, s_states2, time = global_state2
  		
  	"""  Downsampling Layer 3 """
  	xyz3, color3, feat3, states3, _, _ = sample_and_group(int(sampled_points_down3), radius=4.0+1e-20, nsample= 4, xyz=s_xyz2,  color=s_xyz2, features=s_feat2, states = s_states2, knn=True, use_xyz=False)
  	feat3 = xyz3 #tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
  	states3 = tf.reduce_max(states3, axis=[2], keepdims=False, name='maxpool')
  	time = tf.fill( (xyz3.shape[0],xyz3.shape[1],1), (i/1.0))
  	
  	""" Graph-rnn cell 3 """
  	with tf.variable_scope('graphrnn_3', reuse=tf.AUTO_REUSE) as scope:
  		global_state3 = graph_cell3( (xyz3, None, feat3, states3, time, bn_decay, is_training), global_state3)
  		s_xyz3, s_color3,s_feat3, s_states3, time = global_state3
  		

  	""" ===  Features Progation Phase ===== """
  	with tf.variable_scope('fp', reuse=tf.AUTO_REUSE) as scope:  	
                l2_feat,_ = pointnet_fp_module_original_interpolated(xyz2,
                                             xyz3,
                                             s_states2,
                                             s_states3,
                                             mlp=[out_channels],
                                             BN_FLAG = BN_FLAG,
                                             last_mlp_activation=True,
                                             is_training=is_training,
                                             scope='fp2')
                                             
                l1_feat,_ = pointnet_fp_module_original_interpolated(xyz1,
                                             xyz2,
                                             s_states1,
                                             l2_feat,
                                             mlp=[out_channels],
                                             BN_FLAG= BN_FLAG,
                                             last_mlp_activation=True,
                                             is_training=is_training,
                                             scope='fp1')
                                             
                l0_feat,_ = pointnet_fp_module_original_interpolated(xyz0,
                                             xyz1,
                                             None,
                                             l1_feat,
                                             mlp=[out_channels],
                                             BN_FLAG = BN_FLAG,
                                             last_mlp_activation=True,
                                             is_training=is_training,
                                             scope='fp0')
                        

  	"""  === Fully Connected layers ===  """ 
  	with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
  		
  		""" normal tf implementation """
  		# FC 1 Layer
  		net = tf.layers.conv1d(inputs=l0_feat, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc1')
  		if (BN_FLAG): net_norm = tf.layers.batch_normalization(net, training=is_training, name='fc1_bn')
  		else: net_norm = net
  		net_norm = tf.nn.relu(net_norm)

  		#Dropout-layer
  		if(drop_rate > 0.0): predicted_label = tf_util.dropout(net_norm, keep_prob= drop_rate, is_training=is_training, scope='dp2')
  		# FC 2 Layer
  		predicted_label = tf.layers.conv1d(inputs=predicted_label, filters=2, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc2')
  		
  		
  		""" tf_util implementation """
  		#predicted_label  = tf_util.conv1d(inputs = l0_feat, num_output_channels=out_channels, kernel_size=1, stride=1, padding='VALID', bn=BN_FLAG, is_training=is_training, scope='fc1', bn_decay=bn_decay)
  		#Dropout-layer
  		#drop_layer = tf.layers.Dropout(drop_rate, predicted_label.shape)
  		#if(drop_rate > 0.0): predicted_label = tf_util.dropout(predicted_label, keep_prob= drop_rate, is_training=is_training, scope='dp2')
  		# FC 2 Layer
  		#predicted_label = tf_util.conv1d(inputs = predicted_label, num_output_channels=2, kernel_size=1, stride=1, padding='VALID', bn=False, is_training=is_training, scope='fc2', bn_decay=bn_decay)
 		
  	predicted_labels.append(predicted_label)

  predicted_labels = tf.stack(values=predicted_labels, axis=1)

  return  predicted_labels,  end_points 	  	


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
  

 
