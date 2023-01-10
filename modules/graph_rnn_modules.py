import os
import sys
import numpy as np
import tensorflow as tf
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))


from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point, knn_feat
from tf_interpolate import three_nn, three_interpolate
import tf_util


"""
Modules Index:
	1. Simple_GraphRNNCell : 
	Simplified version of Graph-RNN does message passing convolution
	
	2.Simple_GraphRNNCell_util:
	-- Equal to Simple_GraphRNNCell, uses tf_util, has batch normalization included in the function


"""

"""
================================================== 
              Simple_GraphRNNCell                       
==================================================
"""
class Simple_GraphRNNCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=True,
                 BN_FLAG = False,
                 pooling='max',
                 activation= None):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.BN_FLAG = BN_FLAG
        self.pooling = pooling
        self.activation = activation

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: 
            P1: cordinates in t
            C1: color in t
            F1: spatial feature in t
            T1: time step in t
            bn_decay: is not used
            is_traiining: is not used
        Returns:
            A tube of tensors representing the learned states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, C, F, X, T, bn_decay, is_training = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]
        #inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        #C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        C = None
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)

        return (P, C, F, S, T)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1, bn_decay, is_training= inputs
        P2, C2, F2, S2, T2 = states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        activation = self.activation
        BN_FLAG = self.BN_FLAG
    
        """ Comment this for debug """
        #print("== GraphRNN Operation ===") 
        #print("P1:",P1)  # time t
        #print("P2:",P2)  # time t-1
        #print("C1:",C1)
        #print("C2:",C2)
        #print("F1:",F1)
        #print("F2:",F2)
        #print("X1:",X1) #S1
        #print("S2:",S2)
        #print("T1:",T1)
        #print("T2:",T2)
        k_mulipler_P2 = 3
        nsample_P1 = nsample 
        nsample_P2 = nsample * k_mulipler_P2      
        """ Create adjacent matrix on feature space F1 """
        P1_adj_matrix = tf_util.pairwise_distance(F1)
        P1_nn_idx = tf_util.knn(P1_adj_matrix, k= nsample_P1)
        """ Create adjacent matrix on feature space F2 """
       	P2_adj_matrix = tf_util.pairwise_distance_2point_cloud(F2, F1)
       	P2_nn_idx = tf_util.knn(P2_adj_matrix, k = nsample_P2)
        
        if (knn == False) : # DO A BALL QUERY
        	pritn(" [Error] BALL QUERY NOT IMPLEMENTED ")
        	exit()

        """ Group points acording to Adjancy matrix """
        # 2.1 Group P1 points
        P1_grouped = group_point(P1, P1_nn_idx)                      
        T1_grouped = group_point(T1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.2 Group P1 states
        if (X1 is not None):
        	S1_grouped = group_point(X1, P1_nn_idx)  
        # 2.1 Group P2 points
        P2_grouped = group_point(P2, P2_nn_idx)                      
        #2.4 Group S2 states
        S2_grouped = group_point(S2, P2_nn_idx)   
        # 2.4 Group P2 time
        T2_grouped = group_point(T2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels

        ##  Neighborhood P1
        # 3. Calculate displacements
        P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
        displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        T1_expanded = tf.expand_dims(T1, 2)                     # batch_size, npoint, 1,       3
        displacement_time = T1_grouped - T1_expanded           # batch_size, npoint, nsample, 3
        ##  Neighborhood P2 
        # 3. Calculate displacements
        P2_expanded = tf.expand_dims(P2, 2)                     # batch_size, npoint, 1,       3
        displacement_2 = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        T2_expanded = tf.expand_dims(T2, 2)                     # batch_size, npoint, 1,       3
        displacement_time_2 = T2_grouped - T1_expanded           # batch_size, npoint, nsample, 3

        """  4. Concatenate X1, S2 and displacement """
        if X1 is not None:
        	""" Concatenation (t) = [ P1_expanded | S_ij | displacement_Pij | displacement_Fij| displacement_T] """
        	X1_expanded_to_P1 = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample_P1, 1])  
        	X1_expanded_to_P2 = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample_P2, 1])  
        	P1_expanded_to_P1 = tf.tile(tf.expand_dims(P1, 2), [1, 1, nsample_P1, 1])
        	P1_expanded_to_P2 = tf.tile(tf.expand_dims(P1, 2), [1, 1, nsample_P2, 1])       

        	concatenation = tf.concat([P1_expanded_to_P1,X1_expanded_to_P1, S1_grouped], axis=3)         
        	concatenation = tf.concat([concatenation, displacement, displacement_time], axis=3)
        	concatenation_2 = tf.concat([P1_expanded_to_P2,X1_expanded_to_P2, S2_grouped], axis=3)
        	concatenation_2 = tf.concat([concatenation_2, displacement_2, displacement_time_2], axis=3)    
        else:
        	""" Concatenation (t-1) = [P1_expanded, displacement_Pij | displacement_Fij| displacement_T] """
        	P1_expanded_to_P1 = tf.tile(tf.expand_dims(P1, 2), [1, 1, nsample_P1, 1])   
        	P1_expanded_to_P2 = tf.tile(tf.expand_dims(P1, 2), [1, 1, nsample_P2, 1])   

        	concatenation = tf.concat([P1_expanded_to_P1,P1_grouped,displacement, displacement_time], axis=3)
        	concatenation_2 = tf.concat([P1_expanded_to_P2,P2_grouped,displacement_2,displacement_time_2], axis=3)        

        #Unifty both concatenations
        concatenation = tf.concat([concatenation, concatenation_2], axis=2)


        """  5. Fully-connected layer (the only parameters) """
        with tf.variable_scope('graph-rnn') as sc:
        	S1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation, name='fc')
        	# S1   # batch_size, npoint, nsample,out_channels
        	if(BN_FLAG):
        		S1_batch_norm = tf.layers.batch_normalization(S1, training=is_training, name='batch_norm')
        	else:
        		S1_batch_norm = S1
        	
        	#print("S1:", S1)
        	#print("S1_batch_norm:", S1_batch_norm)

        """  6. Pooling """
        if pooling=='max':
        	S1 = tf.reduce_max(S1_batch_norm, axis=[2], keepdims=False)
        elif pooling=='avg':
        	S1 =tf.reduce_mean(S1_batch_norm, axis=[2], keepdims=False)  
        	

        return (P1, C1, F1, S1, T1) 

"""
================================================== 
              Simple_GraphRNNCell_util                       
==================================================
"""
class Simple_GraphRNNCell_util(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=True,
                 BN_FLAG = False,
                 pooling='max',
                 bn_decay = None,
                 activation= None):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.BN_FLAG = BN_FLAG
        self.pooling = pooling
        self.activation = activation

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: 
            P1: cordinates in t
            C1: color in t
            F1: spatial feature in t
            T1: time step in t
            bn_decay: is not used
            is_traiining: is not used
        Returns:
            A tube of tensors representing the learned states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, C, F, X, T, bn_decay, is_training = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]
        #inferred_feature_dimensions = 128 # ASSUMPTION
        
        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        #C = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        C = None
        #F = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=dtype)
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)


        return (P, C, F, S, T)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, C1, F1, X1, T1, bn_decay, is_training= inputs
        P2, C2, F2, S2, T2 = states
        
        radius = self.radius
        nsample = self.nsample
        out_channels = self.out_channels
        knn = self.knn
        pooling = self.pooling
        activation = self.activation
        BN_FLAG = self.BN_FLAG
        k_mulipler_P2 = 3
        nsample_P1 = nsample 
        nsample_P2 = nsample * k_mulipler_P2   
        
        """ Comment this for debug """
        #print("== GraphRNN Operation ===") 
        #print("P1:",P1)  # time t
        #print("P2:",P2)  # time t-1
        #print("C1:",C1)
        #print("C2:",C2)
        #print("F1:",F1)
        #print("F2:",F2)
        #print("X1:",X1) #S1
        #print("S2:",S2)
        #print("T1:",T1)
        #print("T2:",T2)
                
        """ Create adjacent matrix on feature space F1 """
        P1_adj_matrix = tf_util.pairwise_distance(F1)
        P1_nn_idx = tf_util.knn(P1_adj_matrix, k= nsample_P1)
        """ Create adjacent matrix on feature space F2 """
       	P2_adj_matrix = tf_util.pairwise_distance_2point_cloud(F2, F1)
       	P2_nn_idx = tf_util.knn(P2_adj_matrix, k= nsample_P2)
        
        if (knn == False) : # DO A BALL QUERY
        	print("\n [ERROR] BALL QUERY NOT IMPLEMENTED!! ")
        	exit()

        """ Group points acording to Adjancy matrix """
        # 2.1 Group P1 points
        P1_grouped = group_point(P1, P1_nn_idx)                      
        T1_grouped = group_point(T1, P1_nn_idx)                       # batch_size, npoint, nsample, out_channels
        # 2.2 Group P1 states
        if (X1 is not None):
        	S1_grouped = group_point(X1, P1_nn_idx)  
        # 2.1 Group P2 points
        P2_grouped = group_point(P2, P2_nn_idx)                      
        #2.4 Group S2 states
        S2_grouped = group_point(S2, P2_nn_idx)   
        # 2.4 Group P2 time
        T2_grouped = group_point(T2, P2_nn_idx)                       # batch_size, npoint, nsample, out_channels

        ##  Neighborhood P1
        # 3. Calculate displacements
        P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
        displacement = P1_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        T1_expanded = tf.expand_dims(T1, 2)                     # batch_size, npoint, 1,       3
        displacement_time = T1_grouped - T1_expanded           # batch_size, npoint, nsample, 3
        ##  Neighborhood P2 
        # 3. Calculate displacements
        P2_expanded = tf.expand_dims(P2, 2)                     # batch_size, npoint, 1,       3
        displacement_2 = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3
        T2_expanded = tf.expand_dims(T2, 2)                     # batch_size, npoint, 1,       3
        displacement_time_2 = T2_grouped - T1_expanded           # batch_size, npoint, nsample, 3

        """  4. Concatenate X1, S2 and displacement """
        if X1 is not None:
        	""" Concatenation (t) = [ P1_expanded | S_ij | displacement_Pij | displacement_Fij| displacement_T] """
        	X1_expanded_to_P1 = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample_P1, 1])  
        	X1_expanded_to_P2 = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample_P2, 1])  
        	P1_expanded_to_P1 = tf.tile(tf.expand_dims(P1, 2), [1, 1, nsample_P1, 1])
        	P1_expanded_to_P2 = tf.tile(tf.expand_dims(P1, 2), [1, 1, nsample_P2, 1])       

        	concatenation = tf.concat([P1_expanded_to_P1,X1_expanded_to_P1, S1_grouped], axis=3)         
        	concatenation = tf.concat([concatenation, displacement, displacement_time], axis=3)
        	concatenation_2 = tf.concat([P1_expanded_to_P2,X1_expanded_to_P2, S2_grouped], axis=3)
        	concatenation_2 = tf.concat([concatenation_2, displacement_2, displacement_time_2], axis=3)    
        else:
        	""" Concatenation (t-1) = [P1_expanded, displacement_Pij | displacement_Fij| displacement_T] """
        	P1_expanded_to_P1 = tf.tile(tf.expand_dims(P1, 2), [1, 1, nsample_P1, 1])   
        	P1_expanded_to_P2 = tf.tile(tf.expand_dims(P1, 2), [1, 1, nsample_P2, 1])   

        	concatenation = tf.concat([P1_expanded_to_P1,P1_grouped,displacement, displacement_time], axis=3)
        	concatenation_2 = tf.concat([P1_expanded_to_P2,P2_grouped,displacement_2,displacement_time_2], axis=3)        

        #Unifty both concatenations
        concatenation = tf.concat([concatenation, concatenation_2], axis=2)


        """  5. Fully-connected layer (the only parameters) """
        with tf.variable_scope('graph-rnn') as sc:
        	#S1 = tf.layers.conv2d(inputs=concatenation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=activation,  name='fc')
        	S1 = tf_util.conv2d(inputs=concatenation, num_output_channels=out_channels,kernel_size = [1,1],  stride=[1,1], padding='VALID', is_training=is_training, activation_fn=None, bn = BN_FLAG, bn_decay = bn_decay, scope = 'fc')
        
        # 6. Pooling
        if pooling=='max':
        	S1 = tf.reduce_max(S1, axis=[2], keepdims=False)
        elif pooling=='avg':
        	S1 =tf.reduce_mean(S1, axis=[2], keepdims=False)  
        	
        return (P1, C1, F1, S1, T1) 


