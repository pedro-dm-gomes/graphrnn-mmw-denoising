"""
PointNet++ Operations and Layers
Original Author: Charles R. Qi
Modified by Hehe Fan
Data September 2019
"""

import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/dgcnn_utils'))

from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate


import tf_util

def flow_embedding_module(xyz1, xyz2, feat1, feat2, radius, nsample, mlp, is_training, bn_decay, scope, bn=True, pooling='max', knn=True, corr_func='elementwise_product'):
    """
    Input:
        xyz1: (batch_size, npoint, 3)
        xyz2: (batch_size, npoint, 3)
        feat1: (batch_size, npoint, channel)
        feat2: (batch_size, npoint, channel)
    Output:
        xyz1: (batch_size, npoint, 3)
        feat1_new: (batch_size, npoint, mlp[-1])
    """
    if knn:
        _, idx = knn_point(nsample, xyz2, xyz1)
    else:
        idx, cnt = query_ball_point(radius, nsample, xyz2, xyz1)
        _, idx_knn = knn_point(nsample, xyz2, xyz1)
        cnt = tf.tile(tf.expand_dims(cnt, -1), [1,1,nsample])
        idx = tf.where(cnt > (nsample-1), idx, idx_knn)

    xyz2_grouped = group_point(xyz2, idx) # batch_size, npoint, nsample, 3
    xyz1_expanded = tf.expand_dims(xyz1, 2) # batch_size, npoint, 1, 3
    xyz_diff = xyz2_grouped - xyz1_expanded # batch_size, npoint, nsample, 3

    feat2_grouped = group_point(feat2, idx) # batch_size, npoint, nsample, channel
    feat1_expanded = tf.expand_dims(feat1, 2) # batch_size, npoint, 1, channel
    # TODO: change distance function
    if corr_func == 'elementwise_product':
        feat_diff = feat2_grouped * feat1_expanded # batch_size, npoint, nsample, channel
    elif corr_func == 'concat':
        feat_diff = tf.concat(axis=-1, values=[feat2_grouped, tf.tile(feat1_expanded,[1,1,nsample,1])]) # batch_size, npoint, sample, channel*2
    elif corr_func == 'dot_product':
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
    elif corr_func == 'cosine_dist':
        feat2_grouped = tf.nn.l2_normalize(feat2_grouped, -1)
        feat1_expanded = tf.nn.l2_normalize(feat1_expanded, -1)
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
    elif corr_func == 'flownet_like': # assuming square patch size k = 0 as the FlowNet paper
        batch_size = xyz1.get_shape()[0].value
        npoint = xyz1.get_shape()[1].value
        feat_diff = tf.reduce_sum(feat2_grouped * feat1_expanded, axis=[-1], keep_dims=True) # batch_size, npoint, nsample, 1
        total_diff = tf.concat(axis=-1, values=[xyz_diff, feat_diff]) # batch_size, npoint, nsample, 4
        feat1_new = tf.reshape(total_diff, [batch_size, npoint, -1]) # batch_size, npoint, nsample*4
        #feat1_new = tf.concat(axis=[-1], values=[feat1_new, feat1]) # batch_size, npoint, nsample*4+channel
        return xyz1, feat1_new


    feat1_new = tf.concat([feat_diff, xyz_diff], axis=3) # batch_size, npoint, nsample, [channel or 1] + 3
    # TODO: move scope to outer indent
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='conv_diff_%d'%(i), bn_decay=bn_decay)
    if pooling=='max':
        feat1_new = tf.reduce_max(feat1_new, axis=[2], keep_dims=False, name='maxpool_diff')
    elif pooling=='avg':
        feat1_new = tf.reduce_mean(feat1_new, axis=[2], keep_dims=False, name='avgpool_diff')
    return xyz1, feat1_new

def set_upconv_module(xyz1, xyz2, feat1, feat2, nsample, mlp, mlp2, is_training, scope, bn_decay=None, bn=True, pooling='max', radius=None, knn=True):
    """
        Feature propagation from xyz2 (less points) to xyz1 (more points)
    Inputs:
        xyz1: (batch_size, npoint1, 3)
        xyz2: (batch_size, npoint2, 3)
        feat1: (batch_size, npoint1, channel1) features for xyz1 points (earlier layers)
        feat2: (batch_size, npoint2, channel2) features for xyz2 points
    Output:
        feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)
        TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
    """
    with tf.variable_scope(scope) as sc:
        if knn:
            l2_dist, idx = knn_point(nsample, xyz2, xyz1)
        else:
            idx, pts_cnt = query_ball_point(radius, nsample, xyz2, xyz1)
        xyz2_grouped = group_point(xyz2, idx) # batch_size, npoint1, nsample, 3
        xyz1_expanded = tf.expand_dims(xyz1, 2) # batch_size, npoint1, 1, 3
        xyz_diff = xyz2_grouped - xyz1_expanded # batch_size, npoint1, nsample, 3

        feat2_grouped = group_point(feat2, idx) # batch_size, npoint1, nsample, channel2
        net = tf.concat([feat2_grouped, xyz_diff], axis=3) # batch_size, npoint1, nsample, channel2+3

        if mlp is None: mlp=[]
        for i, num_out_channel in enumerate(mlp):
            net = tf_util.conv2d(net, num_out_channel, [1,1],
                                 padding='VALID', stride=[1,1],
                                 bn=True, is_training=is_training,
                                 scope='conv%d'%(i), bn_decay=bn_decay)
        if pooling=='max':
            feat1_new = tf.reduce_max(net, axis=[2], keep_dims=False, name='maxpool') # batch_size, npoint1, mlp[-1]
        elif pooling=='avg':
            feat1_new = tf.reduce_mean(net, axis=[2], keep_dims=False, name='avgpool') # batch_size, npoint1, mlp[-1]

        if feat1 is not None:
            feat1_new = tf.concat([feat1_new, feat1], axis=2) # batch_size, npoint1, mlp[-1]+channel1

        feat1_new = tf.expand_dims(feat1_new, 2) # batch_size, npoint1, 1, mlp[-1]+channel2
        if mlp2 is None: mlp2=[]
        for i, num_out_channel in enumerate(mlp2):
            feat1_new = tf_util.conv2d(feat1_new, num_out_channel, [1,1],
                                       padding='VALID', stride=[1,1],
                                       bn=True, is_training=is_training,
                                       scope='post-conv%d'%(i), bn_decay=bn_decay)
        feat1_new = tf.squeeze(feat1_new, [2]) # batch_size, npoint1, mlp2[-1]
        return feat1_new