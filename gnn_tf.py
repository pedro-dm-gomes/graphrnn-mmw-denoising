#!/usr/bin/env python3

import tensorflow as tf
import keras
from keras import layers

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers

import numpy as np
import json
import os

from utils import load_train_test_data, get_data_and_label2

from utils import gnn_conv2d, gnn_dense, \
    lambda_get_adj_matr, lambda_knn, lambda_edge_feature, \
    CustomOrthogonalRegularizer, gnn_tnet, custom_loss, ValAccThresh_CB

# Where datasets are stored
run_path = "data/"
experiments_path = "experiments/"
DEVICE="GPU"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



def build_model(inputs, 
        num_points, num_dims, k,
        tnet_shape,
        conv_gnns,
        dense_gnn,
        classes=1):
    '''
        Returns the outputs of the model to be compiled.
        ---
        Arguments: 
        * inputs:       Expected (None, 3). instance of tf.Input.
        * num_points:   Number of points per point cloud. Default is 200
        * num_dims:     Number of dimensions per point. Default is 3 (x, y, z)
        * k:            K nearest neighbors
        * tnet_shape:   Array of three lists. (each list's length is the number of layers for that section)
                        1st is a list of filters for convolutional layers before reduce_max.
                        2nd is a list of filters for convolutional layers after reduce_max.
                        3rd is a list of neurons for dense layers after max pooling.
        * conv_gnns:    list. Each row is composed of two lists.
                        1st is a list of filters for convolutional layers before computing edge features.
                        2nd is a list of filters for convolutional layers after computing edge features.
        * dense_gnn:    list of neuorns for dense layers at the end of the network.
        * classes:      number of classes to classify.
    '''

    # adj = lambda_get_adj_matr(inputs)
    # nn_idxs = lambda_knn(adj, k)
    # edge_feats = lambda_edge_feature(inputs, nn_idxs, k, num_points, num_dims)
    # x = gnn_dense(inputs, 8, bn=False)
    x = inputs
    feat_T = gnn_tnet(x, num_dims, tnet_shape[0], bn=False)
    # print(" [main] tnet shape: ", inputs.shape, feat_T.shape)
    pc_tf = layers.Dot(axes=(-1, -2))([x, feat_T]) # Apply affine transformation to input features
    # print(" [main] dot shape: ", pc_tf.shape)

    # adj = lambda_get_adj_matr(pc_tf)
    # nn_idxs = lambda_knn(adj, k)
    # edge_feats = lambda_edge_feature(pc_tf, nn_idxs, k, num_points, num_dims)
    # x = edge_feats
    x = pc_tf

    for l in conv_gnns:

        for gc_filt in l[0]:
            x = gnn_conv2d(x, gc_filt, [1], bn=False)
        
        feat_T = gnn_tnet(x, l[0][-1], tnet_shape[1], bn=False)
        # print(" [main] tnet shape: ", inputs.shape, feat_T.shape)
        x = layers.Dot(axes=(-1, -2))([x, feat_T]) # Apply affine transformation to input features
        # x = tf.reduce_max(x, axis=-2, keepdims=True)
        # x = layers.Lambda(lambda y: tf.reshape(y[0], (y[1], num_points, l[0][-1])))( [x, tf.shape(inputs)[0]] )

        # adj = lambda_get_adj_matr(x)
        # nn_idxs = lambda_knn(adj, k)
        # edge_feats = lambda_edge_feature(x, nn_idxs, k, num_points, l[0][-1])
        # x = edge_feats

        # feat_T = gnn_tnet(x, x.shape[-1], tnet_shape[1], bn=True)
        # print(" [main] tnet SHAPES: ", x.shape, feat_T.shape)
        # x = layers.Dot(axes=(-1, -2))([x, feat_T])
        # print(" [main] DOT SHAPE: ", x.shape)

        # adj = lambda_get_adj_matr(x)
        # nn_idxs = lambda_knn(adj, k)
        # edge_feats = lambda_edge_feature(x, nn_idxs, k, num_points, x.shape[-1])

        for gc_filt in l[1]:
            x = gnn_conv2d(x, gc_filt, [1], bn=False)
        # x = tf.reduce_max(x, axis=-2, keepdims=True)
        # x = layers.Lambda(lambda y: tf.reshape(y[0], (y[1], num_points, l[1][-1])))( [x, tf.shape(inputs)[0]] )

    # print(tf.shape(x))
    # x = layers.GlobalMaxPooling2D(keepdims=True)(x)
    # x = layers.Lambda(lambda y: tf.reshape(y[0], (y[1], y[2])))( [x, x.shape[0], x.shape[-1]] )
    # print(tf.shape(x))
    # x = layers.Lambda(lambda y: tf.reshape(y[0], (y[1], y[2])))( [x, tf.shape(x)[0], x.shape[-1]] )
    # x = layers.GlobalMaxPooling1D()(x)
    # print(" [main] shape: ", x.shape)
    for w_ in dense_gnn:
        x = gnn_dense(x, w_, bn=False)
        # x = layers.Dropout(0.3)(x)

    # outputs = layers.Dense(classes, activation="softmax")(x)
    outputs = layers.Dense(classes, activation="linear")(x)
    # print(tf.shape(outputs))
    return outputs


class CustomMetric(tf.keras.metrics.Accuracy):

  def __init__(self, name='custom_metric', thresh=0.1, **kwargs):
    super(CustomMetric, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')
    # self.accuracy_comp = tf.keras.metrics.sparse_categorical_crossentropy()
    self.thresh = thresh
    self.acc = 0.

  def update_state(self, y_true, y_pred, sample_weight=None):
    # print(y_true.shape, y_pred.shape)
    # y_true = tf.cast((y_true <= self.thresh), tf.bool)
    # y_pred = tf.cast((tf.squeeze(y_pred) <= self.thresh), tf.bool)
    # y_true = tf.squeeze(tf.cast(y_true == 0, tf.int32))
    # y_pred = tf.squeeze(tf.cast(y_pred == 0, tf.int32))
    y_true = tf.cast(y_true <= self.thresh, tf.int32)
    y_pred = tf.cast(y_pred <= self.thresh, tf.int32)

    # print(y_true.shape, y_pred.shape)
    super().update_state(y_true, y_pred, sample_weight)
    # self.acc = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
    # tf.print("[CustomMetric/UpdateState]: Acc: ", self.acc)
    # values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    # values = tf.cast(values, self.dtype)
    # if sample_weight is not None:
    #   sample_weight = tf.cast(sample_weight, self.dtype)
    #   sample_weight = tf.broadcast_to(sample_weight, values.shape)
    #   values = tf.multiply(values, sample_weight)
    # self.true_positives.assign_add(tf.reduce_sum(values))

#   def result(self):
#     # return self.true_positives
#     return self.acc


if __name__ == "__main__":

    valid_filter = [9, 69, 73, 3, 55]
    train_filter=[4, 5, 7, 8, 49, 51, 54, 58, 59, 62, 63, 65, 66, 68, 71, 72, 75, 76, 77, 79, 81, 82, 84, 86, 87, 89, 90, 93, 95, 97, 98]
    test_filter=[53, 67, 74, 80, 88, 96, 61, 6, 64, 92, 70, 85, 50, 56, 57]

    train, valid, test = load_train_test_data(run_path, train_filter=train_filter, valid_filter=valid_filter, test_filter=test_filter)

    # Shuffle point cloud dataset
    np.random.shuffle(train)
    np.random.shuffle(valid)
    np.random.shuffle(test)

    # Get X and Y data for training
    train_x, train_y = get_data_and_label2(train)
    valid_x, valid_y = get_data_and_label2(valid)
    test_x, test_y = get_data_and_label2(test)
    
    test_name = "manual_test_regress_01"

    k=16
    batch_size=128
    steps_per_epoch=32
    validation_steps=25
    lr = 0.0001

    num_points=200
    num_dims=3

    checkpoint_path = experiments_path+test_name+"/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        save_weights_only=False,
        save_freq=10*steps_per_epoch)

    inputs = keras.Input(shape=(None, 3))

    # TNET : first list is CNN, second list is dense
    tnet_shape = [[[256, 256,], [512, 256]],
                  [[256, 128,], [128,128]]]
    conv_gnns = [[[256,256], [128]]]
    dense_gnn = [512, 256,128]

    outputs = build_model(inputs,
                        num_points, num_dims, k,
                        tnet_shape,
                        conv_gnns,
                        dense_gnn
                    )

    model = keras.Model(inputs=[inputs], outputs=outputs, name=test_name+"net")


    opt_pi = tf.optimizers.Adam(learning_rate=lr)

    # def custom_loss(y_true, y_pred):
    #     return tf.math.reduce_mean(tf.math.square(y_true*100 - y_pred*100), axis=-1)

    model.compile(loss='huber', optimizer=opt_pi, metrics=[CustomMetric()])
    # model.compile(loss=custom_loss, optimizer=opt_pi, metrics=[CustomMetric()])
    # model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=opt_pi, metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # Try to load the model. If it does not exist, create it.
    # latest = tf.train.latest_checkpoint(experiments_path+test_name+"/")
    latest = sorted([ f.path for f in os.scandir(experiments_path+test_name) if f.is_dir() ])[-1] \
        if os.path.isdir(experiments_path+test_name) else None

    if latest:
        # https://www.tensorflow.org/tutorials/keras/save_and_load
        # model.load(latest)
        model = tf.keras.models.load_model(latest, custom_objects={'CustomOrthogonalRegularizer': CustomOrthogonalRegularizer, 'custom_loss': custom_loss, 'CustomMetric': CustomMetric})
        latest_ep = int(latest.split('/')[-1].split('-')[-1].split('.')[0])
        print(" Model loaded correctly:", latest, " - Epoch ", latest_ep)
    else:
        print(" The model at ", experiments_path+test_name+"/", "could not be loaded properly: ", latest)
        model.save(checkpoint_path.format(epoch=0))
        latest_ep = 0

    # This grants no overwriting of the history file
    filename=experiments_path+test_name+"/history"+str(latest_ep)+".csv"
    history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    # Use CPU as default due to GPU's memory issues
    with tf.device(DEVICE):
        history = model.fit(
            train_x,
            train_y,
            
            initial_epoch=latest_ep,
            batch_size=batch_size, 
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,

            # validation_split=0.3,
            validation_data=(valid_x, valid_y),
            epochs=3000,
            # shuffle=True,
            callbacks=[ValAccThresh_CB(thresh=0.85, experiments_path=experiments_path, test_name=test_name), cp_callback, history_logger],
            # use_multiprocessing=True,
            # workers=8,
        )
