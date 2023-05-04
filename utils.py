'''
    Requirements:
     * TF:      2.7.0
     * Keras:   2.7.0
'''

import tensorflow as tf
import keras
from keras import layers

'''
    Setting Up matplotlib for paper compliant figures
    (this should avoid problems when compiling latex stuff)
'''
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers

import optuna
import numpy as np
from scipy import stats
import json
import os
import time

# Dataset Related functions

def load_datasets(init_path):
    '''
        Return dataset as list of pointclouds
        ---
        Parameters:
        * init_path: string; path to folder holding folders of datasets
                     (expected folder structure: 'init_path/run_X/labelled_mmw_run_X.json')
    '''
    data = []
    for run in sorted(os.listdir(init_path)):
        if "run_" in run:
            data.extend(json.load(open(init_path+run+"/labelled_mmw_"+run+".json"))['labelled_mmw'])
    return data

def load_train_test_data(init_path, train_filter=[], valid_filter=[], test_filter=[]):
    '''
        Return dataset as list of pointclouds
        ---
        Parameters:
        * init_path: string; path to folder holding folders of datasets
                     (expected folder structure: 'init_path/run_X/labelled_mmw_run_X.json')
    '''
    train = []
    valid = []
    test = []
    for run in sorted(os.listdir(init_path)):
        if "run_" in run:
            if (int(run.split('_')[-1]) in train_filter):
                # train.extend(json.load(open(init_path+run+"/labelled_mmw_"+run+".json"))['labelled_mmw'])
                train.extend(json.load(open(init_path+run+"/norm_mmw_"+run+".json"))['labelled_mmw'])
            elif (int(run.split('_')[-1]) in valid_filter):
                valid.extend(json.load(open(init_path+run+"/labelled_mmw_"+run+".json"))['labelled_mmw'])
            elif (int(run.split('_')[-1]) in test_filter):
                test.extend(json.load(open(init_path+run+"/labelled_mmw_"+run+".json"))['labelled_mmw'])
    return train, valid, test

def get_data_and_label(data, points_per_cloud=200):
    '''
        Return samples for training as np.array, divided as unlabelled data and related labels.
        ---
        Parameters:
        * data: list of point clouds. (Usually loaded with function load_datasets)
        * points_per_cloud: number of points to be found in each point cloud. (Default is 200)
    '''
    d_x, d_y = [], []
    skipped = 0
    for pc in data:
        for i in range(0, len(pc), points_per_cloud): 
            if len(pc[i:i+points_per_cloud]) == points_per_cloud:
                t_ = np.array(pc[i:i+points_per_cloud])[:, :3]
                d_x.append(t_)
                d_y.append(np.array(pc[i:i+points_per_cloud], dtype=np.float32)[:, -1])
                # d_y.append(tf.one_hot(np.array(pc[i:i+points_per_cloud], dtype=np.float32)[:, -1], 2)) # One Hotted
            else:
                skipped += 1
    d_x, d_y = np.stack(d_x), np.stack(d_y)
    print("Skipped {} point clouds".format(skipped))
    return d_x, d_y

def get_data_and_label2(data, points_per_cloud=200):
    '''
        Return samples for training as np.array, divided as unlabelled data and related labels.
        ---
        Parameters:
        * data: list of point clouds. (Usually loaded with function load_datasets)
        * points_per_cloud: number of points to be found in each point cloud. (Default is 200)
    '''
    d_x, d_y = [], []
    skipped = 0
    data = np.vstack(data)
    print(" Data shape: ", data.shape)
    # for pc in data:
    for i in range(0, len(data), points_per_cloud): 
        if len(data[i:i+points_per_cloud]) == points_per_cloud:
            t_ = np.array(data[i:i+points_per_cloud])[:, :3]
            d_x.append(t_)
            d_y.append(np.array(data[i:i+points_per_cloud], dtype=np.float32)[:, -1])
            # d_y.append(tf.one_hot(np.array(pc[i:i+points_per_cloud], dtype=np.float32)[:, -1], 2)) # One Hotted
        else:
            print("Points per cloud: ", points_per_cloud)
            print("Skipping because: ", data[i:i+points_per_cloud].shape, i, i+points_per_cloud)
            skipped += 1
        #     break
        # break
    d_x, d_y = np.stack(d_x), np.stack(d_y)
    print("Skipped {} point clouds".format(skipped))
    return d_x, d_y


# Model Related functions

def gnn_conv2d(inputs,
            filters,
            kernel_size,
            stride=[1],
            padding='SAME',
            use_xavier=True,
            stddev=1e-3,
            activation_fn=tf.nn.relu,
            bn=False):

    x = layers.Conv1D(
        filters, 
        kernel_size, 
        strides=stride, 
        padding=padding,
        activation=activation_fn,
        kernel_initializer='glorot_uniform' if use_xavier else keras.initializers.TruncatedNormal(stddev=stddev),
        bias_initializer='zeros'
    )(inputs)

    if bn: x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    return x

def gnn_dense(inputs,
            units,
            use_xavier=True,
            stddev=1e-3,
            activation_fn=tf.nn.relu,
            bn=False):
            
    x = layers.Dense(units,
        activation=activation_fn,
        kernel_initializer='glorot_uniform' if use_xavier else keras.initializers.TruncatedNormal(stddev=stddev),
        bias_initializer='zeros'
    )(inputs)

    if bn: x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    return x

def lambda_get_adj_matr(input):
    pcT = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input)
    pc_inn = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))( (input, pcT) )
    pc2 = layers.Lambda(lambda x: tf.reduce_sum(tf.square(x), axis=-1, keepdims=True))(input)
    pc2T = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(pc2)
    output = layers.Lambda(lambda x: x[0] + -2 * x[1] + x[2])( (pc2, pc_inn, pc2T) )
    # Uncomment line below to use reciprocal of adj matrix (1/distance)
    # output = layers.Lambda(lambda x: tf.math.reciprocal(x))(output)
    return output

def lambda_knn(adj, k=20):
    x = layers.Lambda(lambda x: tf.math.top_k(-x[0], x[1]))( (adj, k) )
    return x.indices

def lambda_edge_feature(inputs, nn_idxs, k=20, num_points=200, num_dims=3):

    pc_central = inputs
    batch_size = tf.shape(inputs)[0]

    idx_ = layers.Lambda(lambda x: tf.range(x[0]) * x[1])( (batch_size, num_points) )
    idx_ = layers.Lambda(lambda x: tf.reshape(x[0], (x[1], 1, 1)))( (idx_, batch_size) )
    # Adding to list of idxs of k points the points themselves
    pc_temp1 = layers.Lambda(lambda x: x[0]+x[1])( (nn_idxs, idx_) )

    # Flattening of points into a list of coordinates (x,y,z)
    pc_flat = layers.Lambda(lambda x: tf.reshape(x[0], [-1, x[1]]))( (inputs, num_dims) )

    # Collect points from computed idxs
    pc_neighbors = layers.Lambda(lambda x: tf.gather(x[0], x[1]) )( (pc_flat, pc_temp1) )

    # Reshape points into shape (batch, num_points, NEW_AXIS = 1, num_dims)
    pc_central = layers.Lambda(lambda x: tf.expand_dims(x, axis=-2))(pc_central)
    # Points are repeated k-times along new dimension ==> (batch, num_points, k, num_dims)
    pc_central = layers.Lambda(lambda x: tf.tile(x[0], [1, 1, x[1], 1]))( (pc_central, k) )

    pc_temp2 = layers.Lambda(lambda x: tf.subtract(x[0], x[1]))( (pc_neighbors, pc_central) )
    edge_feature = layers.Lambda(lambda x: tf.concat((x[0], x[1]), axis=-1))((pc_central, pc_temp2))
    return edge_feature

class CustomOrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features=3, l2reg=0.0005):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
    def get_config(self):
        return {'num_features': self.num_features, 'l2reg': self.l2reg}

def gnn_tnet(inputs, num_dims, tnet_shapes, bn=False):
    batch_size = tf.shape(inputs)[0]
    for filt in tnet_shapes[0]:
        x = gnn_conv2d(inputs, filters=filt, kernel_size=[1], bn=bn)
    # x = tf.reduce_max(x, axis=-2, keepdims=True)
    # for filt in tnet_shapes[1]:
    #     x = gnn_conv2d(inputs, filters=filt, kernel_size=[1], bn=bn)
    x = layers.GlobalMaxPooling1D(keepdims=True)(x)
    # print(" [tnet] global maxpool 2d shape: ", x.shape)
    x = layers.Lambda(lambda y: tf.reshape(y[0], (y[1], y[2])))( [x, batch_size, x.shape[-1]] )
    # print(" [tnet] reshape: ", x.shape)

    for neur in tnet_shapes[2]:
        x = gnn_dense(x, neur, bn)
    
    bias = keras.initializers.Constant(np.eye(num_dims).flatten())
    reg = CustomOrthogonalRegularizer(num_features=num_dims)
    x = layers.Dense(
        num_dims * num_dims,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_dims, num_dims), name='tnet_last_reshape'+str(np.round(time.time(), 5)))(x)
    return feat_T

# function to test custom losses
def custom_loss(pred, labels):
    # loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    # classify_loss = tf.reduce_mean(loss)
    loss = tf.reduce_mean(tf.reduce_sum(tf.math.square(tf.math.subtract(labels, pred)), axis=-1))
    return loss

'''
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
tf.math.l2_normalize
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
'''

# Callback to save good models. Threshold is on validation accuracy.
class ValAccThresh_CB(keras.callbacks.Callback):
    def __init__(self, thresh=0.85, experiments_path="experiments/", test_name="test"):
        self.thresh = thresh
        super(keras.callbacks.Callback, self).__init__()
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.stopped_epoch = 0
        self.experiments_path = experiments_path
        self.test_name = test_name

    def on_epoch_end(self, epoch, logs=None):
        # self.current_epoch += 1
        val_key = ""
        for k in logs.keys():
            if "val" in k and "accuracy" in k:
                val_key = k
                break
        assert logs.get(val_key) != None, print(" Validation Accuracy not found", self.thresh, logs.get(val_key), val_key, logs.get(val_key) == None)

        current = logs.get(val_key)
        # current = logs.get("val_sparse_categorical_accuracy")
        # current = logs.get("val_accuracy")
        if current >= self.thresh:
            self.thresh = current
            self.model.save_weights(self.experiments_path+self.test_name+"/best_weights/cp-"+str(epoch)+".ckpt")
            print(" New good model saved.")

def build_model(inputs, 
        num_points, num_dims, k,
        tnet_shape,
        conv_gnns,
        dense_gnn,
        classes=2):
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
    x = gnn_dense(inputs, 32, bn=False)
    feat_T = gnn_tnet(x, 32, tnet_shape[0], bn=True)
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
            x = gnn_conv2d(x, gc_filt, [1], bn=True)
        
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
            x = gnn_conv2d(x, gc_filt, [1], bn=True)
        # x = tf.reduce_max(x, axis=-2, keepdims=True)
        # x = layers.Lambda(lambda y: tf.reshape(y[0], (y[1], num_points, l[1][-1])))( [x, tf.shape(inputs)[0]] )

    # x = layers.GlobalMaxPooling1D(keepdims=True)(x)
    # x = layers.Lambda(lambda y: tf.reshape(y[0], (y[1], y[2])))( [x, tf.shape(x)[0], x.shape[-1]] )
    
    for w_ in dense_gnn:
        x = gnn_dense(x, w_, bn=False)

    outputs = layers.Dense(classes, activation="softmax")(x)
    return outputs

# Optuna Related functions

def objective(trial):
    test_name = "fixed_test_"+str(trial.number)
    filename=experiments_path+test_name+"/history.csv"
    history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)
    ############################ HyperParameter Setup ############################
    ######################### Check build_model for docs #########################
    k = trial.suggest_int('k', 5,75) #30
    batch_size = trial.suggest_int('batch_size', 8,128) #16
    # tnet_before_max = trial.suggest_int('tnet_before_max', 1,3)
    # tnet_before = []
    # for i in range(tnet_before_max):
    #     tnet_before.append(trial.suggest_int('tnet_beforemax_layer_'+str(i), 8,128))
    # tnet_after = []
    # tnet_after_max = trial.suggest_int('tnet_after_max', 1,3)
    # for i in range(tnet_after_max):
    #     tnet_after.append(trial.suggest_int('tnet_aftermax_layer_'+str(i), 8,128))
    # tnet_dense = []
    # tnet_dense_layers = trial.suggest_int('tnet_dense_layers', 1,3)
    # for i in range(tnet_dense_layers):
    #     tnet_dense.append(trial.suggest_int('tnet_dense_layer_'+str(i), 16,256))
    tnet_before = [32]
    tnet_after = [128]
    tnet_dense = [128, 128]
    tnet_shape = [tnet_before, tnet_after, tnet_dense]

    # gc_layers = trial.suggest_int('gc_layers', 1,3) #1
    conv_gnns = []
    # for _ in range(gc_layers):
    #     before_edge_gcl = trial.suggest_int('before_edge_gcl', 1,3) #2
    #     after_edge_gcl = trial.suggest_int('after_edge_gcl', 1,3) #2

    #     bfr_edge = []
    #     for i in range(before_edge_gcl):
    #         bfr_edge.append(trial.suggest_int('before_edge_gcl_'+str(i), 8,128))
            
    #     aft_edge = []
    #     for i in range(after_edge_gcl):
    #         aft_edge.append(trial.suggest_int('after_edge_gcl_'+str(i), 8,128))
            
    #     conv_gnns.append([bfr_edge, aft_edge])
    bfr_edge = [32, 64]
    aft_edge = [128]
    conv_gnns.append([bfr_edge, aft_edge])
    # dense_layers = trial.suggest_int('dense_layers', 1,3)
    # dense_gnn = []
    # for i in range(dense_layers):
    #     dense_gnn.append(trial.suggest_int('dense_layer_'+str(i), 16,256))
    dense_gnn = [256, 128]
        
    lr = trial.suggest_float('lr', 0.0001, 0.1)
    steps_per_epoch=trial.suggest_int('steps_per_epoch', 15,100)

    validation_steps=25     # Static
    num_points = 200        # Static
    num_dims = 3            # Static
    ##############################################################################

    inputs = keras.Input(shape=(None, num_dims))
    
    outputs = build_model(inputs,
                    num_points, num_dims, k,
                    tnet_shape,
                    conv_gnns,
                    dense_gnn
                )
    model = keras.Model(inputs=[inputs], outputs=outputs, name="gnn_pointnet")

    opt_pi = tf.optimizers.Adam(learning_rate =  lr )
    # opt_pi = tf.optimizers.RMSprop(learning_rate =  lr )
    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=opt_pi, metrics=[keras.metrics.SparseCategoricalAccuracy()])
    # model.compile(loss=tf.nn.sparse_softmax_cross_entropy_with_logits , optimizer=opt_pi, metrics=['accuracy'])
    # model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=opt_pi, metrics=['accuracy'])
    # model.compile(loss=custom_loss, optimizer=opt_pi, metrics=['accuracy'])

    checkpoint_path = experiments_path+test_name+"/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=0, 
        save_weights_only=False,
        save_freq=10*batch_size)
        
    latest = tf.train.latest_checkpoint(experiments_path+test_name+"/")
    if latest:
        model.load_weights(latest)
        latest_ep = int(latest.split('/')[-1].split('-')[-1].split('.')[0])
        print(" Model loaded correctly:", latest, " - Epoch ", latest_ep)
    else:
        print(" The model could not be loaded properly: ", latest)
        model.save(checkpoint_path.format(epoch=0))
        latest_ep = 0

    # Use CPU as default due to GPU's memory issues
    with tf.device('/CPU:0'):
        history = model.fit(
            train_x, 
            train_y, 
            
            initial_epoch=latest_ep,
            batch_size=batch_size, 
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,

            validation_split=0.3,
            epochs=150 - latest_ep, # Train for 150 epochs to find the configuration that can later be trained for more epochs.
            shuffle=True,
            callbacks=[ValAccThresh_CB(thresh=0.9), cp_callback, history_logger],
            use_multiprocessing=False,
            workers=8,
        )
    return np.mean(history.history['val_sparse_categorical_accuracy'][-10:])