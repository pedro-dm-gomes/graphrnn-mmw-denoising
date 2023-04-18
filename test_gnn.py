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
import sklearn

import optuna
import numpy as np
from scipy import stats
import json
import os
import time
import utils

# Where datasets are stored
run_path = "data/"
experiments_path = "experiments/"
test_name = "manual_test_12_relu"


valid_filter = [9, 69, 73, 3, 55]
train_filter=[4, 5, 7, 8, 49, 51, 54, 58, 59, 62, 63, 65, 66, 68, 71, 72, 75, 76, 77, 79, 81, 82, 84, 86, 87, 89, 90, 93, 95, 97, 98]
test_filter=[53, 67, 74, 80, 88, 96, 61, 6, 64, 92, 70, 85, 50, 56, 57]

# dataset = load_datasets(run_path)
_, _, test = utils.load_train_test_data(run_path, train_filter=train_filter, valid_filter=valid_filter, test_filter=test_filter)


# Shuffle point cloud dataset
# np.random.shuffle(train)
# np.random.shuffle(valid)
np.random.shuffle(test)

# Get X and Y data for training
# train_x, train_y = get_data_and_label2(train)
# valid_x, valid_y = get_data_and_label2(valid)
test_x, test_y = utils.get_data_and_label2(test)


k=16
batch_size=32
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

tnet_shape = [[[64], [64], [128]]]
conv_gnns = [[[128], [64]]]
dense_gnn = [64, 64]

outputs = utils.build_model(inputs,
                    num_points, num_dims, k,
                    tnet_shape,
                    conv_gnns,
                    dense_gnn
                )

model = keras.Model(inputs=[inputs], outputs=outputs, name=test_name+"net")


opt_pi = tf.optimizers.Adam(learning_rate=lr)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer=opt_pi, metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Try to load the model. If it does not exist, create it.
# latest = tf.train.latest_checkpoint(experiments_path+test_name+"/")
latest = sorted([ f.path for f in os.scandir(experiments_path+test_name) if f.is_dir() ])[-1] \
    if os.path.isdir(experiments_path+test_name) else None

if latest:
    # https://www.tensorflow.org/tutorials/keras/save_and_load
    # model.load(latest)
    model = tf.keras.models.load_model(latest, custom_objects={'CustomOrthogonalRegularizer': utils.CustomOrthogonalRegularizer})
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
with tf.device('/CPU:0'):
    # test the model on the test set
    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)
    # compute precision, recall, f1-score 
    y_pred = np.argmax(model.predict(test_x), axis=1)
    y_true = test_y
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')

# pretty print loss, accuracy, precision, recall, f1-score, each on a new line
print(" Test results for model: ", test_name)
print(" Test loss: {:.4f}".format(test_loss))
print(" Test accuracy: {:.4f}".format(test_acc))
print(" Test precision: {:.4f}".format(precision))
print(" Test recall: {:.4f}".format(recall))
print(" Test f1-score: {:.4f}".format(f1))

