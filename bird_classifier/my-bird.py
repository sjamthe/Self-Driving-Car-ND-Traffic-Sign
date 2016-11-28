import pickle
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

# Load the data set
X, Y, X_test, Y_test = pickle.load(open("full_dataset.pkl","rb"),encoding='latin1')
#
# Shuffle the data
X, Y = shuffle(X, Y)

g = tf.Graph()
with g.as_default():
# Real-time data preprocessing
# Make sure the data is normalized
  img_prep = ImagePreprocessing()
  img_prep.add_featurewise_zero_center(mean=0.472978413436)
  img_prep.add_featurewise_stdnorm(std=0.248771212446)

# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
  img_aug = ImageAugmentation()
  img_aug.add_random_flip_leftright()

  #network = input_data(shape=[None, 32, 32, 3])
  network = input_data(shape=[None, 32, 32, 3],
                       data_preprocessing=img_prep,
                       data_augmentation=img_aug)
  network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
  network = max_pool_2d(network, 2)
  network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
  network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
  network = max_pool_2d(network, 2)
  network = fully_connected(network, 512, activation='relu')
  network = dropout(network, 0.5)
#
  network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
  network = regression(network, optimizer='adam', learning_rate=0.0001,
                                          loss='categorical_crossentropy', name='target')
# Wrap the network in a model object
  model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='my-birdV2.ckpt')

print ("Network ready")

# Train it! We'll do 100 training passes and monitor it as it goes.
with g.as_default():
  model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=(X_test, Y_test), 
                show_metric=True, batch_size=64,
                snapshot_epoch=True,run_id='my-bird-V2')

  model.save("my-birdV2.tfl")
print ("model trained")
