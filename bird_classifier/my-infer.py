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
import argparse
import scipy

# Real-time data preprocessing
# Make sure the data is normalized

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(mean=0.472978413436)
img_prep.add_featurewise_stdnorm(std=0.248771212446)

#network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep)
g = tf.Graph()
with g.as_default():
  network = input_data(shape=[None, 32, 32, 3])
  network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
  network = max_pool_2d(network, 2)
  network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
  network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
  network = max_pool_2d(network, 2)
  network = fully_connected(network, 512, activation='relu')
  network = dropout(network, 0.5)

  network = fully_connected(network, 2, activation='softmax')

# Tell tflearn how we want to train the network
  network = regression(network, optimizer='adam', learning_rate=0.0001,
                                   loss='categorical_crossentropy', name='target')
# Wrap the network in a model object
  model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='my-bird.ckpt')

#print ("Network ready")

parser = argparse.ArgumentParser(description='Decide if an image is a picture of a bird')
parser.add_argument('image', type=str, help='The image image file to check')
args = parser.parse_args()

with g.as_default():
  model.load("my-bird.tfl")
  #model.load("my-bird.ckpt-26640")

# Load the image file
img = scipy.ndimage.imread(args.image, mode="RGB")

# Scale it to 32x32
img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

# Predict
with g.as_default():
  prediction = model.predict([img])

# Check the result.
is_bird = np.argmax(prediction[0]) == 1

if is_bird:
    print("That's a bird!")
else:
    print("That's not a bird!")
