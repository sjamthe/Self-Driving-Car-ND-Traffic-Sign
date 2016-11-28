import sys
import pickle
import numpy as np
#
testing_file = 'data/test.p'
#
with open(testing_file, mode='rb') as f:
  test = pickle.load(f)

X_test, y_test = test['features'], test['labels']
test_sizes, test_coords = test['sizes'], test['coords']
print (X_test.shape, y_test.shape)

import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

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
  network = fully_connected(network, 43, activation='softmax')
  network = regression(network, optimizer='adam', learning_rate=0.0001,
                                          loss='categorical_crossentropy', name='target')
  model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='traffic-sign.ckpt')

print ("Network ready")

with g.as_default():
  #model.load("traffic-sign.tfl")
  modelName = sys.argv[1]
  model.load(modelName)
print ("model "  +  modelName  +  " loaded")

#print(tflearn.variables.get_all_variables ())

testY = to_categorical(y_test,43)
with g.as_default():
  val = model.evaluate(X_test,testY)
  print("Test accuracy is = ", val)
  results=model.predict(X_test)

#compare predicted results and print images that are not correct

y_out = np.array(results).argmax(1) #argmax converts to_categorical
incorrects = []
y_errors = []
for i,test in enumerate(y_test):
      if(y_out[i] != test):
                #print (i, test, y_out[i])
        incorrects.append([i, test, y_out[i]])
        y_errors.append(test)

print("mismatches = ", len(incorrects))

# Display how many errors for each class of image
print (np.bincount(np.array(y_errors)))
