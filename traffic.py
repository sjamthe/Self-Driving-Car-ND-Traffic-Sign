import pickle
import numpy as np
#
# TODO: fill this in based on where you saved the training and testing data
training_file = 'data/train.p'
testing_file = 'data/test.p'
#
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
  test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
train_sizes, train_coords = train['sizes'], train['coords']
test_sizes, test_coords = test['sizes'], test['coords']
print (X_train.shape,y_train.shape,X_test.shape, y_test.shape)

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

trainY = to_categorical(y_train,43)
X_prep = np.array(X_train, dtype=float)

with g.as_default():
  model.fit(X_train, trainY, n_epoch=30, shuffle=True, validation_set=0.10, 
                show_metric=True, batch_size=64,
                snapshot_epoch=True,run_id='traffic-sign-V6')

  model.save("traffic-sign.tfl")
print ("model trained")

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
