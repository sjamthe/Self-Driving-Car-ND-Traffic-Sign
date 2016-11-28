import pickle
import numpy as np

# TODO: fill this in based on where you saved the training and testing data
training_file = 'data/train.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
train_sizes, train_coords = train['sizes'], train['coords']
test_sizes, test_coords = test['sizes'], test['coords']
print (X_train.shape,y_train.shape,X_test.shape, y_test.shape)

trainY = np.array([[0 for j in range(43)] for i in range(len(y_train))])
print (trainY.shape, y_train[10],trainY[10][y_train[10]])
i=0
for val in y_train:
	#print (i,val)
	trainY[i][val] = 1
	i+=1

#print (trainY.shape, y_train[10],trainY[10])

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 100, 7, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 150, 4, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 250, 4, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = dropout(network, 0.8)
network = fully_connected(network, 300, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 43, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

print ("Network ready")

model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X_train, trainY, n_epoch=10,validation_set=0.05, batch_size=128,
                  snapshot_step=100,show_metric=True,run_id='convnet_trafficsign')
