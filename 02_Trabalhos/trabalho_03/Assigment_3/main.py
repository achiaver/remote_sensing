from utils import *
from keras.datasets import cifar10
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
#from keras.callbacks import ModelCheckpoint , EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as k
import keras

import numpy as np
from sklearn.utils import shuffle


def extract_traning_patches():
	# Build function to extract pathches from the training image.
	return 0


def FCN(rows, cols, channels):
	# Define here the FCN model
  	input_img = Input(shape=(rows , cols , channels))
  	#...
  	# output = 
  	return Model(input_img , output)


def Train(net, train_pathes_dir, batch_size,  epochs):
	
	print('Start the training')
	for epoch in range(epochs):
		loss_tr = np.zeros((1 , 2))
		loss_ts = np.zeros((1 , 2))
		# Loading the data set
		trn_data_dirs = glob.glob(train_pathes_dir + '/*.npy')
		# Random shuffle the data
		np.shuffle(trn_data_dirs)
		# Computing the number of batchs
		batch_idxs = len(trn_data_dirs) // batch_size

        for idx in xrange(0, batch_idxs):
        	batch_files = trn_data_dirs[idx*batch_size:(idx+1)*batch_size]
        	batch = [load_data(batch_file) for batch_file in batch_files]
        	batch_images = np.array(batch).astype(np.float32)
        	x_train_b = batch[:, :, :, 0:3]
        	y_train = batch[:, :, :, 3]

			# Performing hot encoding    		

    		loss_tr = loss_tr + net.train_on_batch(x_train_b , y_train_hot_encoding)
    	loss_tr = loss_tr/n_batchs_tr

	    # # Evaluating the network in the test set

	    # for  batch in range(n_batchs_ts):
	    # 	x_test_b = x_test[batch * batch_size : (batch + 1) * batch_size , : , : , :]
	    # 	y_test_h_b = y_test_h[batch * batch_size : (batch + 1) * batch_size , : ]
		   #  loss_ts = loss_ts + net.test_on_batch(x_test_b , y_test_h_b)
    	# loss_ts = loss_ts/n_batchs_ts

    	# print("%d [training loss: %f , Train acc.: %.2f%%][Test loss: %f , Test acc.:%.2f%%]" %(epoch , loss_tr[0 , 0], 100*loss_tr[0 , 1] , loss_ts[0 , 0] , 100 * loss_ts[0 , 1]))

if __name__=='__main__':
	# 
	extract_traning_patches() # run this function once.
	adam = Adam(lr = 0.0001 , beta_1=0.9)
	net = FCN(rows, cols, channels)
	net.summary()
	net.compile(loss = 'binary_crossentropy',optimizer=adam , metrics=['accuracy'])
	#Call the train function
	Train(net , train_pathes_dir='dir', batch_size=100, epochs=50)