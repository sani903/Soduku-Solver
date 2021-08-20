# Import ML library 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization,Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.model_selection import KFold
from keras import backend as K
import os.path

import argparse
import random
import numpy as np
from dataLoader import Loader
import os
import cv2



# This is the class for training our model
class Trainer:
	def __init__(self):

		# This is the point where we seed the ML library
		seed_value = 12345
		os.environ['PYTHONHASHSEED']=str(seed_value)
		np.random.seed(seed_value)
		random.seed(seed_value)
		tf.random.set_seed(seed_value)
		session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
		sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
		K.set_session(sess)

		# Set hyperparameters
		self.batch_size = 64  # Batch Size 
		self.num_epochs = 8 # Number of Epochs to train for
		self.lr = 0.003       # Learning rate 

		# Init the model, loss, optimizer etc
		# Define the model (the neural net architecture)
		self.model = Sequential([
			Conv2D(filters=32,kernel_size = (3,3), activation = 'relu', padding = 'same',  input_shape = (28, 28, 1)),
			MaxPool2D(pool_size=(2,2), strides=2),
			Conv2D(filters = 64, kernel_size = (3,3), activation ='relu', padding = 'same'),
			MaxPool2D(pool_size=(2,2), strides=2),
			Flatten(),
			Dense(units = 10, activation='softmax'),
			])
		self.loss = categorical_crossentropy
		self.optimizer = Adam

	def load_data(self):
		# Load Data
		self.loader = Loader()

		# Change Data into representation favored by ML library 
		# This is the place we can reshape the data (eg for CNN's we need each data point as 28x28 tensor and not 784 vector)
		# Normalize the data (divide by 255 to bring the data into the range of 0-1)
		img_rows, img_cols=28, 28
		self.loader.train_data = self.loader.train_data.reshape(self.loader.train_data.shape[0], img_rows, img_cols, 1)
		self.loader.test_data = self.loader.test_data.reshape(self.loader.test_data.shape[0], img_rows, img_cols, 1)
					
		self.loader.train_data = self.loader.train_data.astype('float32')
		self.loader.test_data = self.loader.test_data.astype('float32')
		self.loader.train_data /= 255
		self.loader.test_data /= 255
		self.loader.train_labels = keras.utils.to_categorical(self.loader.train_labels)
		self.loader.test_labels = keras.utils.to_categorical(self.loader.test_labels)	

		pass

	def save_model(self):
		# Save the model parameters into the file 'assets/model'
		if os.path.isdir('assets/model') is False:
			os.mkdir('assets/model')
		self.model.save('assets/model/mnistcnn.h5')

		pass

	def load_model(self):
		# Load the model parameters from the file 'assets/model'
		if os.path.exists('assets/model'):
			self.model = load_model('assets/model/mnistcnn.h5')
		else:
			raise Exception('Model not trained')

		

	def train(self):
		if not self.model:
			return

		print("Training...")
		VALIDATION_ACCURACY = []
		VALIDATION_LOSS = []

		save_dir = 'assets/model'
		fold_var = 1
		for layer in self.model.layers:
			layer.trainable = True	  
		
		kf = KFold(n_splits=3) #change to 6
		for train_index, val_index in kf.split(self.loader.train_data, self.loader.train_labels):
			self.X_train, self.X_val = self.loader.train_data[train_index], self.loader.train_data[val_index]
			self.y_train, self.y_val = self.loader.train_labels[train_index], self.loader.train_labels[val_index]	
			self.model.compile(optimizer=self.optimizer(self.lr),
              loss=self.loss,
              metrics=['accuracy'])	
			self.model.fit(self.X_train,self.y_train,
              batch_size=self.batch_size,
			  epochs=self.num_epochs,
			  verbose = 2,
			  validation_data=(self.X_val, self.y_val))
			results = self.model.evaluate(self.X_val, self.y_val)
			results = dict(zip(self.model.metrics_names,results))
			
			VALIDATION_ACCURACY.append(results['accuracy'])
			VALIDATION_LOSS.append(results['loss'])
			
			tf.keras.backend.clear_session()
			
			fold_var += 1
		self.save_model()
		print('Training Complete')


	
	def test(self):
		if not self.model:
			return 0

		print(f'Running test...')
		# Set the ML library to freeze the parameter training
		for layer in self.model.layers:
			layer.trainable = False
		
		score, acc = self.model.evaluate(x = self.loader.test_data, y = self.loader.test_labels, batch_size = self.batch_size)
		print('Test score:', score)
		print('Test accuracy:', acc)
		return acc

	

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Trainer')
	parser.add_argument('-train', action='store_true', help='Train the model')
	parser.add_argument('-test', action='store_true', help='Test the trained model')
	parser.add_argument('-preview', action='store_true', help='Show a preview of the loaded test images and their corresponding labels')
	parser.add_argument('-predict', action='store_true', help='Make a prediction on a randomly selected test image')

	options = parser.parse_args()

	t = Trainer()
	if options.train:
		t.load_data()
		t.train()
		t.test()
	if options.test:
		t.load_data()
		t.load_model()
		t.test()
	if options.preview:
		t.load_data()
		t.loader.preview()
	if options.predict:
		t.load_data()
		try:
			t.load_model()
		except:
			pass
		i = np.random.randint(0,t.loader.test_data.shape[0])

		print(f'Predicted: {t.predict(t.loader.test_data[i])}')
		print(f'Actual: {t.loader.test_labels[i]}')

		image = t.loader.test_data[i].reshape((28,28))
		image = cv2.resize(image, (0,0), fx=16, fy=16)
		cv2.imshow('Digit', image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()