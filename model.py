# Import any ML library here (eg torch, keras, tensorflow)
# Start Editing
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization,Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from tensorflow.keras.applications import imagenet_utils
from sklearn.model_selection import KFold
import itertools
import shutil
import glob
import matplotlib.pyplot as plt
from keras import backend as K
import os.path
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# End Editing

import argparse
import random
import numpy as np
from dataLoader import Loader
import os
import cv2



# (Optional) If you want to define any custom module (eg a custom pytorch module), this is the place to do so
# Start Editing
# End Editing


# This is the class for training our model
class Trainer:
	def __init__(self):

		# Seed the RNG's
		# This is the point where you seed your ML library, eg torch.manual_seed(12345)
		# Start Editing
		seed_value = 12345
		os.environ['PYTHONHASHSEED']=str(seed_value)
		np.random.seed(seed_value)
		random.seed(seed_value)
		tf.set_random_seed(seed_value)
		session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
		sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
		K.set_session(sess)
		# End Editing

		# Set hyperparameters. Fiddle around with the hyperparameters as different ones can give you better results
		# (Optional) Figure out a way to do grid search on the hyperparameters to find the optimal set
		# Start Editing
		#######ADD GRID SEARCH######
		self.batch_size = 64 # Batch Size
		self.num_epochs = 20 # Number of Epochs to train for
		self.lr = 0.01       # Learning rate
		# End Editing

		# Init the model, loss, optimizer etc
		# This is the place where you define your model (the neural net architecture)
		# Experiment with different models
		# For beginners, I suggest a simple neural network with a hidden layer of size 32 (and an output layer of size 10 of course)
		# Don't forget the activation function after the hidden layer (I suggest sigmoid activation for beginners)
		# Also set an appropriate loss function. For beginners I suggest the Cross Entropy Loss
		# Also set an appropriate optimizer. For beginners go with gradient descent (SGD), but others can play around with Adam, AdaGrad and you can even try a scheduler for the learning rate
		# Start Editing
		self.model = Sequential([
			Conv2D(filters=32,kernel_size = (3,3), activation = 'relu', padding = 'same',  input_shape = (28, 28, 1)),
			MaxPool2D(pool_size=(2,2), strides=2),
			Conv2D(filters = 64, kernel_size = (3,3), activation ='relu', padding = 'same'),
			MaxPool2D(pool_size=(2,2), strides=2),
			Flatten(),
			Dense(units = 10, activation='softmax'),
			])
		self.loss = categorical_crossentropy
		self.optimizer = Adadelta
		# End Editing

	def load_data(self):
		# Load Data
		self.loader = Loader()

		# Change Data into representation favored by ML library (eg torch.Tensor for pytorch)
		# This is the place you can reshape your data (eg for CNN's you will want each data point as 28x28 tensor and not 784 vector)
		# Don't forget to normalize the data (eg. divide by 255 to bring the data into the range of 0-1)
		# Start Editing
		img_rows, img_cols=28, 28
		# if K.image_data_format() == 'channels_first':
		self.train_data = self.train_data.reshape(self.train_data.shape[0], img_rows, img_cols, 1)
		self.test_data = self.test_data.reshape(self.test_data.shape[0], img_rows, img_cols, 1)
		# inpx = (1, img_rows, img_cols)
		
		# else:
		# 	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		# 	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		# 	inpx = (img_rows, img_cols, 1)
			
		self.train_data = self.train_data.astype('float32')
		self.test_data = self.test_data.astype('float32')
		self.train_data /= 255
		self.test_data /= 255
		self.train_labels = keras.utils.to_categorical(self.train_labels)
		self.test_labels = keras.utils.to_categorical(self.test_labels)	


		# End Editing
		pass

	def save_model(self):
		# Save the model parameters into the file 'assets/model'
		# eg. For pytorch, torch.save(self.model.state_dict(), 'assets/model')
		# Start Editing
		if os.path.isdir('assets/model') is False:
			os.mkdir('assets/model')
		self.model.save('assets/model/mnistcnn.h5')

		# End Editing
		pass

	def load_model(self):
		# Load the model parameters from the file 'assets/model'
		if os.path.exists('assets/model'):
			self.model = load_model('assets/model/mnistcnn.h5')
			# eg. For pytorch, self.model.load_state_dict(torch.load('assets/model'))
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
		kf = KFold(n_splits=8)
		for train_index, val_index in kf.split(self.train_data, self.train_labels):
			self.X_train, self.X_val = self.train_data[train_index], self.train_data[val_index]
			self.y_train, self.y_val = self.train_labels[train_index], self.train_labels[val_index]	
			self.model.compile(optimizer=self.optimizer,
              loss=self.loss,
              metrics=['accuracy'])
			checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+'model_'+str(fold_var)+'.h5', 
							monitor='val_accuracy', verbose=1, 
							save_best_only=True, mode='max') 
			callbacks_list = [checkpoint]				 
			history = self.model.fit(self.X_train, self.y_train,
              batch_size=self.batch_size,
			  validation_data=(self.X_val, self.y_val),
              epochs=self.num_epochs,
              verbose=2)
			self.model.load_weights("assets/model/model_"+str(fold_var)+".h5") 
			results = self.model.evaluate(self.X_val, self.y_val)
			results = dict(zip(self.model.metrics_names,results))
			
			VALIDATION_ACCURACY.append(results['accuracy'])
			VALIDATION_LOSS.append(results['loss'])
			
			tf.keras.backend.clear_session()
			
			fold_var += 1

		print('Training Complete')


	def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `normalize=True`.
		"""
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=45)
		plt.yticks(tick_marks, classes)

		if normalize:
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print("Normalized confusion matrix")
		else:
			print('Confusion matrix, without normalization')

		print(cm)

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j],
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black")

		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')


	def test(self):
		if not self.model:
			return 0

		print(f'Running test...')
		# Initialize running loss

		# Start Editing
		for layer in self.model.layers:
			layer.trainable = False
		# Set the ML library to freeze the parameter training

		i = 0 # Number of batches
		correct = 0 # Number of correct predictions
		for batch in range(0, self.test_data.shape[0], self.batch_size):
			batch_X = self.test_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
			batch_Y = self.test_labels[batch: batch+self.batch_size] # shape [batch_size,]

			# Find the predictions
			# Find the loss
			# Find the number of correct predictions and update correct
			predictions = self.model.predict(x=batch_X, )
			cm = confusion_matrix(self.test_labels, predictions)
			plot_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
			plot_confusion_matrix(cm = cm, classes = plot_labels)


			# Update running_loss

			i += 1
		
		score, acc = self.model.evaluate(x = self.test_data, y = self.test_labels, batch_size = self.batch_size)
		print('Test score:', score)
		print('Test accuracy:', acc)
		return acc
	

	# def run_epoch(self):
	# 	# Initialize running loss
	# 	running_loss = 0.0

	# 	# Start Editing

	# 	# Set the ML library to enable the parameter training
		
	# 	# Shuffle the data (make sure to shuffle the train data in the same permutation as the train labels)
		
	# 	i = 0 # Number of batches
	# 	for batch in range(0, self.train_data.shape[0], self.batch_size):
	# 		batch_X = self.train_data[batch: batch+self.batch_size] # shape [batch_size,784] or [batch_size,28,28]
	# 		batch_Y = self.train_labels[batch: batch+self.batch_size] # shape [batch_size,]


	# 		# Zero out the grads for the optimizer
			
	# 		# Find the predictions
	# 		# Find the loss
	# 		# Backpropagation

	# 		# Update the running loss
	# 		i += 1
		
	# 	# End Editing

	# 	return running_loss / i


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