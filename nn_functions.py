#!/usr/bin/env
import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.metrics import recall_score, precision_recall_fscore_support
from .metrics import avg_recall

'''
 Neural Network definition and initialization:
 - In this function we define the neural network that we will use to classify our data
 - We will use the library Keras. A full guide for this library is provided in https://keras.io/
'''

def initialize_network(input_shape, learning_rate=0.01, decay=0.01):

	# Input shape: (batch_size, number_features)
	# Learning rate and weight decay: Network parameters

	# Initialize the Neural Network
	nn = Sequential()

	'''
	Dense Layer:
	 - The number of neurons in the layer has to be defined by the user.
	 - Increasing the number of neurons in a layer increases the capability of the network to model the desired function.
	 - However, a very high number of neurons may lead the network to overfit, especially in situations where the training set is small.
	 '''
	nn.add(Dense(512, input_shape=(input_shape[1],)))

	'''
	Batch Normalization Layer:
	 - Provides regularization to the network by adjusting its inputs to a Normal distribution
	 - This layer helps reduce the amount of epochs required to train the model, and it will allow it to generalize better.
	'''
	nn.add(BatchNormalization())

	'''
	Activation functions:
	 - Add nonlinearity to the network
	 - Some examples are: 'relu', 'sigmoid', 'tanh', 'softmax'
	 - Other activation functions implemented in Keras can be found in: https://keras.io/activations/
	'''
	nn.add(Activation('relu'))

	'''
	 Dropout Layer:
	 - Another regularization layer
	 - Helps the network "generalize" better to unseen data by randomly dropping weights during training
	 - The probability of a weight being dropped is defined by the user
	'''
	nn.add(Dropout(0.5))

	'''
	 Hidden Layers:
	 - A neural network can have an unlimited number of hidden layers
	 - Additional hidden layers allow the network to learn more complex functions
	 - Too many hidden layers can make the network overfit to the training data, and it will not be able to generalize correctly
	   to unseen data.
	 '''
    # arch A hidden layers
	nn.add(Dense(128))
	nn.add(BatchNormalization())
	nn.add(Activation('relu'))
	nn.add(Dropout(0.5))

    # arch B = A + this hidden layers
	#nn.add(Dense(64))
	#nn.add(BatchNormalization())
	#nn.add(Activation('relu'))
	#nn.add(Dropout(0.5))

	'''
	 Output Layer:
	 - We are dealing with a binary classification problem, as such we only need one output neuron
	 - For non-binary classification problems the number of neurons of the ouput layer would need to be equal to the number of classes
	'''
	nn.add(Dense(1))

	'''
	Sigmoid output activation layer:
	 - For classification problems usually a sigmoid/softmax is inserted after the last fully connected/dense layer
	 - This is done to ensure the ouput is a value between 0 and 1
	 - In regression problems this layer is dropped
	'''
	nn.add(Activation('sigmoid'))

	'''
	Optimizer:
	  - Here we select the optimizer that will be used to perform backpropagation.
	  - In Keras we can select from several optimizers, listed in https://keras.io/optimizers/
	'''

	optimizer = optimizers.RMSprop(learning_rate, rho=0.9, epsilon=None, decay=decay)

	'''
	Learning rate:
	 - The most important parameter to be defined for the optimizer is the learning rate.
	 - A higher learning rate can speed up the training time and help the model converge faster.
	 - If the learning rate is too high the model may not be able to converge properly.

	 - After our model is defined we need to compile it by providing the optimizer,
	 - the loss function we want to minimize as well as other metrics we want to evaluate during training.
	 - The loss functions implemented in Keras are listed in: https://keras.io/losses/

	 - Since we want evaluate our model with regard to the Unweighted Average Recall,
	   we provide both the accuracy and the UAR as metrics.
	'''
	nn.compile(optimizer=optimizer,
				  loss='binary_crossentropy',
				  metrics=['accuracy', avg_recall])

	# After the model is defined we can print its summary:
	print(nn.summary())

	return nn

# In this function we take our datasets and use them to train the neural network
def train_nn(train_data, train_labels, devel_data, devel_labels, parms):

	# Initialize the network
	nn = initialize_network(train_data.shape, parms['learning_rate'], parms['decay'])

	'''
	Early Stopping:
	 - Keras allows callback functions that are performed after each epoch
	 - Early stopping forces the training to stop if a metric being measured stops improving after a user defined number of epochs.
	 - This can help the model generalize better
	'''
	early = EarlyStopping(monitor='val_loss', min_delta=parms['min_delta'], patience=parms['patience'], verbose=1, mode='auto')

	'''
	NN Training:
	 To train the network we need the training data and labels, and define the batch size and number of epochs:

	 - Batch Size: 		The batch size is the number of samples used at each iteration in an epoch.
	 - Epochs: 			The number of epochs is the number of times the network's weights are adjusted using the whole dataset

	 - Validation Data: To monitor how our network is performing on unseen data at the end of each epoch, we can supply the training function
						with a validation/development dataset.

	'''
	history = nn.fit(train_data, train_labels, epochs=parms['epochs'], batch_size=parms['batch_size'],
					validation_data=(devel_data, devel_labels),
					class_weight=parms['class_weights'],
					callbacks=[early])

	return nn, history

def test_nn(test_data, test_labels, model):

	# After we have trained the model we can compute predictions on unseen data and use them to evaluate other metrics
	preds = model.predict(test_data)

	# In this case we are using as metrics the average F1 Score, Precision and Recall.
	# If we want to learn better how the model is behaving for each class we can remove "average" from the function's inputs
	f1 = precision_recall_fscore_support(test_labels, np.round(preds), labels=[0,1], average='macro')

	return f1, preds

def plot_history(history, filename):

	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(filename+"_acc.png");plt.close()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(filename+"_loss.png");plt.close()

	# Plot training & validation loss values
	plt.plot(history.history['avg_recall'])
	plt.plot(history.history['val_avg_recall'])
	plt.title('Model Recall')
	plt.ylabel('Average Recall')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig(filename+"_rec.png")
