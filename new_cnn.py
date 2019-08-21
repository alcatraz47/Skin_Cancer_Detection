import os
import keras
import pickle
import numpy as np
# from keras.regularizers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten

def load_data():
	pickle_in = open("X.pickle", "rb")
	X = pickle.load(pickle_in)
	pickle_in.close()

	pickle_in = open("y.pickle", "rb")
	y = pickle.load(pickle_in)
	pickle_in.close()

	return X, y	

class Custom_CNN:

	@staticmethod
	def build_model(input_shape):

		input_data = Input(shape = input_shape)

		x = Conv2D(32, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_1')(input_data)#, kernel_regularizer = regularizers.l2(0.01)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)
		x = MaxPooling2D((2.,2))(x)

		x = Conv2D(64, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_2')(x)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)
		x = MaxPooling2D((2.,2))(x)

		x = Conv2D(128, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_3')(x)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)
		x = MaxPooling2D(5,5)(x)

		x = Flatten()(x)
		x = Dense(256, name = 'fcn_1', activation = 'relu')(x)
		# x = Dropout(.25)(x)
		x = Dense(512, name = 'fcn_2', activation = 'relu')(x)
		# x = Dropout(.5)(x)

		x = Dense(7, name = 'predictor', activation = 'softmax')(x)

		model = Model(input_data, x)

		return model

if __name__ == '__main__':

	X, y = load_data()

	custom_cnn = Custom_CNN()

	model = custom_cnn.build_model((100, 100, 3))

	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

	model.fit(X, y, batch_size = 42, epochs = 10, validation_split = 0.3)