import os
import keras
import pickle
import numpy as np
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam

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

		x = Conv2D(32, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_1_1', kernel_regularizer = regularizers.l2(0.01))(input_data)#, kernel_regularizer = regularizers.l2(0.01)
		# x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)


		x = Conv2D(32, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_1_2', kernel_regularizer = regularizers.l2(0.01))(x)#, kernel_regularizer = regularizers.l2(0.01)
		# x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)


		x = Conv2D(32, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_1_3', kernel_regularizer = regularizers.l2(0.01))(x)#, kernel_regularizer = regularizers.l2(0.01)
		# x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)
		x = MaxPooling2D((2,2))(x)


		x = Conv2D(64, kernel_size = (3,3), strides = (1,1), activation = 'relu', name = 'Conv2D_2_1')(x)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)

		x = Conv2D(64, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_2_2')(x)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)

		x = Conv2D(64, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_2_3')(x)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)

		x = Conv2D(64, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_2_4')(x)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)
		x = MaxPooling2D((2,2))(x)



		x = Conv2D(128, kernel_size = (3,3), strides = (1,1), activation = 'relu', name = 'Conv2D_3')(x)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)

		x = Conv2D(128, kernel_size = (3,3), padding = 'same', strides = (1,1), activation = 'relu', name = 'Conv2D_4')(x)
		x = Dropout(.25)(x)
		x = BatchNormalization(axis = -1)(x)
		x = MaxPooling2D(3,3)(x)


		x = Flatten()(x)
		

		x = Dense(256, name = 'fcn_1', activation = 'relu')(x)
		x = Dropout(.25)(x)

		x = Dense(256, name = 'fcn_2', activation = 'relu')(x)
		x = Dropout(.25)(x)


		x = Dense(512, name = 'fcn_3', activation = 'relu')(x)
		x = Dropout(.5)(x)

		x = Dense(7, name = 'predictor', activation = 'softmax')(x)

		model = Model(input_data, x)

		return model

if __name__ == '__main__':

	X, y = load_data()

	custom_cnn = Custom_CNN()

	model = custom_cnn.build_model((50, 50, 3))

	optim = Adam(lr = 0.001)

	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	model.summary()

	model.fit(X, y, batch_size = 42, epochs = 50, validation_split = 0.2)

	model_json = model.to_json()
	with open("skin_cancer_detector.json", "w") as json_file:
		json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights("skin_cancer_detector.h5")
	print("Saved model to disk")