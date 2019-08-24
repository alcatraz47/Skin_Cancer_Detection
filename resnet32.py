import os
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras import backend as K
import matplotlib as mpl
mpl.use('TkAgg')


def load_data():
    pickle_in = open("X.pickle", "rb")
    X = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("y.pickle", "rb")
    y = pickle.load(pickle_in)
    pickle_in.close()

    return X, y



def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('resnet32.png')
    plt.show()




class Resnet32:
    @staticmethod
    def resnet(data, num_filters, stride, channel_dimension, reduce_dimension, reg, bnEps, bnMom):
        shortcut = data

        #bn -> ac -> conv2d(stride = (1,1))
        bn_1 = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(data)
        ac_1 = Activation("relu")(bn_1)
        print(ac_1)

        print("accessed 1")
        conv_1 = Conv2D(int(num_filters * .25), (1, 1), padding = 'same', use_bias = False, kernel_regularizer = l2(reg))(ac_1)
        print("conv_1: "+str(conv_1))

        #bn -> ac -> conv2d(stride = (3,3))
        bn_2 = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(conv_1)
        ac_2 = Activation("relu")(bn_2)

        print("accessed 2")
        conv_2 = Conv2D(int(num_filters * .25), (3, 3), padding = 'same', use_bias = False, kernel_regularizer = l2(reg))(ac_2)
        print("conv_2: "+str(conv_2))

        #bn -> ac -> conv2d(stride = (1, 1))
        bn_3 = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(conv_2)
        ac_3 = Activation("relu")(bn_3)

        print("accessed 3")
        conv_3 = Conv2D(num_filters, (1, 1), padding = 'same', use_bias = False, kernel_regularizer = l2(reg))(ac_3)
        print("conv_3: "+str(conv_3))

        #for spatial size reducing
        if reduce_dimension:
            print("accessed 4")
            shortcut = Conv2D(num_filters, (1, 1), use_bias = False, kernel_regularizer = l2(reg))(ac_1)
            print("shortcut: "+str(shortcut))

        #final conv layer adding
        print("accessed 5")
        resnet = add([conv_3, shortcut])
        return resnet

    @staticmethod
    def build_model(height, width, depth, classes, stages, num_filters, reg, bnEps, bnMom): 
        input_shape = (height, width, depth)
        channel_dimension = -1

        #data driven channel dimension
        if(K.image_data_format() == "channels_first"):
            input_shape = (depth, height, width)
            channel_dimension = 1

        #input -> bn to feed into resnet
        inputs = Input(shape = input_shape)
        x = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(inputs)
        print("x: "+str(x))

        #conv2d(stride = (5, 5)) -> bn -> act -> pool
        x = Conv2D(num_filters[0], (5, 5), use_bias = False, padding = "same", kernel_regularizer = l2(reg))(x)
        print("build-conv_1: "+str(x))

        x = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D((1, 1))(x)
        x = MaxPooling2D((3, 3), strides = (2, 2))(x)
        print("build-maxpool: "+str(x))

        #stacking up resnets
        for i in range(0, len(stages)):
            print(i)
            if i == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)

            x = Resnet32.resnet(x, num_filters[i+1], stride, channel_dimension, reduce_dimension = True, reg = reg, bnEps = bnEps, bnMom = bnMom)
            print("accessed 6")

            for j in range(0, stages[i] - 1):
                print("accessed j = %d" %j)
                x = Resnet32.resnet(x, num_filters[i+1], (1, 1), channel_dimension, reduce_dimension = True, reg = reg, bnEps = bnEps, bnMom = bnMom)


        #avoid ffc and use average pooling
        x = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(x)
        x = Activation("relu")(x)
        print("accessed 7")
        x = AveragePooling2D(8, 8)(x)
        print("accessed 8")

        #softmax classifier
        x = Flatten()(x)
        x = Dense(len(classes))(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name = "resnet_32")
        return model


if __name__ == '__main__':

    CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    X, y = load_data()

    stages = (3, 4, 6)
    resnet = Resnet32()
    num_filters = (16, 32, 64, 128)
    model = resnet.build_model(50, 50, 3, CLASSES, stages, num_filters, float(0.0001), float(2e-5), float(0.9))


    epochs = 100
    batch_size = 32
    initial_learning_rate = 1e-4
    opt = Adam(lr = initial_learning_rate, decay = initial_learning_rate / epochs)    

    model.summary()
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])
    model_info = model.fit(X, y, batch_size = batch_size, epochs = epochs, validation_split = 0.2)

    plot_model_history(model_info)

    model_json = model.to_json()
    with open("cnn_model_resnet32.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("cnn_model_resnet32.h5")
    print("Saved model to disk")
