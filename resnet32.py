#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.models import *
from keras.layers import *
from keras.layers.normalization import *
from keras.layers.convolutional import *
from keras.regularizers import *
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras import backend as K


# In[2]:


class Resnet32:
    @staticmethod
    def resnet(data, num_filters, stride, channel_dimension, reduce_dimension, reg, bnEps, bnMom):
        shortcut = data
        
        #bn -> ac -> conv2d(stride = (1,1))
        bn_1 = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(data)
        ac_1 = Activation("relu")(bn_1)
        
        conv_1 = Conv2D(int(num_filters * .25), (1, 1), use_bias = False, kernel_regularizer = l2(reg))(act_1)
        
        #bn -> ac -> conv2d(stride = (3,3))
        bn_2 = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(conv_1)
        ac_2 = Activation("relu")(bn_2)
        
        conv_2 = Conv2D(int(num_filters * .25), (3, 3), use_bias = False, kernel_regularizer = l2(reg))(act_2)
        
        #bn -> ac -> conv2d(stride = (1, 1))
        bn_3 = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(conv_2)
        ac_3 = Activation("relu")(bn_3)
        
        conv_3 = Conv2D(num_filters, (1, 1), use_bias = False, kernel_regularizer = l2(reg))(act_3)
        
        #for spatial size reducing
        if reduce_dimension:
            shortcut = Conv2D(num_filters, (1, 1), use_bias = False, kernel_regularizer = l2(reg))(act_1)
        
        #final conv layer adding
        resnet = add([conv_3, shortcut])
        return resnet
        
    @staticmethod
    def build_model(height, width, depth, classes, stages, num_filters, reg = 0.0001, 
                    bnEps = 2e-5, bnMom = 0.9): 
        input_shape = (height, width, depth)
        channel_dimension = -1
        
        #data driven channel dimension
        if(K.image_data_format() == "channels_first"):
            input_shape = (depth, height, width)
            channel_dimension = 1
        
        #input -> bn to feed into resnet
        inputs = Input(shape = input_shape)
        x = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(inputs)
        
        #conv2d(stride = (5, 5)) -> bn -> act -> pool
        x = Conv2D(num_filters[0], (5, 5), use_bias = False, padding = "same", kernel_regularizer = l2(reg))(x)
        x = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(x)
        x = Activation("relu")(x)
        x = ZeroPadding2D(1, 1)(x)
        x = MaxPooling2D((3, 3), strides = (2, 2))(x)
        
        #stacking up resnets
        for i in range(0, len(stages)):
            if i == 0:
                stride = (1, 1)
            else:
                stride = (2, 2)
            
            x = Resnet32.resnet(x, num_fliters[i+1], stride, channel_dimension, reduce_dimension = True, 
                                bnEps = bnEps, bnMom = bnMom)
            
            for j in range(0, stages[i] - 1):
                x = Resnet32.resnet(x, num_filters[i+1], (1, 1), channel_dimension, bnEps = bnEps, bnMom = bnMom)

                
        #avoid ffc and use average pooling
        x = BatchNormalization(axis = channel_dimension, epsilon = bnEps, momentum = bnMom)(x)
        x = Activation("relu")(x)
        x = AveragePooling2D(8, 8)(x)
        
        #softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_initializer = l2(reg))(x)
        x = Activation("softmax")(x)
        
        model = Model(inputs, x, name = "resnet_32")
        return model


# In[ ]:




