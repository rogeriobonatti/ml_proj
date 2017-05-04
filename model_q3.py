from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
import cv2, numpy as np

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(1,224,224), name='conv1_pad1_finetune'))
    model.add(Convolution2D(64, 3, 3,name='conv1_act1_finetune'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='conv1_pad2'))
    model.add(Convolution2D(64, 3, 3,name='conv1_act2'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='conv1_max'))

    model.add(ZeroPadding2D((1,1),name='conv2_pad1'))
    model.add(Convolution2D(128, 3, 3,name='conv2_act1'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='conv2_pad2'))
    model.add(Convolution2D(128, 3, 3,name='conv2_act2'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='conv2_max'))

    model.add(ZeroPadding2D((1,1),name='conv3_pad1'))
    model.add(Convolution2D(256, 3, 3,name='conv3_act1'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='conv3_pad2'))
    model.add(Convolution2D(256, 3, 3,name='conv3_act2'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='conv3_pad3'))
    model.add(Convolution2D(256, 3, 3,name='conv3_act3'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='conv3_max'))

    model.add(ZeroPadding2D((1,1),name='conv4_pad1'))
    model.add(Convolution2D(512, 3, 3,name='conv4_act1'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='conv4_pad2'))
    model.add(Convolution2D(512, 3, 3,name='conv4_act2'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='conv4_pad3'))
    model.add(Convolution2D(512, 3, 3,name='conv4_act3'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='conv4_max'))

    model.add(ZeroPadding2D((1,1),name='conv5_pad1'))
    model.add(Convolution2D(512, 3, 3,name='conv5_act1'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='conv5_pad2'))
    model.add(Convolution2D(512, 3, 3,name='conv5_act2'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1,1),name='conv5_pad3'))
    model.add(Convolution2D(512, 3, 3,name='conv5_act3'))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2),name='conv5_max'))

    model.add(Flatten(name='fc_flatten_finetune'))
    model.add(Dense(4096,name='fc_act1_finetune'))
    model.add(BatchNormalization(axis=1))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5,name='fc_dp1_finetune'))
    # model.add(Dense(4096,name='fc_act2_finetune'))
    # model.add(BatchNormalization(axis=1))

    model.summary()

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model