from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Lambda
from keras.layers import Add, Dropout, concatenate,Flatten,Dense, MaxoutDense, MaxPooling3D
from keras.layers import Reshape,LocallyConnected2D,SeparableConv1D,LocallyConnected1D,LeakyReLU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='0'
from keras.utils import plot_model
import keras.backend as K
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.convolutional import Conv3D,Conv3DTranspose
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.models import Model,load_model
from keras.layers.core import Reshape,Flatten,Dense
from keras.layers.merge import Concatenate
#from init import *
import numpy as np
import  matplotlib.pyplot as plt
import keras.backend as K
import sys
from  scipy.misc import  imsave,imread
from keras.layers.core import Lambda
import keras
import cv2
#from generator import *
from keras_contrib.losses import DSSIMObjective
from image_warp import *
#from image_warp_keras import *
from scipy import misc
import keras_contrib.backend as KC
from keras.utils import multi_gpu_model
import tensorflow as tf
from generator import *
# from model import *
from model_utils import *
from NewLoss import *

###--------------------------DeepM---------------------------------
def return_deepM(shape=(436,1024,3)):

    I1 = Input(shape=shape)
    I2 = Input(shape=shape)

    act = "tanh"

    I = Concatenate(axis=-1)([I1,I2])
    # I = BatchNormalization()(I)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(I)
    z1 = BatchNormalization()(z1)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(z1)
    z1 = BatchNormalization()(z1)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(z1)
    z1 = BatchNormalization()(z1)

    z2 = MaxPooling2D(pool_size=(2, 2), strides=2)(z1)
    # z2 = Conv2D(16,(3,3),strides=(2,2),padding='same',activation=act)(z1)
    z2 = BatchNormalization()(z2)
    z2 = Conv2D(32,(3,3), padding='same',activation=act)(z2)
    z2 = BatchNormalization()(z2)
    z2 = Conv2D(32,(3,3), padding='same',activation=act)(z2)
    z2 = BatchNormalization()(z2)
    z2 = Conv2D(32,(3,3), padding='same',activation=act)(z2)
    z2 = BatchNormalization()(z2)

    z3 = MaxPooling2D(pool_size=(2, 2), strides=2)(z2)
    # z3 = Conv2D(32,(3,3),strides=(2,2),padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(64,(3,3), padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(64,(3,3), padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(64,(3,3), padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)
    z3 = Lambda(max_channels32,output_shape=(109,256,32))(z3)

    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)

    z3 = Conv2DTranspose(32,(3,3),strides=(2,2), padding='same')(z3)
    z4 = Concatenate(axis=-1)([z3,z2])
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)

    z4 = Lambda(max_channels32,output_shape=(218,512,32))(z4)

    z4 = Conv2D(32,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(32,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(32,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)

    z4 = Lambda(max_channels16,output_shape=(218,512,16))(z4)

    z4 = Conv2D(16,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(16,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(16,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)

    z4 = Conv2DTranspose(16,(3,3),strides=(2,2), padding='same')(z4)
    z5 = Concatenate(axis=-1)([z4,z1])
    z5 = BatchNormalization()(z5)

    z5 = Conv2D(32,(3,3), padding='same',activation=act)(z5)
    z5 = BatchNormalization()(z5)
    z5 = Lambda(max_channels16,output_shape=(436,1024,16))(z5)

    z5 = Conv2D(16,(3,3), padding='same',activation=act)(z5)
    z5 = BatchNormalization()(z5)
    z5 = Lambda(max_channels8,output_shape=(436,1024,8))(z5)

    z5 = Conv2D(8,(3,3), padding='same',activation=act)(z5)
    z5 = BatchNormalization()(z5)
    z5 = Lambda(max_channels4,output_shape=(436,1024,4))(z5)

    z5 = Conv2D(4,(3,3), padding='same',activation=act)(z5)
    z5 = BatchNormalization()(z5)
    z5 = Lambda(max_channels2,output_shape=(436,1024,2))(z5)

    z5 = Conv2D(2,(3,3), padding='same',activation="linear")(z5)
    z5 = BatchNormalization()(z5)
    z6 = Conv2D(2,(3,3), padding='same',activation="linear")(z5)

    model = Model(inputs=[I1,I2], outputs=[z6])
    # model.compile(loss="mse",optimizer='Adam')

    return model
#########################----------------------------------