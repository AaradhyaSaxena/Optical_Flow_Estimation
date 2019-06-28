from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, concatenate,Flatten,Dense
from keras.layers import Lambda,Reshape,LocallyConnected2D,SeparableConv1D,LocallyConnected1D,LeakyReLU
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
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
from model import *
from loss import *

###-----------------Model-----------------------

def return_model_U1(shape=(436,1024,3)):

    I1 = Input(shape=shape)
    I2 = Input(shape=shape)

    I = Concatenate(axis=-1)([I1,I2])
    z1 = Conv2D(12,(5,5), padding='same',activation="relu")(I)
    z2 = MaxPooling2D((2,2))(z1)
    z3 = Conv2D(24,(5,5), padding='same',activation="relu")(z2)
    z4 = MaxPooling2D((2,2))(z3)
    z5 = Conv2D(12,(5,5), padding='same',activation="relu")(z4)

    z6 = Conv2DTranspose(12,(5,5),strides=(2,2), padding='same')(z5)
    z6_concat = Concatenate(axis=-1)([z6,z3])

    z7 = Conv2D(6,(5,5), padding='same',activation="relu")(z6_concat)
    z8 = Conv2DTranspose(6,(5,5),strides=(2,2), padding='same')(z7)
    z8_concat = Concatenate(axis=-1)([z8,z1])

    z9 = Conv2D(2,(5,5), padding='same',activation="relu")(z8_concat)

    model = Model(inputs=[I1,I2], outputs=[z9])

    # model.compile(loss=c_mse,optimizer="Adam")
    # model.compile(loss=c_mseAbs,optimizer="Adam")
    # model.compile(loss=c_grad,optimizer="Adam")
    model.compile(loss=c_grad_ssim,optimizer="Adam")
    # model.compile(loss="mse",optimizer="Adam")
    # model.compile(loss=DSSIMObjective(kernel_size=75),optimizer="Adam")


    return model
##------------------------------------------

model=return_model_U1()
# model=create_model(input_shape=(436,1024,3))
# model.compile(loss="mse",optimizer="Adam")

#-------------------DATA-------------------

# X1 = np.random.rand(100,436,1024,3)
# X2 = np.random.rand(100,436,1024,3)
# y = np.random.rand(100,436,1024,2)


imgen=ImageSequence_new()
[X1,X2],Y = imgen.__getitem__()


#-------------------------Training-----------

model.load_weights('data/grad_ssim_full.h5')

model.fit_generator(imgen,epochs=200)

# model.fit([X1,X2],Y,batch_size=4,epochs=10000)
# model.load_weights('data/grad_ssim_full')

y=model.predict([X1,X2])

model.save_weights("data/grad_ssim_full1.h5")

# np.savez('sample_model1', flow =y1)

####--------------viz-------------------
# %matplotlib
# y=model.predict([X1,X2])
# plt.imshow(y[0,:,:,0])