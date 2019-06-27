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

def return_model(shape=(436,1024,3)):

    I1 = Input(shape=shape)
    I2 = Input(shape=shape)

    I = Concatenate(axis=-1)([I1,I2])
    z = Conv2D(12,(5,5), padding='same',activation="relu")(I)
    z = MaxPooling2D((2,2))(z)
    z = Conv2D(24,(5,5), padding='same',activation="relu")(z)
    z = MaxPooling2D((2,2))(z)
    z = Conv2D(12,(5,5), padding='same',activation="relu")(z)

    z = Conv2DTranspose(12,(5,5),strides=(2,2), padding='same')(z)
    z = Conv2D(6,(5,5), padding='same',activation="relu")(z)
    z = Conv2DTranspose(6,(5,5),strides=(2,2), padding='same')(z)
    z = Conv2D(2,(5,5), padding='same',activation="relu")(z)

    model = Model(inputs=[I1,I2], outputs=[z])
    model.compile(loss=c_mseAbs,optimizer="Adam")
    # model.compile(loss="mse",optimizer="Adam")
    # model.compile(loss=DSSIMObjective(kernel_size=75),optimizer="Adam")


    return model
##------------------------------------------

model=return_model()

# model=create_model(input_shape=(436,1024,3))
# model.compile(loss="mse",optimizer="Adam")

#-------------------DATA-------------------

# X1 = np.random.rand(100,436,1024,3)
# X2 = np.random.rand(100,436,1024,3)
# y = np.random.rand(100,436,1024,2)


imgen=ImageSequence_new()
[X1,X2],Y = imgen.__getitem__()


#-------------------------Training-----------


# model.fit_generator(imgen,epochs=100)

model.fit([X1,X2],Y,batch_size=4,epochs=100)
# model.load_weights('data/model_1')

y=model.predict([X1,X2])

model.save_weights("data/temp.h5")

# np.savez('sample_model1', flow =y1)

####--------------viz-------------------
# %matplotlib
# plt.imshow(y[0,:,:,0])