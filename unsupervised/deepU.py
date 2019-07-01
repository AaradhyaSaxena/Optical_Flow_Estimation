from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from keras.layers import Add, Dropout, concatenate,Flatten,Dense, MaxoutDense
from keras.layers import Lambda,Reshape,LocallyConnected2D,SeparableConv1D,LocallyConnected1D,LeakyReLU
import os
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
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

###-----------------Model-----------------------

def return_deepU(shape=(436,1024,3)):

    I1 = Input(shape=shape)
    I2 = Input(shape=shape)

    act = "tanh"

    I = Concatenate(axis=-1)([I1,I2])
    I = BatchNormalization()(I)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(I)
    z1 = BatchNormalization()(z1)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(z1)
    z1 = BatchNormalization()(z1)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(z1)
    z1 = BatchNormalization()(z1)

    z2 = Conv2D(16,(3,3),strides=(2,2),padding='same',activation=act)(z1)
    z2 = BatchNormalization()(z2)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z2)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z2)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z2)
    z3 = BatchNormalization()(z3)

    z4 = Conv2D(32,(3,3),strides=(2,2),padding='same',activation=act)(z3)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)

    z5 = Conv2D(32,(3,3), padding='same',activation=act)(z4)
    z5 = BatchNormalization()(z5)
    z5 = Conv2D(32,(3,3), padding='same',activation=act)(z5)
    z5 = BatchNormalization()(z5)
    z5 = Conv2D(32,(3,3), padding='same',activation=act)(z5)
    z5 = BatchNormalization()(z5)

    z6 = Conv2DTranspose(32,(3,3),strides=(2,2), padding='same')(z5)
    z7 = Concatenate(axis=-1)([z6,z3])
    z8 = BatchNormalization()(z7)
    z8 = Conv2D(64,(3,3), padding='same',activation=act)(z8)
    z8 = BatchNormalization()(z8)

    z8 = Conv2D(32,(3,3), padding='same',activation=act)(z8)
    z8 = BatchNormalization()(z8)
    z8 = Conv2D(32,(3,3), padding='same',activation=act)(z8)
    z8 = BatchNormalization()(z8)
    z8 = Conv2D(32,(3,3), padding='same',activation=act)(z8)
    z8 = BatchNormalization()(z8)

    z9 = Conv2D(16,(3,3), padding='same',activation=act)(z8)
    z9 = BatchNormalization()(z9)
    z9 = Conv2D(16,(3,3), padding='same',activation=act)(z9)
    z9 = BatchNormalization()(z9)
    z9 = Conv2D(16,(3,3), padding='same',activation=act)(z9)
    z9 = BatchNormalization()(z9)

    z10 = Conv2DTranspose(16,(3,3),strides=(2,2), padding='same')(z9)
    z11 = Concatenate(axis=-1)([z10,z1])
    z12 = BatchNormalization()(z11)
    z12 = Conv2D(32,(3,3), padding='same',activation=act)(z12)
    z12 = BatchNormalization()(z12)
    z12 = Conv2D(16,(3,3), padding='same',activation=act)(z12)
    z12 = BatchNormalization()(z12)
    z12 = Conv2D(8,(3,3), padding='same',activation=act)(z12)
    z12 = BatchNormalization()(z12)
    z12 = Conv2D(4,(3,3), padding='same',activation=act)(z12)
    z12 = BatchNormalization()(z12)
    z12 = Conv2D(2,(3,3), padding='same',activation=act)(z12)
    z12 = BatchNormalization()(z12)
    z12 = Conv2D(2,(3,3), padding='same',activation=keras.layers.LeakyReLU(alpha=0.3))(z12)
    z13 = BatchNormalization()(z12)

    model = Model(inputs=[I1,I2], outputs=[z13])
    model.compile(loss="mse",optimizer='Adam')

    return model
###---------------------loss----------------------------
def compile_model(model,lambda1 = 0.005):

    I1=model.inputs[0]
    I2=model.inputs[1]
    o1=model.outputs[0] 

    input1_rec=image_warp(I1,o1)
    # input0_rec=image_warp(model.inputs[1],-model.outputs[0])

    ux,uy=grad_xy(o1[:,:,:,:1])
    vx,vy=grad_xy(o1[:,:,:,1:2])
    sm_loss=lambda1*(K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy)))

    re_loss_mse = K.mean(K.square(I2 - input1_rec))

    # loss_mse = K.mean(K.square(model.outputs[0] - Y))

    total_loss = lambda1*sm_loss+re_loss_mse # + loss_mse

    model.add_loss(total_loss)
    model = Model(inputs=[I1,I2], outputs=[o1])
    model.compile(loss="mse",optimizer='Adam')
   	
    return model

###--------------------compile--------------------------

model_base = return_deepU()

model = compile_model(model_base)
###---------------------debug

# model = return_deepU()

#-------------------DATA-------------------

imgen=ImageSequence_fixed()
[X1,X2],Y = imgen.__getitem__()

# imgen=ImageSequence_new()
# [X1,X2],Y = imgen.__getitem__()

#-------------------------Training-----------

model.load_weights('../data/deepU1.h5')

# model.fit_generator(imgen,epochs=2000)

model.fit([X1,X2],Y,epochs=10000)

model.save_weights("../data/deepU1_fxd_after500.h5")

# y=model.predict([X1,X2])
# y1 = flow_mag(y)


# np.savez('sample_model1', flow =y1)

####--------------viz-------------------

"""
%matplotlib
y=model.predict([X1,X2])
y1 = flow_mag(y)
plt.imshow(y1[0])

"""

"""
%matplotlib
plt.imshow(y1[0])
plt.imsave("temp1",y1[0])
model.save_weights("data/sampleU12.h5")
"""