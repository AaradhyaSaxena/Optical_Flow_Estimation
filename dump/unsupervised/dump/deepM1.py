from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Lambda
from keras.layers import Add, Dropout, concatenate,Flatten,Dense, MaxoutDense, MaxPooling3D
from keras.layers import Reshape,LocallyConnected2D,SeparableConv1D,LocallyConnected1D,LeakyReLU
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
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z2)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z3)
    z3 = BatchNormalization()(z3)

    z4 = MaxPooling2D(pool_size=(2, 2), strides=2)(z3)
    # z4 = Conv2D(32,(3,3),strides=(2,2),padding='same',activation=act)(z3)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    z4 = BatchNormalization()(z4)
    z5 = Lambda(max_channels32,output_shape=(109,256,32))(z4)

    z6 = Conv2D(32,(3,3), padding='same',activation=act)(z5)
    z6 = BatchNormalization()(z6)
    z6 = Conv2D(32,(3,3), padding='same',activation=act)(z6)
    z6 = BatchNormalization()(z6)
    z6 = Conv2D(32,(3,3), padding='same',activation=act)(z6)
    z6 = BatchNormalization()(z6)

    z6 = Conv2DTranspose(32,(3,3),strides=(2,2), padding='same')(z6)
    z7 = Concatenate(axis=-1)([z6,z3])
    z8 = BatchNormalization()(z7)
    z8 = Conv2D(64,(3,3), padding='same',activation=act)(z8)
    z8 = BatchNormalization()(z8)

    z9 = Lambda(max_channels32,output_shape=(218,512,32))(z8)

    z10 = Conv2D(32,(3,3), padding='same',activation=act)(z9)
    z10 = BatchNormalization()(z10)
    z10 = Conv2D(32,(3,3), padding='same',activation=act)(z10)
    z10 = BatchNormalization()(z10)
    z10 = Conv2D(32,(3,3), padding='same',activation=act)(z10)
    z10 = BatchNormalization()(z10)

    z11 = Lambda(max_channels16,output_shape=(218,512,16))(z10)

    z12 = Conv2D(16,(3,3), padding='same',activation=act)(z11)
    z12 = BatchNormalization()(z12)
    z12 = Conv2D(16,(3,3), padding='same',activation=act)(z12)
    z12 = BatchNormalization()(z12)
    z12 = Conv2D(16,(3,3), padding='same',activation=act)(z12)
    z12 = BatchNormalization()(z12)

    z13 = Conv2DTranspose(16,(3,3),strides=(2,2), padding='same')(z12)
    z14 = Concatenate(axis=-1)([z13,z1])
    z14 = BatchNormalization()(z14)

    z15 = Conv2D(32,(3,3), padding='same',activation=act)(z14)
    z15 = BatchNormalization()(z15)
    z15 = Lambda(max_channels16,output_shape=(436,1024,16))(z15)

    z15 = Conv2D(16,(3,3), padding='same',activation=act)(z15)
    z15 = BatchNormalization()(z15)
    z15 = Lambda(max_channels8,output_shape=(436,1024,8))(z15)

    z15 = Conv2D(8,(3,3), padding='same',activation=act)(z15)
    z15 = BatchNormalization()(z15)
    z15 = Lambda(max_channels4,output_shape=(436,1024,4))(z15)

    z15 = Conv2D(4,(3,3), padding='same',activation=act)(z15)
    z15 = BatchNormalization()(z15)
    z15 = Lambda(max_channels2,output_shape=(436,1024,2))(z15)

    z15 = Conv2D(2,(3,3), padding='same',activation="linear")(z15)
    z15 = BatchNormalization()(z15)
    z15 = Conv2D(2,(3,3), padding='same',activation="linear")(z15)

    model = Model(inputs=[I1,I2], outputs=[z15])
    # model.compile(loss="mse",optimizer='Adam')

    return model
###---------------------loss----------------------------
def compile_model(model,lambda1 = 0.05):

    s1 = tf.get_variable("sig1",  trainable=True,initializer=tf.constant([0.3]))
    s2 = tf.get_variable("sig2",  trainable=True,initializer=tf.constant([0.7]))
    s1_2=s1*s1
    s2_2=s1*s1

    I1=model.inputs[0]
    I2=model.inputs[1]
    o1=model.outputs[0] 

    # this is to calculate the inverse_warp
    o2 = image_warp(-o1,o1)

    I2_rec=image_warp(I1,o1)
    I1_rec=image_warp(I2,o2)

    ux,uy=grad_xy(o1[:,:,:,:1])
    vx,vy=grad_xy(o1[:,:,:,1:2])
    sm_loss=(K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy)))

    # re_loss_mse = K.mean(K.square(I2 - input1_rec))
    re_loss1=DSSIMObjective(kernel_size=50)(I2,I2_rec)
    re_loss2=DSSIMObjective(kernel_size=50)(I1,I1_rec)

    re_loss=re_loss1 + re_loss2

    total_loss=(1/s1_2)*re_loss+(1/s2_2)*sm_loss+K.log(s1_2)+K.log(s2_2)

    model = Model(inputs=[I1,I2], outputs=[o1])
    model.add_loss(total_loss)
    model.compile(optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0))
    
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

# model.load_weights('../data/deepM1_newloss.h5')

# model.fit_generator(imgen,epochs=200)

model.fit([X1,X2],None,epochs=10000)

# model.save_weights("../data/deepM1_newloss_batch.h5")

y=model.predict([X1,X2])
y1 = flow_mag(y)


# np.savez('sample_model1', flow =y1)

####--------------viz-------------------

"""
y=model.predict([X1,X2])
y1 = flow_mag(y)
%matplotlib
plt.figure("flow_pred")
plt.imshow(y1[0])
plt.figure("scene")
plt.imshow(X1[0])

"""

"""
%matplotlib
plt.imshow(y1[0])
plt.imsave("temp1",y1[0])
plt.imsave("tempX1",X1[0])
model.save_weights("data/sampleU12.h5")
"""