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

    #act = keras.layers.LeakyReLU(alpha=0.3)
    act="tanh"

    I = Concatenate(axis=-1)([I1,I2])
    #I = BatchNormalization()(I)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(I)
    #z1 = BatchNormalization()(z1)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(z1)
    #z1 = BatchNormalization()(z1)
    z1 = Conv2D(16,(3,3), padding='same',activation=act)(z1)
    #z1 = BatchNormalization()(z1)

    z2 = Conv2D(16,(3,3),strides=(2,2),padding='same',activation=act)(z1)
    #z2 = BatchNormalization()(z2)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z2)
    #z3 = BatchNormalization()(z3)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z3)
    #z3 = BatchNormalization()(z3)
    z3 = Conv2D(32,(3,3), padding='same',activation=act)(z3)
    #z3 = BatchNormalization()(z3)

    z4 = Conv2D(32,(3,3),strides=(2,2),padding='same',activation=act)(z3)
    #z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    #z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    #z4 = BatchNormalization()(z4)
    z4 = Conv2D(64,(3,3), padding='same',activation=act)(z4)
    #z4 = BatchNormalization()(z4)

    z5 = Conv2D(32,(3,3), padding='same',activation=act)(z4)
    #z5 = BatchNormalization()(z5)
    z5 = Conv2D(32,(3,3), padding='same',activation=act)(z5)
    #z5 = BatchNormalization()(z5)
    z5 = Conv2D(32,(3,3), padding='same',activation=act)(z5)
    #z5 = BatchNormalization()(z5)

    z6 = Conv2DTranspose(32,(3,3),strides=(2,2), padding='same')(z5)
    z7 = Concatenate(axis=-1)([z6,z3])
    #z8 = BatchNormalization()(z7)
    z8 = Conv2D(64,(3,3), padding='same',activation=act)(z7)
    #z8 = BatchNormalization()(z8)

    z8 = Conv2D(32,(3,3), padding='same',activation=act)(z8)
    #z8 = BatchNormalization()(z8)
    z8 = Conv2D(32,(3,3), padding='same',activation=act)(z8)
    #z8 = BatchNormalization()(z8)
    z8 = Conv2D(32,(3,3), padding='same',activation=act)(z8)
    #z8 = BatchNormalization()(z8)

    z9 = Conv2D(16,(3,3), padding='same',activation=act)(z8)
    #z9 = BatchNormalization()(z9)
    z9 = Conv2D(16,(3,3), padding='same',activation=act)(z9)
    #z9 = BatchNormalization()(z9)
    z9 = Conv2D(16,(3,3), padding='same',activation=act)(z9)
    #z9 = BatchNormalization()(z9)

    z10 = Conv2DTranspose(16,(3,3),strides=(2,2), padding='same')(z9)
    z11 = Concatenate(axis=-1)([z10,z1])
    #z12 = BatchNormalization()(z11)
    z12 = Conv2D(32,(3,3), padding='same',activation=act)(z11)
    #z12 = BatchNormalization()(z12)
    z12 = Conv2D(16,(3,3), padding='same',activation=act)(z12)
    #z12 = BatchNormalization()(z12)
    z12 = Conv2D(8,(3,3), padding='same',activation=act)(z12)
    #z12 = BatchNormalization()(z12)
    z12 = Conv2D(4,(3,3), padding='same',activation=act)(z12)
    #z12 = BatchNormalization()(z12)
    z12 = Conv2D(2,(3,3), padding='same',activation="linear")(z12)
    #z12 = BatchNormalization()(z12)
    z12 = Conv2D(2,(3,3), padding='same',activation="linear")(z12)

    model = Model(inputs=[I1,I2], outputs=[z12])
    #model.compile(loss="mse",optimizer='Adam')

    return model
###---------------------loss----------------------------
def compile_model(model,lambda1 = 0.005):
    s1 = tf.get_variable("sig1",  trainable=True,initializer=tf.constant([0.3]))
    s2 = tf.get_variable("sig2",  trainable=True,initializer=tf.constant([0.7]))
    s1_2=s1*s1
    s2_2=s2*s2

    I1=model.inputs[0]
    I2=model.inputs[1]
    o1=model.outputs[0] 

    I2_rec=image_warp(I1,o1)
    I1_rec=image_warp(I2,-o1)

    ux,uy=grad_xy(o1[:,:,:,:1])
    vx,vy=grad_xy(o1[:,:,:,1:2])
    sm_loss=(K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy)))

    # re_loss_mse = K.mean(K.square(I2 - input1_rec))
    re_loss1=DSSIMObjective(kernel_size=50)(I2,I2_rec)
    re_loss2=DSSIMObjective(kernel_size=50)(I1,I1_rec)

###############################################
    # input1_rec=image_warp(model.inputs[0],model.outputs[0],num_batch=b_size)
    # input0_rec=image_warp(model.inputs[1],-model.outputs[0],num_batch=b_size)
    # ux,uy=grad_xy(model.outputs[0][:,:,:,:1])
    # vx,vy=grad_xy(model.outputs[0][:,:,:,1:2])
    # sm_loss=lambda1*(K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy)))
    # re_loss=DSSIMObjective(kernel_size=50)(model.inputs[1],input1_rec)
################################################

    # loss_mse = K.mean(K.square(model.outputs[0] - Y))

    re_loss=re_loss1 + re_loss2


    total_loss=(1/s1_2)*re_loss+(1/s2_2)*sm_loss+K.log(s1_2)+K.log(s2_2)



    model = Model(inputs=[I1,I2], outputs=[o1])
    model.add_loss(total_loss)
    model.compile(loss="mse",optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0))
    #model.compile(optimizer='rmsprop')
   	
    return model

###--------------------compile--------------------------

model_base = return_deepU()
model = compile_model(model_base,lambda1=0.01)





#-------------------DATA-------------------

imgen=ImageSequence_new()
[X1,X2],Y = imgen.__getitem__()





"""
#-------------------------Training-----------

model.load_weights('../data/deep_U.h5')

model.fit_generator(imgen,epochs=2000)

# model.fit([X1,X2],Y,epochs=10000)

model.save_weights("../data/deep_U.h5")





y=model.predict([X1,X2])
y1 = flow_mag(y)
# plt.imsave("test1",y1[0])

# np.savez('sample_model1', flow =y1)

####--------------viz-------------------

%matplotlib
y=model.predict([X1,X2])
y1 = flow_mag(y)
y2 = flow_mag(Y)
plt.figure("out")
plt.imshow(y1[0])
plt.figure("gt")
plt.imshow(y2[0])

plt.figure("X1")
plt.imshow(X1[0])
plt.figure("X2")
plt.imshow(X2[0])
"""

"""
%matplotlib
plt.imshow(y1[0])
plt.imsave("temp1",y1[0])
model.save_weights("data/sampleU12.h5")
"""
"""
plt.imsave("tempY",Y[0])
"""
