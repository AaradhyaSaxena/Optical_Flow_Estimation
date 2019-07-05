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
from NewLoss import *

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
###--------occlusion_aware_loss-------------------------
def compile_model(model,lambda_smoothness = 0, lambda_flow=0.0001, lambda_mse=0, occ_punishment = 0):

    i1=model.inputs[0]
    i2=model.inputs[1]
    o1=model.outputs[0]
    o2 = image_warp(-o1,o1)

    oxf, oxb = mask(i1,i2,o1,o2)
    mask_f = oxf[:,:,:,0]
    mask_b = oxb[:,:,:,1]

    err_f, err_b = photometric_error(i1,i2,o1,o2)
    flow_f, flow_b = flow_error(o1,o2)

    ###--------Occlusion_aware_mse_rec_image-------------------------------------
    occ_loss1 = (tf.reduce_sum(tf.boolean_mask(charbonnier(err_f), mask_f)))#/(436*1024)
    occ_loss2 = (tf.reduce_sum(tf.boolean_mask(charbonnier(err_b), mask_b)))#/(436*1024)
    occ_loss = (occ_loss1 + occ_loss2)*lambda_mse

    ###--------Occlusion_aware_mse_flow------------------------------------
    flow_loss1 = tf.reduce_sum(tf.boolean_mask(charbonnier(flow_f), mask_f))
    flow_loss2 = tf.reduce_sum(tf.boolean_mask(charbonnier(flow_b), mask_f))
    flow_loss = (flow_loss1 + flow_loss2)*lambda_flow

    ###--------Punishment_for_occlusion-----------------------------------------
    occ_punish1 = tf.multiply(tf.reduce_sum(tf.cast(mask_f, tf.float32)),occ_punishment)
    occ_punish2 = tf.multiply(tf.reduce_sum(tf.cast(mask_b, tf.float32)),occ_punishment)
    occ_punish = occ_punish1 + occ_punish2

    ###--------Gradient_smoothness--------------------------------------------
    ux,uy=grad_xy(o1[:,:,:,:1])
    vx,vy=grad_xy(o1[:,:,:,1:2])
    sm_loss_o1 = K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy))
    ux,uy=grad_xy(o2[:,:,:,:1])
    vx,vy=grad_xy(o2[:,:,:,1:2])
    sm_loss_o2 = K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy))
    sm_loss = (sm_loss_o1 + sm_loss_o2)*lambda_smoothness   

    ### Reconstruction_loss_ssim_(occlusion_not considered)
    i2_rec=image_warp(i1,o1)
    i1_rec=image_warp(i2,o2)
    re_loss1=DSSIMObjective(kernel_size=50)(i2,i2_rec)
    re_loss2=DSSIMObjective(kernel_size=50)(i1,i1_rec)
    re_loss_ssim = re_loss1 + re_loss2

    total_loss = sm_loss + occ_loss + occ_punish + re_loss_ssim + flow_loss 

    model = Model(inputs=[i1,i2], outputs=[o1])
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

# model.load_weights('../data/deepM2.h5')

# model.fit_generator(imgen,epochs=2000)

model.fit([X1,X2],None,epochs=5000)

# model.save_weights("../data/deepM2_loss_new.h5")


###_--------------------test------------------

# imgen=ImageSequence_new()
# [X1,X2],Y = imgen.__getitem__()

# y=model.predict([X1,X2])
# y1 = flow_mag(y)
plt.imsave("testy",y1[0])
plt.imsave("testX",X1[0])

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
model.save_weights("data/sampleU12.h5")
"""