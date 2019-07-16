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
from keras_contrib.losses import DSSIMObjective
from scipy import misc
import keras_contrib.backend as KC
from keras.utils import multi_gpu_model
import tensorflow as tf
from image_warp import *
from generator import *
from model_utils import *
from NewLoss import *
from network import *

###----------------------unflow---------------------------------

def combined(shape=(436,1024,3)):

    model = return_deepM()
    I1 = Input(shape=shape)
    I2 = Input(shape=shape)
    F1 = model([I1,I2])
    F2 = model([I2,I1])  
    model_all = Model(inputs=[I1,I2], outputs=[F1,F2])
    return model_all, model

###-----------------compile_model---------------------------
def compile_model(model1,lambda_smoothness=0.01,lambda_ssim=5,lambda_mse=0.0002,lambda_flow=0.0001,occ_punishment=0.0002):

    i1=model1.inputs[0]
    i2=model1.inputs[1]
    o1=model1.outputs[0]
    o2=model1.outputs[1]

    ###--------Flow_continuity------------------------------------------------
    double_recon1 = image_warp((image_warp(i1,o2)),o1)
    double_recon2 = image_warp((image_warp(i2,o1)),o2)
    mse_image_recon = K.mean(K.square(i1-double_recon1)) + K.mean(K.square(i2-double_recon2))       


    ###--------Smoothness--------------------------------------------
    g1 = tf.image.rgb_to_grayscale(i1[:,:,:,:])
    i1x,i1y=grad_xy(g1[:,:,:,:1])

    fm = flow_mag_tensor(o1)
    fmx, fmy = grad_xy(fm[:,:,:])

    # ux,uy=grad_xy(o1[:,:,:,:1])
    # vx,vy=grad_xy(o1[:,:,:,1:2])
    # body_sm_loss = K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy))

    edge_sm_new = K.abs(fmx)*(1-K.exp(-(K.abs(i1y)))) + K.abs(fmy)*(1-K.exp(-(K.abs(i1x))))
    # edge_sm_old = K.abs(fmx)*K.exp(-(K.abs(i1x))) + K.abs(fmy)*K.exp(-(K.abs(i1y)))
    # sm_loss = body_sm_loss*(0.01) + edge_sm_old*(0.01)

    ###--------Reconstruction_loss_ssim_(occlusion_not considered)
    i1_rec=image_warp(i2,o1)
    i2_rec=image_warp(i1,o2)

    ###-----contrast_and_luminous-----------------------------------
    # CL_loss = DSSIM_updated(i2,i2_rec) + DSSIM_updated(i1,i1_rec)

	###-------------------------------------------------------------

    #gradient loss
    mse = K.mean(K.square(i2-i2_rec)) +K.mean(K.square(i1-i1_rec))
    i1_ux,i1_uy = grad_xy(i1)
    i2_ux,i2_uy = grad_xy(i2)
    i1_rec_ux,i1_rec_uy = grad_xy(i1_rec)
    i2_rec_ux,i2_rec_uy = grad_xy(i2_rec)
    mse_grad = K.mean(K.square(i2_ux-i2_rec_ux)) + K.mean(K.square(i2_uy-i2_rec_uy)) + K.mean(K.square(i1_ux-i1_rec_ux)) + K.mean(K.square(i1_uy-i1_rec_uy))
    grad_loss = mse + mse_grad

    total_loss = edge_sm_new*(0.1) + mse_image_recon + grad_loss #+ CL_loss

    model1.add_loss(total_loss)
    model1.compile(optimizer="adadelta")

    return model1
###--------------------compile---------------------------------

model1, model2 = combined()
model1 = compile_model(model1)

#-------------------DATA-------------------

# imgen=ImageSequence_fixed()
# [X1,X2],Y = imgen.__getitem__()

imgen=ImageSequence_new()
[X1,X2],Y = imgen.__getitem__()

#-------------------------Training-----------
# model1.load_weights("../data/old_sm_loss/unflow_001sm_imgrecon_grad_14.h5")
model1.load_weights("../data/unflow_001sm_new_imgrecon_grad_2.h5")
model1.fit_generator(imgen,epochs=10001)
# model1.fit([X1,X2],None,epochs=400)
# model1.save_weights("../data/unflow_001sm_new_imgrecon_grad_3.h5")





###_--------------------test------------------

# """
# imgen=ImageSequence_new()
# [X1,X2],Y = imgen.__getitem__()
# y=model1.predict([X1,X2])
# u = y[0][0,:,:,0]
# v = y[0][0,:,:,1]
# tu = Y[0,:,:,0]
# tv = Y[0,:,:,1]
# epe = epe_flow_error(tu, tv, u, v)
# epe
# """


"""
y=model1.predict([X1,X2])
y1 = flow_mag(y[0])
plt.imsave("testy",y1[0])
plt.imsave("testX1",X1[0])
plt.imsave("testX2",X2[0])

"""
# np.savez('sample_model1', flow =y1)

####--------------viz-------------------

"""
imgen=ImageSequence_new()
[X1,X2],Y = imgen.__getitem__()
y=model1.predict([X1,X2])
y1 = flow_mag(y[0])
%matplotlib
plt.figure("flow_pred")
plt.imshow(y1[0])
plt.figure("scene")
plt.imshow(X1[0])

"""

"""
%matplotlib
plt.imshow(y1[0])
plt.imsave("l3_new_10",y1[0])
model.save_weights("data/sampleU12.h5")
"""


"""
###-------------------ground_truth_generator_fixed----------------

flow_path = "/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/flow/alley_2/frame_0001.flo"
fl = read_flow(flow_path)
y1 = flow_mag(y)
plt.imsave("ground_truth",y1)
%matplotlib
plt.figure("flow_pred")
plt.imshow(y1)
"""


########-----------------------LOSS------------##########

##------------------------flow_consistency---------------------
## we can't run flow consistency loss alone because it will try to make the flow at points equal to zero.
## This may also result in overall decrease in magnitude of points.
## 

##---------------------new_flow_consistency---------------------
## This one does not work alone as it only wants the flow1 and flow to be invertible of each other
## It does not want the flow to be able to reconstruct the image2 from inage1
## On using this alone with smoothness gives very poor value



##---------------------grad_loss-----------------
## On combining mse and mse_grad, we idntify flow more than mse but magnitude is lower than mse,
## but this may perform better as mse is expected to underperform at whole dataset.
##

##-------------------only_mse---------------------
## A lot of grains are visible, not all positions of flow are identified.
## Error values reach 0.0015 at the end of 1000 epochs.
## On comparing with mse_grad, only mse is better in atleast capturing change in magnitude,
## but structural features, which are not moving are still preserved.

##------------------only_mse_grad------------------
## coarse grained pixels, values reach a max of 2. error at 0.06 at the end of 1000 epochs.
## mse_grad only identifies changes, magnitude of the whole image somewhat remains the same.
## On camparison with only mse, 
## this identifies flow changes better but is not able to assign suitable magnitude to those points.

##------------------only_ssim-----------------------
## Identifies flow better than rest but identifies other structural features as well
##

##------------------ssim_and_edge_aware_sm----------
## error ends at 0.03, solves the problem of ssim of identifying structural features
## gives noise at the places of occlusions. earlier the ssim gave discontinous values at places which is no more present
## smoothness added.

##------------------grad_and_edge_aware_sm---------------
## this gives better result in terms of boundaries than grad alone,
## the magnitude of flow values is lower than what we got from ssim and grad_sm

###-----------------edge_aware_sm_old---------------------
## This gives nice outline to  the flow values, magnitude is very close to the real values
## but the problems is that the body part is not is getting smoothened a lot
## and all the bodies look hollow, 
## we may train two networks with edge_aware and not edge aware sm and then 
## take element wise max in both channels so that the max of both body from not edge aware sm
## and boundaries from the edge aware sm could pass through, in a way it is similar to masking though

###-------------------edge_aware_sm_new---------------------
## This is less dominant than the previous (edge_aware_sm_old), it needs more weight attached to it
## performs almost similar to that of (edge_aware_sm_old)

