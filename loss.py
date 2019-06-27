from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, concatenate,Flatten,Dense
from keras.layers import Lambda,Reshape,LocallyConnected2D,SeparableConv1D,LocallyConnected1D,LeakyReLU
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
from model import *
from loss import *


####-------------Two-Arguments-------------

def c_mse(y_true, y_pred):

    loss = K.mean(K.square(y_pred - y_true))
    return loss

def c_mseAbs(y_true,y_pred):

    loss_mse = K.mean(K.square(y_pred - y_true))
    loss_abs = K.mean(K.abs(y_pred - y_true))
    loss = loss_abs + loss_mse
    return loss

def cconv(image, g_kernel):
    g_kernel=g_kernel[:,:,np.newaxis,np.newaxis]
    d_kernel=tf.Variable(g_kernel)
    #image=tf.Variable(image)
    out=tf.nn.conv2d(image,d_kernel,strides=[1, 1, 1, 1], padding='VALID')
    #conv = K.function(inputs=[M],outputs=[out])

    return out

# u shape=(?, 240, 368, 1)
def grad_xy(y):
    pw_x = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]).astype(np.float32)
    y_x = cconv(y, pw_x)
    pw_y = np.array([[-1,-1,-1],[0,0,0],[1,1,1]]).astype(np.float32)
    y_y = cconv(y, pw_y)
   
    return (y_x,y_y)

def c_grad(y_true,y_pred):

    fxx,fxy=grad_xy(y_true[:,:,:,:1])
    fyx,fyy=grad_xy(y_true[:,:,:,1:2])

    fxx_pred,fxy_pred=grad_xy(y_pred[:,:,:,:1])
    fyx_pred,fyy_pred=grad_xy(y_pred[:,:,:,1:2])

    loss_grad = (K.mean(K.square(fxx - fxx_pred))*(1/4) + K.mean(K.square(fxy - fxy_pred))*(1/4)
    			+ K.mean(K.square(fyx - fyx_pred))*(1/4) + K.mean(K.square(fyy - fyy_pred))*(1/4))

    loss_mse = K.mean(K.square(y_pred - y_true))

    loss = loss_mse + loss_grad 

    return loss

def c_grad_ssim(y_true,y_pred):

    fxx,fxy=grad_xy(y_true[:,:,:,:1])
    fyx,fyy=grad_xy(y_true[:,:,:,1:2])

    fxx_pred,fxy_pred=grad_xy(y_pred[:,:,:,:1])
    fyx_pred,fyy_pred=grad_xy(y_pred[:,:,:,1:2])

    loss_grad = (K.mean(K.square(fxx - fxx_pred))*(1/4) + K.mean(K.square(fxy - fxy_pred))*(1/4)
    			+ K.mean(K.square(fyx - fyx_pred))*(1/4) + K.mean(K.square(fyy - fyy_pred))*(1/4))

    loss_mse = K.mean(K.square(y_pred - y_true))

    loss_ssim = DSSIMObjective(kernel_size=20)(y_true,y_pred)

    total_loss = loss_mse + loss_grad + loss_ssim

    return total_loss


####-----------Multiple_Argument------------


# def dice_coef(y_true, y_pred, smooth, thresh):
#     y_pred = y_pred > thresh
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)

#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# def dice_loss(smooth, thresh):
#   def dice(y_true, y_pred)
#     return -dice_coef(y_true, y_pred, smooth, thresh)
#   return dice


# model = my_model()
# # get the loss function
# model_dice = dice_loss(smooth=1e-5, thresh=0.5)
# # compile model
# model.compile(loss=model_dice)
