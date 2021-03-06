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
# from model import *


####---------------LOSS--------------------
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

    n_channel=K.get_variable_shape(image)[-1]
    g_kernel=np.tile(g_kernel,[1,1,n_channel,n_channel])

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

    fxx,fxy=grad_xy(y_pred[:,:,:,:1])
    fyx,fyy=grad_xy(y_pred[:,:,:,1:2])

    loss_grad = (K.mean(K.abs(fxx*fxx)+ K.abs(fyy*fyy)+ K.abs(fyx*fyx)+ K.abs(fyy*fyy)))
    # loss_grad = (K.mean(K.square(fxx - fxx_pred))*(1/4) + K.mean(K.square(fxy - fxy_pred))*(1/4)
    # 			+ K.mean(K.square(fyx - fyx_pred))*(1/4) + K.mean(K.square(fyy - fyy_pred))*(1/4))

    loss_mse = K.mean(K.square(y_pred - y_true))

    # loss_ssim = DSSIMObjective(kernel_size=20)(y_true,y_pred)

    total_loss = loss_mse + loss_grad #+ loss_ssim

    return total_loss


### adjust the constant to change the smoothness of the image
##############################################################
def c_grad1(y_true,y_pred):

    const = 0.001

    fxx,fxy=grad_xy(y_pred[:,:,:,:1])
    fyx,fyy=grad_xy(y_pred[:,:,:,1:2])

    loss_grad = (K.mean(K.abs(fxx*fxx)+ K.abs(fyy*fyy)+ K.abs(fyx*fyx)+ K.abs(fyy*fyy)))

    loss_mse = K.mean(K.square(y_pred - y_true))

    total_loss = loss_mse + const*loss_grad

    return total_loss 


def c_grad_rec(y_true,y_pred,model):

    lambda1 = 0.005

    input1_rec=image_warp(model.inputs[0],model.outputs[0])
    # input0_rec=image_warp(model.inputs[1],-model.outputs[0],num_batch = 2)

    ux,uy=grad_xy(model.outputs[0][:,:,:,:1])
    vx,vy=grad_xy(model.outputs[0][:,:,:,1:2])
    sm_loss=lambda1*(K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy)))

    re_loss_mse = K.mean(K.square(model.inputs[1] - input1_rec))

    loss_mse = K.mean(K.square(y_pred - y_true))

    total_loss = lambda1*sm_loss+re_loss_mse + loss_mse

    return total_loss

def c_recMse_grad(model):

    def loss(y_true,y_pred):

        return c_grad_rec(y_true,y_pred,model)

    return loss



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



###############################################
##---------------------------------------------

def flow_mag(y):

    # f = np.zeros((10,436,1024))
    # f[:,:,:] = np.sqrt(np.square(y[:,:,:,0])+ np.square(y[:,:,:,1]))
    f = np.sqrt(np.square(y[:,:,:,0])+ np.square(y[:,:,:,1]))
    
    return f
###--------------------------------------------
def flow_mag_tensor(y):

    # f = np.zeros((10,436,1024))
    # f[:,:,:] = np.sqrt(np.square(y[:,:,:,0])+ np.square(y[:,:,:,1]))
    f = tf.sqrt(tf.square(y[:,:,:,0])+ tf.square(y[:,:,:,1]))
    f = tf.expand_dims(f,-1)
    
    return f
##############################################

def read_flow(filename):
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)

    return flow
###--------------------------    
def read_flo_file(filename):
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print("Reading %d x %d flow file in .flo format" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


##########################################################
###-----------------evaluation----------------------------

###-----------------mpi_epe-------------------------------

UNKNOWN_FLOW_THRESH = 1e7

def epe_flow_error(tu, tv, u, v):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]
    index_su = su[tuple(ind2)]
    index_sv = sv[tuple(ind2)]
    an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
    un = index_su * an
    vn = index_sv * an

    index_stu = stu[tuple(ind2)]
    index_stv = stv[tuple(ind2)]
    tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
    tun = index_stu * tn
    tvn = index_stv * tn

    '''
    angle = un * tun + vn * tvn + (an * tn)
    index = [angle == 1.0]
    angle[index] = 0.999
    ang = np.arccos(angle)
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    '''

    epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
    epe = epe[tuple(ind2)]
    mepe = np.mean(epe)

    return mepe

###-----------------------kitti_metric_F1-----------------------

def compute_Fl(flow_gt, flow_est, mask):
    # F1 measure
    err = tf.multiply(flow_gt - flow_est, mask)
    err_norm = tf.norm(err, axis=-1)
    
    flow_gt_norm = tf.maximum(tf.norm(flow_gt, axis=-1), 1e-12)
    F1_logic = tf.logical_and(err_norm > 3, tf.divide(err_norm, flow_gt_norm) > 0.05)
    F1_logic = tf.cast(tf.logical_and(tf.expand_dims(F1_logic, -1), mask > 0), tf.float32)
    F1 = tf.reduce_sum(F1_logic) / (tf.reduce_sum(mask) + 1e-6)
    return F1


def return_compute_Fl():

    F_gt=Input(shape=(436,1024,2))
    fl_est=Input(shape=(436,1024,2))
    M=Input(shape=(436,1024,2))
    out1=compute_Fl(F_gt, fl_est, M)
    com_FL=K.function(inputs=[F_gt,fl_est,M],outputs=[out1])

    return com_FL
###########################################







































###------dirty_hard_coded_stuff---------------
###-------------------------------------------

def max_channels32(inputs, num_units = 32, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    
    return outputs
###------------------
def max_channels16(inputs, num_units = 16, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    
    return outputs
###------------------
def max_channels8(inputs, num_units = 8, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    
    return outputs
###------------------
def max_channels4(inputs, num_units = 4, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    
    return outputs
###------------------
def max_channels2(inputs, num_units = 2, axis=None):
    shape = inputs.get_shape().as_list()
    if shape[0] is None:
        shape[0] = -1
    if axis is None:  # Assume that channel is the last dimension
        axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(num_channels, num_units))
    shape[axis] = num_units
    shape += [num_channels // num_units]
    outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
    
    return outputs
###------------------
####-----------------------------------------

