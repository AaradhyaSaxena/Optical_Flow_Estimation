# coding=utf-8
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, concatenate,Flatten,Dense
from keras.layers import Lambda,Reshape,LocallyConnected2D,SeparableConv1D,LocallyConnected1D,LeakyReLU
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] ='-1'
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
# from init import *

batch_size=4

SHAPE_Y=416
SHAPE_X=416

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

def unet_down_1(filter_count, inputs, activation='linear', pool=(2, 2), n_layers=3):
    down = inputs
    for i in range(n_layers):
        down = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(down)
        down = BatchNormalization()(down)
        if pool is not None:
            x = MaxPooling2D(pool, strides=pool)(down)
        else:
            x = down
    return (x, down)

def unet_up_1(filter_count, inputs, down_link, activation='linear', n_layers=3):
    reduced = Conv2D(filter_count, (1, 1), padding='same', activation=activation)(inputs)
    up = UpSampling2D((1, 1))(reduced)
    up = BatchNormalization()(up)
    link = Conv2D(filter_count, (1, 1), padding='same', activation=activation)(down_link)
    link = BatchNormalization()(link)
    up = Add()([up,link])
    for i in range(n_layers):
        up = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(up)
        up = BatchNormalization()(up)
    return up

def div_into_patch_back(x):
    p_y=p_x=8
    patch_size = [1,p_y,p_x,1]
    stride_size=[1,p_y,p_x,1]
    patches = tf.extract_image_patches(x,patch_size, stride_size, [1, 1, 1, 1], 'VALID')


    x2=tf.depth_to_space(patches,p_x)
    return x2

def create_model(input_shape=(SHAPE_Y,SHAPE_X,1)):
    n_layers_down = [2,2,2]
    n_layers_up = [2,2,2]
    n_filters_down = [5,8,10]
    n_filters_up = [4,8,10]
    kernels_up=[(3,3),(3,3),(3,3),(3,3)]
    n_filters_center=6
    n_layers_center=4


    print('n_filters_down:%s  n_layers_down:%s'%(str(n_filters_down),str(n_layers_down)))
    print('n_filters_center:%d  n_layers_center:%d'%(n_filters_center, n_layers_center))
    print('n_filters_up:%s  n_layers_up:%s'%(str(n_filters_up),str(n_layers_up)))
    activation='relu'
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
    inputs=Concatenate()([input1,input2])   

    x = inputs
    x = BatchNormalization()(x)
    xbn = x
    depth = 0
    back_links = []

    for n_filters in n_filters_down:
        n_layers = n_layers_down[depth]
        x, down_link = unet_down_1(n_filters, x, activation=activation,pool=None, n_layers=n_layers)
        back_links.append(down_link)
        depth += 1

    
    #x2=Lambda(div_into_patch_back,output_shape=K.get_variable_shape(x)[1:])(x)
    x2=x


    center, _ = unet_down_1(n_filters_center, x2, activation='relu', pool=None, n_layers=n_layers_center)

    # center
    x3 = center
    while depth > 0:
        depth -= 1
        link = back_links.pop()
        n_filters = n_filters_up[depth]
        n_layers = n_layers_up[depth]
        x3 = unet_up_1(n_filters, x3, link, activation='relu', n_layers=n_layers)
        #if depth <= 1:
            #x1 = Dropout(0.25)(x1)
    
    #x1 = concatenate([x1,xbn])
    x3 = Conv2D(4, (3, 3), padding='same', activation=None)(x3)
    x3 = BatchNormalization()(x3)
    F = Conv2D(2, (1, 1), activation=None)(x3)

    model = Model(inputs=[input1,input2], outputs=[F])
    return model

#compile with new loss
def compile_model_new(model,b_size,lambda1=0.02):
    """
    session=tf.Session()
    session.run(tf.global_variables_initializer())
    
    var = [v for v in tf.trainable_variables() if v.name == "sig1:0"][0]
    session.run(var)
    """
    #s1 = tf.get_variable("sig1", shape=(1,), trainable=True,initializer=tf.constant([0.3]))
    #s2 = tf.get_variable("sig2", shape=(1,), trainable=True,initializer=tf.constant([0.7]))
    #s1 = tf.get_variable("sig1",  trainable=True,initializer=tf.constant([0.3]))
    #s2 = tf.get_variable("sig2",  trainable=True,initializer=tf.constant([0.7]))
    #s1_2=s1*s1
    #s2_2=s1*s1
    
    input1_rec=image_warp(model.inputs[0],model.outputs[0],num_batch=b_size)
    input0_rec=image_warp(model.inputs[1],-model.outputs[0],num_batch=b_size)

   
    ux,uy=grad_xy(model.outputs[0][:,:,:,:1])
    vx,vy=grad_xy(model.outputs[0][:,:,:,1:2])
    sm_loss=lambda1*(K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy)))

    re_loss=DSSIMObjective(kernel_size=50)(model.inputs[1],input1_rec)
    
    #total_loss=(1/s1_2)*re_loss+(1/s2_2)*sm_loss+K.log(s1_2)+K.log(s2_2)
    total_loss=lambda1*sm_loss+re_loss

    model.add_loss(total_loss)

    model.compile(optimizer='rmsprop')
   	
    return model

#input_image shape  (sample,SHAPE_Y,SHAPE_X,1)    F->(sample,SHAPE_Y,SHAPE_X,F)
def recons_img(input_image,F):

    inp1 = K.placeholder(shape=input_image.shape)
    F_in = K.placeholder(shape=F.shape)
    input1_rec=image_warp(inp1,F,num_batch=F.shape[0])
    f1=K.function(inputs=[inp1,F_in],outputs=[input1_rec])
   
    out=f1([input_image,F])
    return out

