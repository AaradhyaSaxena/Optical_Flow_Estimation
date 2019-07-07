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




def return_grid_np(grid_shape=(436,1024)):
    grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1,  1]),
        [1, grid_shape[1], 1 ])
    grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1,  1]),
        [grid_shape[0], 1, 1 ])
    grid = np.concatenate([grid_x, grid_y],axis=-1)      #grid shape=(13, 13, 2)
    grid=grid.astype("float32")
    return grid
#def return_grid(y_pred,grid_shape):
def return_grid(grid_shape=(436,1024),batch_size=4):
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1,  1]),
        [1, grid_shape[1], 1 ])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1,  1]),
        [grid_shape[0], 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid=tf.reshape(grid,(1,)+grid_shape+(2,))
    grid=K.tile(grid,[batch_size,1,1,1])
    grid = K.cast(grid, dtype="int32")
    #grid = K.cast(grid, K.dtype(y_pred))
    return grid







###-----------------compile_model---------------------------
def compile_model(model1,lambda_smoothness=0.01,lambda_ssim=5,lambda_mse=0.0002,lambda_flow=0.0001,occ_punishment=0.0002):

    i1=model1.inputs[0]
    i2=model1.inputs[1]
    o1=model1.outputs[0]
    o2=model1.outputs[1]

    #------flow consistiancy loss------
    grid=return_grid(batch_size=2)
    I=tf.placeholder(tf.float32,shape=(2,436,1024,3))
    I1=tf.batch_gather(I,grid)
    f=K.function(inputs=[I],outputs=[I1])





    ###--------Gradient_smoothness--------------------------------------------
    ux,uy=grad_xy(o1[:,:,:,:1])
    vx,vy=grad_xy(o1[:,:,:,1:2])
    sm_loss_o1 = K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy))
    ux,uy=grad_xy(o2[:,:,:,:1])
    vx,vy=grad_xy(o2[:,:,:,1:2])
    sm_loss_o2 = K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy))
    sm_loss = (sm_loss_o1 + sm_loss_o2) 

    ### Reconstruction_loss_ssim_(occlusion_not considered)
    i2_rec=image_warp(i1,o1)
    i1_rec=image_warp(i2,o2)
    re_loss1=DSSIMObjective(kernel_size=50)(i2,i2_rec)
    re_loss2=DSSIMObjective(kernel_size=50)(i1,i1_rec)
    re_loss_ssim = (re_loss1 + re_loss2)

    total_loss = sm_loss + occ_loss + occ_punish + re_loss_ssim + flow_loss 

    #### model = Model(inputs=[i1,i2], outputs=[o1])
    model1.add_loss(total_loss)

    model1.compile(optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0))

    return model1
###--------------------compile---------------------------------

model1, model2 = combined()
model1 = compile_model(model1)

#-------------------DATA-------------------

imgen=ImageSequence_fixed()
[X1,X2],Y = imgen.__getitem__()


#-------------------------Training-----------
model1.load_weights('../data/unflow1.h5')
#model1.fit_generator(imgen,epochs=2000)
model1.fit([X1,X2],None,epochs=1000)
#model1.save_weights("../data/unflow1_batch.h5")





###_--------------------test------------------

# imgen=ImageSequence_new()
X1,Y = imgen.__getitem__()

# y=model.predict([X1,X2])
# y1 = flow_mag(y[0])
# plt.imsave("testy",y1[0])
# plt.imsave("testX",X1[0])

# np.savez('sample_model1', flow =y1)

####--------------viz-------------------

"""
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
plt.imsave("temp1",y1[0])
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
