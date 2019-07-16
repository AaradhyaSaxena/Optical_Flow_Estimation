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
from unflow import *


model1, model2 = combined()
model1 = compile_model(model1)
model1.load_weights("../data/old_sm_loss/unflow_001sm_imgrecon_grad_14.h5")

#-------------------DATA-------------------

# imgen=ImageSequence_fixed()
# [X1,X2],Y = imgen.__getitem__()

# imgen=ImageSequence_new()
# [X1,X2],Y = imgen.__getitem__()

imgen=ImageSequence_eval()
[X1,X2],Y,mask,occ = imgen.__getitem__()



###_--------------------test------------------


imgen=ImageSequence_eval()
[X1,X2],Y,mask,occ = imgen.__getitem__()

###-----------EPE_MPI_sintel--------------------
y = model1.predict([X1,X2])
u = y[0][0,:,:,0]
v = y[0][0,:,:,1]
tu = Y[0,:,:,0]
tv = Y[0,:,:,1]

# u = y[0][:,:,:,0]
# v = y[0][:,:,:,1]
# tu = Y[:,:,:,0]
# tv = Y[:,:,:,1]

epe = epe_flow_error(tu, tv, u, v)
print(epe)
###-------------F1_KITTI------------------------
mask = np.ones((2,436,1024,2), dtype=np.float32)

function = return_compute_Fl()
f1= function([Y, y[0], mask])
print(f1)
###---------------------------------------------


