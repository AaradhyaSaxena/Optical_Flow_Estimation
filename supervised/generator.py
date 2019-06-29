# coding=utf-8
import skimage.transform
from keras.utils import Sequence
import numpy as np
import cv2
import glob
import pandas as pd
import os
from scipy import misc
from skimage.transform import resize

import png
import matplotlib.colors as cl
import matplotlib.pyplot as plt
from PIL import Image

########################-----------------------------------------------
###### this gives a defined batch as o/p not a random batch ###########

class ImageSequence_fixed(Sequence):
    def __init__(self,  batch_size=12, input_size=(436, 1024),f_gap=1):
        self.image_seq_path="/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/"
        self.flow_seq_path="/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/flow/"

        self.input_shape=input_size

        self.im_dirs=os.listdir(self.image_seq_path)
        self.fl_dirs=os.listdir(self.flow_seq_path)

        self.im_dirs.sort()
        self.fl_dirs.sort()

        self.len_im_dirs=len(self.im_dirs)
        self.len_fl_dirs=len(self.fl_dirs)

        self.batch_size = batch_size
        self.epoch = 0
        self.f_gap=f_gap
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        # self.im_SHAPE_C=3
        # self.fl_SHAPE_C=2


    def __len__(self):
        return (180)

    def read_image(self,file_path):
        Img=misc.imread(file_path)
        Img=resize(Img,(self.SHAPE_Y,self.SHAPE_X,))#self.im_SHAPE_C))
        return Img

########-----------------------------------------

    def read_flow(self,filename):
        if filename.endswith('.flo'):
            flow = self.read_flo_file(filename)

        return flow

    def read_flo_file(self,filename):
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

###########--------------------------------------

    def __getitem__(self, idx = None):

        x_batch = []
        flow_batch = []
        c = 0
        ###############################################
        # d_idx = np.random.randint(0,self.len_im_dirs)
        d_idx = 1 #2             #######################
        ################################################

        im_path = self.image_seq_path+self.im_dirs[d_idx]+"/"
        fl_path = self.flow_seq_path+self.fl_dirs[d_idx]+"/"

        self.im_frames = os.listdir(im_path)
        self.im_frames.sort()
        num_im_frames = len(self.im_frames)

        self.fl_frames = os.listdir(fl_path)
        self.fl_frames.sort()
        num_fl_frames = len(self.fl_frames)

        while(c<self.batch_size):
            
            #######################################################                  
            # s_idx = np.random.randint(0,num_im_frames-self.f_gap)
            s_idx = 0 #3               ############################
            #######################################################

            I1_file = self.im_frames[s_idx]  
            I2_file = self.im_frames[s_idx+self.f_gap]
            flow_file = self.fl_frames[s_idx]

            c = c+1
            I1=self.read_image(im_path+I1_file)
            I2=self.read_image(im_path+I2_file)
            flo = self.read_flow(fl_path+flow_file)

            #I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,1))                
            #I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,1))               
            # I1=np.expand_dims(I1,axis=-1) 
            # I2=np.expand_dims(I2,axis=-1) 
            x_batch.append([I1,I2])
            flow_batch.append(flo)

        x_batch = np.array(x_batch, np.float32)
        flow_batch = np.array(flow_batch) 

        #x_batch=np.zeros((4,2,240, 360, 1))
        #return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],y_batch)
        x_batch1=x_batch[:,0,:self.SHAPE_Y,:self.SHAPE_X,:]
        x_batch2=x_batch[:,1,:self.SHAPE_Y,:self.SHAPE_X,:]


        # y_flow=np.random.random((x_batch1.shape[:-1]+(2,)))
        return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],flow_batch)
        #return (x_batch1,x_batch2,None)
        #return x_batch

    def on_epoch_end(self):
        self.epoch += 1


###########----------------------------------------------------------------
########### this was the original function which did not contain flow #####

# class ImageSequence(Sequence):
#     def __init__(self,  batch_size=4, input_size=(436, 1024,3),f_gap=1):
#         self.image_seq_path="/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/"
#         self.input_shape=input_size
#         self.dirs=os.listdir(self.image_seq_path)
#         self.dirs.sort()
#         self.len_dirs=len(self.dirs)
#         self.batch_size = batch_size
#         self.epoch = 0
#         self.f_gap=f_gap
#         self.SHAPE_Y=self.input_shape[0]
#         self.SHAPE_X=self.input_shape[1]


#     def __len__(self):                     
#         return (180)

#     def read_image(self,file_path):
#         Img=misc.imread(file_path)
#         Img=resize(Img,(self.SHAPE_Y,self.SHAPE_X))
#         return Img

#     def __getitem__(self, idx = None):

#         x_batch = []
#         c = 0
#         d_idx = np.random.randint(0,self.len_dirs) 
#         path = self.image_seq_path+self.dirs[d_idx]+"/"
#         self.frames = os.listdir(path)
#         self.frames.sort()
#         num_frames = len(self.frames)

#         while(c<self.batch_size):
                              
#             s_idx = np.random.randint(0,num_frames-self.f_gap)  
                             
#             I1_file = self.frames[s_idx]  
#             I2_file = self.frames[s_idx+self.f_gap]
#             c = c+1
#             I1=self.read_image(path+I1_file)
#             I2=self.read_image(path+I2_file)
#             #I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,1))                
#             #I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,1))               
#             # I1=np.expand_dims(I1,axis=-1) 
#             # I2=np.expand_dims(I2,axis=-1) 
#             x_batch.append([I1,I2])

#         x_batch = np.array(x_batch, np.float32) 
#         #x_batch=np.zeros((4,2,240, 360, 1))
#         #return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],y_batch)
#         x_batch1=x_batch[:,0,:self.SHAPE_Y,:self.SHAPE_X,:]
#         x_batch2=x_batch[:,1,:self.SHAPE_Y,:self.SHAPE_X,:]

#         #return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],None)
#         return (x_batch1,x_batch2,None)
#         #return x_batch

#     def on_epoch_end(self):
#         self.epoch += 1


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#---------------------------------------------------------------------
###### this will provide random batches ##############################

class ImageSequence_new(Sequence):
    def __init__(self,  batch_size=10, input_size=(436, 1024),f_gap=1):
        self.image_seq_path="/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/"
        self.flow_seq_path="/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/flow/"

        self.input_shape=input_size

        self.im_dirs=os.listdir(self.image_seq_path)
        self.fl_dirs=os.listdir(self.flow_seq_path)

        self.im_dirs.sort()
        self.fl_dirs.sort()

        self.len_im_dirs=len(self.im_dirs)
        self.len_fl_dirs=len(self.fl_dirs)

        self.batch_size = batch_size
        self.epoch = 0
        self.f_gap=f_gap
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        # self.im_SHAPE_C=3
        # self.fl_SHAPE_C=2


    def __len__(self):
        return (180)

    def read_image(self,file_path):
        Img=misc.imread(file_path)
        Img=resize(Img,(self.SHAPE_Y,self.SHAPE_X,))#self.im_SHAPE_C))
        return Img

########-----------------------------------------

    def read_flow(self,filename):
        if filename.endswith('.flo'):
            flow = self.read_flo_file(filename)

        return flow

    def read_flo_file(self,filename):
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

###########--------------------------------------

    def __getitem__(self, idx = None):

        x_batch = []
        flow_batch = []
        c = 0
        d_idx = np.random.randint(0,self.len_im_dirs) 

        im_path = self.image_seq_path+self.im_dirs[d_idx]+"/"
        fl_path = self.flow_seq_path+self.fl_dirs[d_idx]+"/"

        self.im_frames = os.listdir(im_path)
        self.im_frames.sort()
        num_im_frames = len(self.im_frames)

        self.fl_frames = os.listdir(fl_path)
        self.fl_frames.sort()
        num_fl_frames = len(self.fl_frames)

        while(c<self.batch_size):
                              
            s_idx = np.random.randint(0,num_im_frames-self.f_gap)

            I1_file = self.im_frames[s_idx]  
            I2_file = self.im_frames[s_idx+self.f_gap]
            flow_file = self.fl_frames[s_idx]

            c = c+1
            I1=self.read_image(im_path+I1_file)
            I2=self.read_image(im_path+I2_file)
            flo = self.read_flow(fl_path+flow_file)

            #I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,1))                
            #I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,1))               
            # I1=np.expand_dims(I1,axis=-1) 
            # I2=np.expand_dims(I2,axis=-1) 
            x_batch.append([I1,I2])
            flow_batch.append(flo)

        x_batch = np.array(x_batch, np.float32)
        flow_batch = np.array(flow_batch) 

        #x_batch=np.zeros((4,2,240, 360, 1))
        #return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],y_batch)
        x_batch1=x_batch[:,0,:self.SHAPE_Y,:self.SHAPE_X,:]
        x_batch2=x_batch[:,1,:self.SHAPE_Y,:self.SHAPE_X,:]


        # y_flow=np.random.random((x_batch1.shape[:-1]+(2,)))
        return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],flow_batch)
        #return (x_batch1,x_batch2,None)
        #return x_batch

    def on_epoch_end(self):
        self.epoch += 1






# from generator import *
# A=ImageSequence_new()

# [x1,x2],fl = A.__getitem__()

