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

class ImageSequence(Sequence):
    def __init__(self,  batch_size=4, input_size=(436, 1024),f_gap=1):
        self.image_seq_path="/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/"
        self.input_shape=input_size
        self.dirs=os.listdir(self.image_seq_path)
        self.dirs.sort()
        self.len_dirs=len(self.dirs)
        self.batch_size = batch_size
        self.epoch = 0
        self.f_gap=f_gap
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]


    def __len__(self):                     
        return (180)

    def read_image(self,file_path):
        Img=misc.imread(file_path)
        Img=resize(Img,(self.SHAPE_Y,self.SHAPE_X))
        return Img

    def __getitem__(self, idx = np.random.randint(0,self.len_dirs)):

        x_batch = []
        c = 0
        d_idx = np.random.randint(0,self.len_dirs) 
        path = self.image_seq_path+self.dirs[d_idx]+"/"
        self.frames = os.listdir(path)
        self.frames.sort()
        num_frames = len(self.frames)

        while(c<self.batch_size):
                              
            s_idx = np.random.randint(0,num_frames-self.f_gap)  
                             
            I1_file = self.frames[s_idx]  
            I2_file = self.frames[s_idx+self.f_gap]
            c = c+1
            I1=self.read_image(path+I1_file)
            I2=self.read_image(path+I2_file)
            #I1=resize(I1,(self.SHAPE_Y,self.SHAPE_X,1))                
            #I2=resize(I2,(self.SHAPE_Y,self.SHAPE_X,1))               
            I1=np.expand_dims(I1,axis=-1) 
            I2=np.expand_dims(I2,axis=-1) 
            x_batch.append([I1,I2])

        x_batch = np.array(x_batch, np.float32) 
        #x_batch=np.zeros((4,2,240, 360, 1))
        #return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],y_batch)
        x_batch1=x_batch[:,0,:self.SHAPE_Y,:self.SHAPE_X,:]
        x_batch2=x_batch[:,1,:self.SHAPE_Y,:self.SHAPE_X,:]

        #return ([x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]],None)
        return ([x_batch1,x_batch2],None)
        #return x_batch

    def on_epoch_end(self):
        self.epoch += 1



# from generator import *
A=ImageSequence()
x1, _ = A.__getitem__()

