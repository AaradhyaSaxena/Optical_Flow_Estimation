import cv2
import os
import numpy as np
from natsort import natsorted
import gc
import  matplotlib.pyplot as plt
import sys
import pexpect
from subprocess import call
import shutil
from  scipy.misc import  imsave,imread
import scipy.io as sio




#read pfm file which we get from flow net 
def readFlow(name):
    if name.endswith('.pfm') or name.endswith('.PFM'):
        return readPFM(name)[0][:,:,0:2]

    f = open(name, 'rb')

    header = f.read(4)
    if header.decode("utf-8") != 'PIEH':
        raise Exception('Flow file header does not contain PIEH')

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

    flow=flow.astype(np.float32)
    flow[flow>10000]=0
    return flow


def get_images(dirx):
    l=os.listdir(dirx)
    frame1=imread(dirx+"/"+'frame10.png')
    frame2=imread(dirx+"/"+'frame11.png')
    return [frame1,frame2]
def get_flow(diry):
    l=os.listdir(diry)
    flow=readFlow(diry+"/"+l[0])
    return flow


def get_data():
	DATA_DIR_X ="/home/ranjan/work/flow/data/middlebury_color/other-data/"
	DATA_DIR_GT ="/home/ranjan/work/flow/data/middlebury_color/other-gt-flow/"
	d=os.listdir(DATA_DIR_GT)
	X=[]
	Y=[]
	for i in range(len(d)):
	    dirx=DATA_DIR_X+d[i]
	    diry=DATA_DIR_GT+d[i]
	    #print os.listdir(dirx),os.listdir(diry)
	    x_temp=get_images(dirx)
	    y_temp=get_flow(diry)
	    X.append(x_temp)
	    Y.append(y_temp)

	X=np.array(X)
	Y=np.array(Y)

	return X,Y


def gen_train_data(shape=(100,100),STRIDE=(50,50)):
    X,Y=get_data()

    PLIST=[] #patch list of 2 consicutive patch
    FLIST=[] #flow list
    for i in range(Y.shape[0]):
	idx=np.r_[:Y[i].shape[0]-shape[0]:STRIDE[0]]
	idy=np.r_[:Y[i].shape[1]-shape[1]:STRIDE[1]]
	for idx1 in idx:
	    for idy1 in idy:    
                patch1=X[i][0][idx1:idx1+shape[0],idy1:idy1+shape[1],:]		
                patch2=X[i][1][idx1:idx1+shape[0],idy1:idy1+shape[1],:]
	        #print patch1.shape,patch2.shape	
		flow=Y[i][idx1:idx1+shape[0],idy1:idy1+shape[1],:]

		
   		PLIST.append([patch1,patch2])
   		FLIST.append(flow)

    PLIST=np.array(PLIST)
    FLIST=np.array(FLIST)



