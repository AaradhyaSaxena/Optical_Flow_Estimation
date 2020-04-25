import numpy as np
# from PIL import Image

from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from numpy import linalg as LA




#flow shape (#num_sample,416,416,2) returns points1 and points2 of size (num_sample,416*416,2)
def cor_from_flow(flow):
    shape_x=flow.shape[-2] 
    shape_y=flow.shape[-3] 
    tx,ty= np.mgrid[0:shape_y, 0:shape_x]
    temp_cord1=np.concatenate([tx[:,:,np.newaxis],ty[:,:,np.newaxis]],axis=-1)

    cord1=np.zeros_like(flow)
    cord1[:]=temp_cord1
    cord2=cord1+flow	

    c1=np.reshape(cord1,(cord1.shape[0],cord1.shape[1]*cord1.shape[2],cord1.shape[3]))    
    cord1=np.reshape(cord1,(cord1.shape[0],cord1.shape[1]*cord1.shape[2],cord1.shape[3]))
    cord2=np.reshape(cord2,(cord2.shape[0],cord2.shape[1]*cord2.shape[2],cord2.shape[3]))

    #cord1=cord1.reshape(cord1.shape[0],cord1.shape[1],cord1.shape[2],cord1.shape[3])    #to revert
    return cord1,cord2


#return essential matrix of size=(3,3) from two point correspondance  points1,points2 of shape=(#num_points,2)  and camera parameters K1 and K2 of same size (3,3)    
def  compute_essential_matrix(points1,points2,K1=np.array([[1,0,0],[0,1,0],[0,0,1]]),K2=None):
    #check both camera parameterare same or not
    if(K2==None):
	K2=K1
    
    #convert both points into homogeneous coordinates 
    t=np.ones((points1.shape[0],1))
    points1=np.append(points1,t,axis=-1)
    points2=np.append(points2,t,axis=-1)


    #normalze points by multiplying with camera matrix
    p1=np.matmul(np.linalg.inv(K1),points1.T).T 
    p2=np.matmul(np.linalg.inv(K2),points2.T).T 

    #E=[e11,e12,e13,e21,e22,e23,e31,e32,e33]  coeff of	E is  A=[x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,1]
    x1=p1[:,0]
    y1=p1[:,1]
    x2=p2[:,0]
    y2=p2[:,1]
    A=np.array([x2*x1,x2*y1,x2,y2*x1,y2*y1,y2,x1,y1,np.ones_like(x1)])    #shape of A=(9,# points)
    A=A.T  #shape of A=(#points,9)
     


    #mse fit
    M = np.matmul(A.T,A)
    U, S, V = np.linalg.svd(M)
    E = V[-1]
    E = E.reshape(3, 3)


    #refine Essential Matrix
    u,s,v=np.linalg.svd(E)
    sig=(s[0]+s[1])/2
    s[0:2]=sig
    s[2]=0
    E_new=np.matmul(np.matmul(u,np.diag(s)),v)    




    return E_new


#return R(3,3)  and T(3,) from essential matrix(3,3)
def return_RT_from_E(E):
    u,s,v=np.linalg.svd(E)

    R = np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype="float32")
    ur=np.matmul(u,R)

    R=np.matmul(ur,v)
    Tx=np.matmul(np.matmul(ur,np.diag(s)),u.T)

    T=np.array([Tx[2][1],Tx[0][2],Tx[1][0]])

    return R,T



#find all depth of correcponding points from and R and T  and  correspoints
#input=points1(#points,2) points2(#points,2) R(3x3) T(3x1)   outputs=depth_scale(#points+1,1)
def find_depth(points1,points2,R,T):

    #make all the points homogeneous 
    t=np.ones((points1.shape[0],1))
    p1=np.append(points1,t,axis=-1)
    p2=np.append(points2,t,axis=-1)

    p2x=np.zeros((p1.shape[0],3,3))
    p2x[:,0,1]=-p2[:,2]
    p2x[:,0,2]=p2[:,1]
    p2x[:,1,0]=p2[:,2]
    p2x[:,1,2]=-p2[:,0]
    p2x[:,2,0]=-p2[:,1]
    p2x[:,2,1]=p2[:,0]


    #p1=p1[:,:,np.newaxis]
    t1=np.matmul(p2x,R)
 
TAG_FLOAT = 202021.25
   
def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M #,N




