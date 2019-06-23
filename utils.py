import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from numpy import linalg as LA

from optical_flow_toolkit import *
from utils_flow import *

#############################################
##### -----------Utils_gamma-----------------

## (im1,im2,flow).shape=(416x416)
## num_corr= number of points taken to calculate depth
def return_corr_from_flow(im1, im2, flow):
    samples = im1.shape[0]
    image_height = im1.shape[1]
    image_width = im1.shape[2]
    flow_height = flow.shape[1] 
    flow_width = flow.shape[2]
    n = image_height * image_width

    essentialMatrix=[]
    index=[]
    tvecs=[]
    rvecs=[]
    depth=[]

    corr1 =[]
    corr2 =[]

    for i in range(samples):

        (iy, ix) = np.mgrid[0:image_height, 0:image_width]
        (fy, fx) = np.mgrid[0:flow_height, 0:flow_width]
        fx = fx.astype(np.float64)
        fy = fy.astype(np.float64)
        fx += flow[i,:,:,0]
        fy += flow[i,:,:,1]
        fx = np.minimum(np.maximum(fx, 0), flow_width)
        fy = np.minimum(np.maximum(fy, 0), flow_height)
        points = np.concatenate((ix.reshape(n,1), iy.reshape(n,1)), axis=1)
        xi = np.concatenate((fx.reshape(n, 1), fy.reshape(n,1)), axis=1)
        corr1.append(points)
        corr2.append(xi)

    corr1 = np.array(corr1)
    corr2 = np.array(corr2)

    Nn = len(corr1)
    for i in range(Nn):
        e = essential_matrix(corr1[i],corr2[i])
        essentialMatrix.append(e)
        tvecs.append(returnT_fromE(e))
        rvecs.append(returnR1_fromE(e))

    for i in range(samples):

        e = essentialMatrix[0]
        t = tvecs[0]
        r = rvecs[0]
        d = []
        q = 416
        for j in range(int(n/q)):
            print(i,j)
            selec1 = corr1[i,j*(q):(j+1)*q,:]
            selec2 = corr2[i,j*(q):(j+1)*q,:]
            d.append(return_depth(selec1,selec2,r,t))

        if(n%q!= 0):
            selec1 = corr1[i,(n/q)*q:(n/q)*q+(n%q),:]
            selec2 = corr2[i,(n/q)*q:(n/q)*q+(n%q),:]       
            d.append(return_depth(selec1,selec2,r,t))

        depth.append(d)

    depth = np.array(depth)
    depth = depth.reshape((samples,image_height,image_width+1,1))
    depth = (depth+1)/2 # pixel values: (-1,1)>>(0,1)

    return essentialMatrix,tvecs,rvecs,depth,corr1,corr2


def find_correspondance(img1,img2,n_pts=20):

    orb = cv2.ORB()
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
                    nlevels=8, fastThreshold=20,scaleFactor=1.2, 
                    WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
                    firstLevel=0,nfeatures=500)
    kp1 = orb.detect(img1,None)
    kp1, des1 = orb.compute(img1, kp1)
    kp2 = orb.detect(img2,None)
    kp2, des2 = orb.compute(img2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    good = []
    pts1 = []
    pts2 = []

    for m in matches:
        good.append([m])
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    
    return pts1, pts2

def return_depth(homo_pt1,homo_pt2,r1,t):

    # print(homo_pt1.shape)
    length = homo_pt1.shape[0]
    homo1 = np.ones((length,3))
    homo2 = np.ones((length,3))
    homo1[:,:2] = homo_pt1[:,:]
    homo2[:,:2] = homo_pt2[:,:]
    kk = np.load("data/parameters.npz") # k = kk['k_new']
    k = kk['k']
    homo_im1 = np.ones((length,3))
    homo_im2 = np.ones((length,3))
    homo_im1[:,:] = np.matmul(inv(k),homo1[:,:].T).T
    homo_im2[:,:] = np.matmul(inv(k),homo2[:,:].T).T

    # print(homo_im1.shape)
    a = np.matmul(homo_im2,r1)
    rot1 = np.matmul(a,homo_im1.T)
    trans1 = np.matmul(homo_im2,t.reshape((3,1)))
    
    A = np.hstack((rot1,trans1))
    ata = np.matmul(A.T,A)
    # print(homo_im1.shape)
    u, s, vh = np.linalg.svd(ata, full_matrices=True)
    # print(homo_im1.shape)
    Depth = vh[-1].reshape(length+1,1)

    return Depth

def find_corner_pts(img,n_pts=500):
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
                nlevels=8, fastThreshold=20,scaleFactor=1.2, 
                WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
                firstLevel=0,nfeatures= n_pts)
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)

    corn_arr = []
    for i in range(len(kp)):
        corn_arr.append(kp[i].pt)
    # corner = np.array(corn_arr)

    # corner_x = corner[:,0]
    # corner_y = corner[:,1]

    # return corner_x, corner_y
    return corn_arr


#shape of im2=im1=(500,2) i
def essential_matrix(im1, im2):

    length = im1.shape[0]
    homo = np.ones((2,length,3))
    homo_im = np.ones((2,length,3))
    # homo[:,:,0] = im1.T
    # homo[:,:,1] = im2.T
    homo[0,:,:2] = im1
    homo[1,:,:2] = im2
    
    kk = np.load("data/parameters.npz")
    # k = kk['k_new']
    k = kk['k']

    #shape of k=3x3 
    homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
    homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T
    
    xa=homo_im[0,:,0]
    xb=homo_im[1,:,0]
    ya=homo_im[0,:,1]
    yb=homo_im[1,:,1]
    c=np.ones((length,))
    A=[xb*xa,xb*ya,yb*xa,yb*ya,xb,yb,xa,ya,c] 
    A=np.array(A).T

    # A = np.hstack(((homo_im[1,:,0]*homo_im[0,:,0]).reshape((length,1)),
    #   (homo_im[1,:,0]*homo_im[0,:,1]).reshape((length,1)),
    #   homo_im[1,:,0].reshape((length,1)),(homo_im[1,:,1]*homo_im[0,:,0]).reshape((length,1)),
    #   (homo_im[1,:,1]*homo_im[0,:,1]).reshape((length,1)),
    #   homo_im[1,:,1].reshape((length,1)),homo_im[0,:,0].reshape((length,1)),
    #   homo_im[0,:,1].reshape((length,1)),np.ones((length,1))))
    
    ata = np.matmul(A.T,A)
    u, s, vh = np.linalg.svd(ata, full_matrices=True)
    L = vh[-1]
    H = L.reshape(3, 3)

    u1, s1, vh1 = np.linalg.svd(H,full_matrices=True)

    s2 = np.array([(s1[0]+s1[1])/2, (s1[0]+s1[1])/2, 0])
    left = np.matmul(u1,np.diag(s2))
    E = np.matmul(left,vh1)

    return E
    # return H

# takes in 3x3 essential matrix
def returnT_fromE(e):
    ete = np.matmul(e.T,e)
    u, s, vh = np.linalg.svd(ete, full_matrices=True)
    v = vh.T

    return v[:,-1]

def returnUV_fromE(e):
    ete = np.matmul(e.T,e)
    u, s, vh = np.linalg.svd(ete, full_matrices=True)
    v = vh.T
    # v = v/v[-1]

    return u, v

def returnR_fromE(e):

    t = returnT_fromE(e)
    tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]
    u, v = returnUV_fromE(e)
    z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    w = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    d = np.array([[1,0,0],[0,1,0],[0,0,0]])
    r1 = np.dot(u,np.dot(w.T,v.T))
    r2 = np.dot(u,np.dot(w,v.T))

    return r1,r2    

def returnR1_fromE(e):

    t = returnT_fromE(e)
    tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]
    u, v = returnUV_fromE(e)
    z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    w = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    d = np.array([[1,0,0],[0,1,0],[0,0,0]])
    r1 = np.dot(u,np.dot(w.T,v.T))
    r2 = np.dot(u,np.dot(w,v.T))

    return r1


def return_depth_im12(im1,im2,r1,t):

    length = im1.shape[0]
    homo = np.ones((2,length,3))
    homo_im = np.ones((2,length,3))
    # homo[:,:,0] = im1.T
    # homo[:,:,1] = im2.T
    homo[0,:,:2] = im1
    homo[1,:,:2] = im2

    # kk = np.load("data/parameters.npz")
    # # k = kk['k_new']
    # k = kk['k']

    # #shape of k=3x3 
    # homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
    # homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T
    
    rot1 = np.matmul(np.matmul(homo_im[1],r1),homo_im[0].T)
    trans1 = np.matmul(homo_im[1],t.reshape((3,1)))

    A = np.hstack((rot1,trans1))
    ata = np.matmul(A.T,A)
    u, s, vh = np.linalg.svd(ata, full_matrices=True)
    Depth = vh[-1].reshape(length+1,1)

    return Depth

#############################################
#####-----------utils_beta-------------------
#############################################

def return_pcv(img_path, corners_, obj_p):
    objpoints = [] 
    imgpoints = []
    imgpoints.append(corners_)
    objpoints.append(obj_p)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    rx = [[1,0,0],[0,math.cos(rvecs[0][0]),math.sin(rvecs[0][0])],[0,(-1)*math.sin(rvecs[0][0]),math.cos(rvecs[0][0])]]
    ry = [[math.cos(rvecs[0][1]),0,math.sin(rvecs[0][1])],[0,1,0],[(-1)*math.sin(rvecs[0][1]),0,math.cos(rvecs[0][1])]]
    rz = [[math.cos(rvecs[0][2]),math.sin(rvecs[0][2]),0],[(-1)*math.sin(rvecs[0][2]),math.cos(rvecs[0][2]),0],[0,0,1]]
    R1 = np.matmul(rx,ry)
    R = np.matmul(R1,rz)
    m = np.ones((3,4))
    m[:,:3] = R
    m[:,[3]]= np.array(tvecs)
    cv_homography = np.matmul(mtx,m)
    pcv = cv_homography/cv_homography[-1,-1]

    return pcv

def return_fundamentalM(P1, P2):
    A1 = P1[:,:3]
    a1 = P1[:,3]

    A2 = P2[:,:3]
    a2 = P2[:,3]

    b = np.matmul(inv(A2),a2) - np.matmul(inv(A1),a1)
    Sb = [0,(-1)*b[2],b[1],b[2],0,(-1)*b[0],(-1)*b[1],b[0],0]
    Sb = np.array(Sb)
    Sb = Sb.reshape((3,3))

    F = np.matmul(np.matmul(inv(A1).T , Sb) ,inv(A2))

    return F

# 
def errorFundamental(F,img_pt1, img_pt2):
    x1 = homo_img(img_pt1)
    x2 = homo_img(img_pt2)

    w = np.matmul(np.kron(x2,x1).T, F.reshape((9,1)))
    print(w.shape)
    w1 = np.sum(w)

    return w1

#im shape = (2(num_of_images),Number_Of_Corners,2(x,y))
def essential_matrix_pt(im):

    length = im.shape[1]
    homo = np.ones((2,length,3))
    homo_im = np.ones((2,length,3))
    homo[:,:,:2] = im[:,:,:]
    kk = np.load("data/parameters.npz")
    k = kk['k_new']
    # k = kk['k']
    homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
    homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T
    
    A = np.hstack(((homo_im[1,:,0]*homo_im[0,:,0]).reshape((length,1)),
        (homo_im[1,:,0]*homo_im[0,:,1]).reshape((length,1)),
        homo_im[1,:,0].reshape((length,1)),(homo_im[1,:,1]*homo_im[0,:,0]).reshape((length,1)),
        (homo_im[1,:,1]*homo_im[0,:,1]).reshape((length,1)),
        homo_im[1,:,1].reshape((length,1)),homo_im[0,:,0].reshape((length,1)),
        homo_im[0,:,1].reshape((length,1)),np.ones((length,1))))
    
    ata = np.matmul(A.T,A)
    u, s, vh = np.linalg.svd(ata, full_matrices=True)
    L = vh[-1]
    H = L.reshape(3, 3)

    u1, s1, vh1 = np.linalg.svd(H,full_matrices=True)

    s2 = np.array([(s1[0]+s1[1])/2, (s1[0]+s1[1])/2, 0])
    left = np.matmul(u1,np.diag(s2))
    E = np.matmul(left,vh1)


    return E


### Im shape = (2(num_of_images),Number_Of_Corners,2(x,y))
def essential_matrix_ignoredTh2(im):

    length = im.shape[1]
    homo = np.ones((2,length,3))
    homo_im = np.ones((2,length,3))
    homo[:,:,:2] = im[:,:,:]
    kk = np.load("data/parameters.npz")
    k = kk['k_new']
    # k = kk['k']
    homo_im[0,:,:] = np.matmul(inv(k),homo[0,:,:].T).T
    homo_im[1,:,:] = np.matmul(inv(k),homo[1,:,:].T).T
    
    A = np.hstack(((homo_im[1,:,0]*homo_im[0,:,0]).reshape((length,1)),
        (homo_im[1,:,0]*homo_im[0,:,1]).reshape((length,1)),
        homo_im[1,:,0].reshape((length,1)),(homo_im[1,:,1]*homo_im[0,:,0]).reshape((length,1)),
        (homo_im[1,:,1]*homo_im[0,:,1]).reshape((length,1)),
        homo_im[1,:,1].reshape((length,1)),homo_im[0,:,0].reshape((length,1)),
        homo_im[0,:,1].reshape((length,1)),np.ones((length,1))))
    
    ata = np.matmul(A.T,A)
    u, s, vh = np.linalg.svd(ata, full_matrices=True)
    L = vh[-1]
    H = L.reshape(3, 3)

    return H


def essential_matrix_kinv_ignored(homo_im):

    length = homo_im.shape[1]

    A = np.hstack(((homo_im[1,:,0]*homo_im[0,:,0]).reshape((length,1)),
        (homo_im[1,:,0]*homo_im[0,:,1]).reshape((length,1)),
        homo_im[1,:,0].reshape((length,1)),(homo_im[1,:,1]*homo_im[0,:,0]).reshape((length,1)),
        (homo_im[1,:,1]*homo_im[0,:,1]).reshape((length,1)),
        homo_im[1,:,1].reshape((length,1)),homo_im[0,:,0].reshape((length,1)),
        homo_im[0,:,1].reshape((length,1)),np.ones((length,1))))
    
    ata = np.matmul(A.T,A)
    u, s, vh = np.linalg.svd(ata, full_matrices=True)
    L = vh[-1]
    H = L.reshape(3, 3)

    return H

# homo_im is numpy array
def essential_matrix_cal(im):

    length = im.shape[1]
    homo = np.ones((2,length,1,3))
    homo_im = np.ones((2,length,1,3))
    homo[:,:,0,:2] = im[:,:,0,:]
    kk = np.load("data/parameters.npz")
    k = kk['k']
    homo_im[0,:,0,:] = np.matmul(inv(k),homo[0,:,0,:].T).T
    homo_im[1,:,0,:] = np.matmul(inv(k),homo[1,:,0,:].T).T

    A = np.hstack(((homo_im[1,:,0,0]*homo_im[0,:,0,0]).reshape((315,1)),
        (homo_im[1,:,0,0]*homo_im[0,:,0,1]).reshape((315,1)),
        homo_im[1,:,0,0].reshape((315,1)),(homo_im[1,:,0,1]*homo_im[0,:,0,0]).reshape((315,1)),
        (homo_im[1,:,0,1]*homo_im[0,:,0,1]).reshape((315,1)),
        homo_im[1,:,0,1].reshape((315,1)),homo_im[0,:,0,0].reshape((315,1)),
        homo_im[0,:,0,1].reshape((315,1)),np.ones((315,1))))
    
    ata = np.matmul(A.T,A)
    u, s, vh = np.linalg.svd(ata, full_matrices=True)
    L = vh[-1]
    H = L.reshape(3, 3)

    return H,ata

# takes in 3x3 essential matrix
def returnT_fromE(e):
    ete = np.matmul(e.T,e)
    u, s, vh = np.linalg.svd(ete, full_matrices=True)
    v = vh.T

    return v[:,-1]

def returnUV_fromE(e):
    ete = np.matmul(e.T,e)
    u, s, vh = np.linalg.svd(ete, full_matrices=True)
    v = vh.T
    # v = v/v[-1]

    return u, v

def returnR_fromE(e):

    t = returnT_fromE(e)
    tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]
    u, v = returnUV_fromE(e)
    z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    w = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    d = np.array([[1,0,0],[0,1,0],[0,0,0]])
    r1 = np.dot(u,np.dot(w.T,v.T))
    r2 = np.dot(u,np.dot(w,v.T))

    return r1,r2    

def returnP_fromE(e):

    t = returnT_fromE(e)
    tx = [[0,(-1)*t[2],t[1]],[t[2],0,(-1)*t[0]],[(-1)*t[1],t[0],0]]
    u, v = returnUV_fromE(e)
    z = np.array([[0,1,0],[-1,0,0],[0,0,0]])
    w = np.array([[0,1,0],[-1,0,0],[0,0,1]])
    d = np.array([[1,0,0],[0,1,0],[0,0,0]])
    r1 = np.dot(u,np.dot(w.T,v.T))
    r2 = np.dot(u,np.dot(w,v.T))

    m1 = np.dot(r1,np.array([[1,0,0,t[0]],[0,1,0,t[1]],[1,0,0,t[2]]]))
    m2 = np.dot(r1,np.array([[1,0,0,(-1)*t[0]],[0,1,0,(-1)*t[1]],[1,0,0,(-1)*t[2]]]))
    m3 = np.dot(r2,np.array([[1,0,0,t[0]],[0,1,0,t[1]],[1,0,0,t[2]]]))
    m4 = np.dot(r2,np.array([[1,0,0,(-1)*t[0]],[0,1,0,(-1)*t[1]],[1,0,0,(-1)*t[2]]]))

    return m1, m2, m3, m4

def find_corners(img):
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31,
                nlevels=8, fastThreshold=20,scaleFactor=1.2, 
                WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, 
                firstLevel=0,nfeatures= 500)
    kp = orb.detect(img,None)
    kp, des = orb.compute(img, kp)

    corn_arr = []
    for i in range(len(kp)):
        corn_arr.append(kp[i].pt)
    # corner = np.array(corn_arr)

    # corner_x = corner[:,0]
    # corner_y = corner[:,1]

    # return corner_x, corner_y
    return corn_arr

### homo_im.shape = (2,numCorners,3) and E = 3x3
def error_essential_matrix(homo_im,E):
    err = np.matmul(np.matmul(homo_im[1],E),homo_im[0].T)
    return np.sum(err)/np.shape(err)[0]

#############################################
##########----------utils--------------------
#############################################



# output = 3x3 projection_matrix, input image shape=(num_points,2); obj shape=(num_points,3)
def projection_matrix_direct(imgp1, objp):
    
    obj = objp.T
    ab = np.matmul(obj,obj.T)
    abinv = inv(ab)
    abc = np.matmul(obj.T,abinv)
    camera_matrix = np.matmul(imgp1,abc)
    
    return camera_matrix

# output = 3x3 projection_matrix, input image shape=(num_points,2); obj shape=(num_points,3) 
def projection_matrix3(img_p, obj_p):
    
    C = []

    for i in range(315):
        C.append(np.array([obj_p[i,0], obj_p[i,1],1,0,0,0, (-1)*obj_p[i,0]*img_p[i,0], 
                           (-1)*obj_p[i,1]*img_p[i,0], (-1)*img_p[i,0]]))
        C.append(np.array([0,0,0, obj_p[i,0], obj_p[i,1],1, 
                           (-1)*obj_p[i,0]*img_p[i,1], (-1)*obj_p[i,1]*img_p[i,1],(-1)*img_p[i,1]]))
    
    c = np.array(C)
    ctc = np.matmul(c.T,c)
    u, s, vh = np.linalg.svd(ctc, full_matrices=True)
    L = vh[-1]
    H = L.reshape(3, 3)
    H = H/H[-1,-1]
    
    return H

# output = 3x4 projection_matrix, input image shape=(num_points,2); obj shape=(num_points,3) 
def projection_matrix4(img_p, obj_p):
    
    C = []

    for i in range(obj_p.shape[0]):
        C.append(np.array([obj_p[i,0], obj_p[i,1], obj_p[i,2],1,0,0,0,0,(-1)*obj_p[i,0]*img_p[i,0], 
                           (-1)*obj_p[i,1]*img_p[i,0],(-1)*obj_p[i,2]*img_p[i,0], (-1)*img_p[i,0]]))
        
        C.append(np.array([0,0,0,0, obj_p[i,0], obj_p[i,1], obj_p[i,2],1,(-1)*obj_p[i,0]*img_p[i,1], 
                           (-1)*obj_p[i,1]*img_p[i,1], (-1)*obj_p[i,2]*img_p[i,1],(-1)*img_p[i,1]]))
    
    c = np.array(C)
    ctc = np.matmul(c.T,c)
    u, s, vh = np.linalg.svd(ctc, full_matrices=True)
    L = vh[-2]
    H = L.reshape(3, 4)

#     H = H/H[-1,-1]
    
    return H

def return_pcv(img_path, corners_, obj_p):
    objpoints = [] 
    imgpoints = []
    imgpoints.append(corners_)
    objpoints.append(obj_p)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    rx = [[1,0,0],[0,math.cos(rvecs[0][0]),math.sin(rvecs[0][0])],[0,(-1)*math.sin(rvecs[0][0]),math.cos(rvecs[0][0])]]
    ry = [[math.cos(rvecs[0][1]),0,math.sin(rvecs[0][1])],[0,1,0],[(-1)*math.sin(rvecs[0][1]),0,math.cos(rvecs[0][1])]]
    rz = [[math.cos(rvecs[0][2]),math.sin(rvecs[0][2]),0],[(-1)*math.sin(rvecs[0][2]),math.cos(rvecs[0][2]),0],[0,0,1]]
    R1 = np.matmul(rx,ry)
    R = np.matmul(R1,rz)
    m = np.ones((3,4))
    m[:,:3] = R
    m[:,[3]]= np.array(tvecs)
    cv_homography = np.matmul(mtx,m)
    pcv = cv_homography/cv_homography[-1,-1]

    return pcv

def returnP_fromK(mtx,rvecs,tvecs):
    rx = [[1,0,0],[0,math.cos(rvecs[0][0]),math.sin(rvecs[0][0])],[0,(-1)*math.sin(rvecs[0][0]),math.cos(rvecs[0][0])]]
    ry = [[math.cos(rvecs[0][1]),0,math.sin(rvecs[0][1])],[0,1,0],[(-1)*math.sin(rvecs[0][1]),0,math.cos(rvecs[0][1])]]
    rz = [[math.cos(rvecs[0][2]),math.sin(rvecs[0][2]),0],[(-1)*math.sin(rvecs[0][2]),math.cos(rvecs[0][2]),0],[0,0,1]]
    R1 = np.matmul(rx,ry)
    R = np.matmul(R1,rz)
    m = np.ones((3,4))
    m[:,:3] = R
    m[:,[3]]= np.array(tvecs)
    cv_homography = np.matmul(mtx,m)

    return cv_homography

# returns k,r1,r2 ; takes input of projection_matrix(3,4)
def RQ_decomposition(projection_matrix4):
    
    p1 = projection_matrix4[:,:3]
    p2 = projection_matrix4[:,3]
    
    k, r1 = linalg.rq(p1)
    #np.allclose(p1, k @ r1)
    k = k/k[-1,-1]
    
    r2 = np.matmul(inv(k),p2)
    
    return k,r1,r2

#returns (num_corners,2)
def return_imagepoints(image_path,grid):
    grid_x,grid_y=grid
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (grid_x,grid_y),None)
    if(ret==False):
        print("image/corner doesnt exist")
    else:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpt = np.ones((grid_x*grid_y,2))
        imgpt[:,0] = corners[:,0,0]
        imgpt[:,1] = corners[:,0,1]
        
        return imgpt, corners

    #returns (num_corners,2)
def return_imgGraypoints(gray,grid):
    grid_x,grid_y=grid
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (grid_x,grid_y),None)
    if(ret==False):
        print("image/corner doesnt exist")
        return ret,None, None

    else:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpt = np.ones((grid_x*grid_y,2))
        imgpt[:,0] = corners[:,0,0]
        imgpt[:,1] = corners[:,0,1]
        
        return ret,imgpt, corners

# returns projection of a point in image plane, arg: p(3,4) and obj list len=3;
def img_projection(projection_matrix, obj_p):
    
    length = obj_p.shape[0]
    new_var = np.ones((length,4))
    new_var[:,:3] = obj_p
    
    img_p = np.matmul(projection_matrix, new_var.T)
#     img_p = img_p/img_p[-1]
    
    return img_p


#returns (num_obj,3)
def return_objpoints(grid):
    grid_x,grid_y=grid
    objp = np.zeros((grid_x*grid_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:grid_x,0:grid_y].T.reshape(-1,2)

    return objp



def homo_img(img_p):
    img_homo = np.ones((3,315))
    img_homo[0,:] = img_p[:,0]
    img_homo[1,:] = img_p[:,1]
    
    return img_homo

def homo_obj3(obj_p):
    obj_homo3 = np.ones((3,315))
    obj_homo3[0,:] = obj_p[:,0]
    obj_homo3[1,:] = obj_p[:,1]
    
    return obj_homo3

def homo_obj4(obj_p):
    obj_homo4 = np.ones((4,315))
    obj_homo4[0,:] = obj_p[:,0]
    obj_homo4[1,:] = obj_p[:,1]
    obj_homo4[2,:] = obj_p[:,2]
    
    return obj_homo4    

"""
img_p=return_imagepoints()
obj_p=return_objpoints()
P=projection_matrix3(img_p,obj_p)

print(P)
"""
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
