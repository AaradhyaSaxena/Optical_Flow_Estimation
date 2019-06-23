import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
from utils import *
# from keras.models import Model
# from keras.layers import *


##---------------crop--------------------

# images = glob('*.png')
# image_counter = 0

# for fname in images:
#     img = cv2.imread(fname,0)
#     crop_img = img[32:32+416, 112:112+416]

#     img_name = "crop_{}.png".format(image_counter)
#     cv2.imwrite(img_name, crop_img)
#     image_counter = image_counter +1
#     # cv2.imshow("cropped", crop_img)
#     # cv2.waitKey(0)

##---------------save---------------------

# X1 =[]
# X2 =[]

# img_paths=glob('*.png')
# N = len(img_paths)

# for i in range(N):
# 	j = i+1
# 	while(j<N):
# 		img1 = cv2.imread(img_paths[i],0)
# 		img2 = cv2.imread(img_paths[j],0)
# 		X1.append(img1)
# 		X2.append(img2)
# 		j =j+1

# X1 = np.array(X1)
# X2 = np.array(X2)
# X1 = X1[:,:,:,np.newaxis]
# X2 = X2[:,:,:,np.newaxis]
# np.savez('data/unet', X1 = X1, X2 = X2)
# data = np.load('data/unet.npz')
# print(data.files)

##-------------------flow_to_depth------------------

# im1 = np.load('data/unet.npz')['X1']
# im2 = np.load('data/unet.npz')['X2']
# flow = np.load('data/unet_out.npz')['y']

# essentialMatrix,tvecs,rvecs,depth,corr1,corr2 = return_corr_from_flow(im1, im2, flow)

# np.savez('data/flow_to_depth', e = essentialMatrix, r = rvecs, t = tvecs, 
# 			depth = depth, corr1 = corr1, corr2 = corr2)
# l = np.load('data/flow_to_depth.npz')
# print(l.files)

##-------------------depth_plot---------------------

# data = np.load('data/flow_to_depth.npz')
# print(data.files)

# depth = data['depth']
# # depth = (depth +1)/2

# for i in range(len(depth)):
# 	im0 = depth[i]
# 	cv2.imwrite('depth_{}.jpg', im0).format(i)
# 	# cv2.imshow("image", im0);
# 	# cv2.waitKey();













