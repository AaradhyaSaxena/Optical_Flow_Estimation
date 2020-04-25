import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob

import math
from scipy import linalg
from numpy.linalg import inv
# from utils import *
from optical_flow_toolkit import *
from utils_flow import *

###----------------------data------------------------------

f=read_flo_file("data/flo/frame_0001.flo")
f = f[np.newaxis,:,:,:]

# im1 = cv2.imread("00_flow_data/image/frame_0001.png",0)
# im2 = cv2.imread("00_flow_data/image/frame_0002.png",0)
# im1 = im1[np.newaxis,:,:]
# im2 = im2[np.newaxis,:,:]

depth,essentialMatrix,tvecs,rvecs,corr1,corr2 = return_corr_from_flow(f)

print(essentialMatrix)

# np.savez('depth_12', depth=depth)

###----------------------preprocessing------------------------------

# data = np.load('depth_12.npz')
# depth = data['depth']

# ## coeffe of translation term
# # scale = depth[0,:,436,0]

# # dividing by the co-effe of translation term to scale all the other co-effe
# depth_scale = depth[0,:,:,0]/depth[0,:,436,0,None]

# # plt.imshow(depth[0,:,:,0], cmap=plt.get_cmap('flag'))

# plt.imshow(depth_scale[:,:].T, cmap=plt.get_cmap('flag'))

# plt.show()








###----------------------------------------------------

# f=read_flo_file("/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/flow/alley_1/frame_0001.flo")

# im1 = cv2.imread("/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/alley_1/frame_0001.png",0)
# im2 = cv2.imread("/media/newhd/data/flow/MPI_SINTEL/MPI-Sintel-complete/training/albedo/alley_1/frame_0002.png",0)
# f = f[np.newaxis,:,:,:]
# im1 = im1[np.newaxis,:,:]
# im2 = im2[np.newaxis,:,:]

# essentialMatrix,tvecs,rvecs,depth,corr1,corr2 = return_corr_from_flow(im1, im2, f)



##--------------crop------------------------
# cropped_flow =[]
# flo_counter=0
# for flo in flo_paths:
# 	flow = read_flow(flo)
# 	crop_flo = flow[10:10+416, 304:304+416,:]
# 	cropped_flow.append(crop_flo)
# 	flo_name = "flo_crop_%d.flo"%(flo_counter)
# 	cv2.imwrite(flo_name, crop_flo)
# 	flo_counter = flo_counter +1
#  	# cv2.imshow("cropped", crop_flo)
#  	# # cv2.waitKey(0)
# cropped_flow = np.array(cropped_flow)
##--------------------------------------------



































# image_counter=0
# images = glob('*.png')
# for fname in images:
# 	img = cv2.imread(fname,0)
# 	# crop_img = img[10:10+416, 304:304+416]

# 	# img_name = "crop_{}.png".format(image_counter)
# 	# cv2.imwrite(img_name, crop_img)
# 	# image_counter = image_counter +1
# 	# cv2.imshow("cropped", crop_img)
# 	# cv2.waitKey(0)

# np.savez('flow_data', flow = cropped_flow)
# fl = np.load('flow_data.npz')

# fl.files
