import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.transform import resize

from generator import *
from image_warp import *
from keras.layers import *




def return_func():
	I=Input(shape=(436,1024,3))
	F=Input(shape=(436,1024,2))
	F_new=image_warp(-F,F)
	# I2=image_warp(I,F)
	# f=K.function(inputs=[I,F],outputs=[I2])
	return f



###############-------------------------------------------------

imgen=ImageSequence_fixed()
[X1,X2],Y = imgen.__getitem__()

f=return_func()

X1_out=f([X1,Y])

plt.imshow(X1[0])

im_rec = image_warp(X1[0],Y[0])

# plt.figure("rec")
# plt.imshow(im_rec)



# def read_image(file_path):
#     Img=misc.imread(file_path)
#     Img=resize(Img,(436,1024))
#     return Img


# I1=read_image("../data/frame_0001.png")
# I2=read_image("../data/frame_0001.png")

# x_batch = []
# x_batch.append([I1,I2])
# x_batch = np.array(x_batch, np.float32)

# [X1,X2] = [x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]]
