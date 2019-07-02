
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from skimage.transform import resize


def read_image(file_path):
    Img=misc.imread(file_path)
    Img=resize(Img,(436,1024))
    return Img


I1=read_image("../data/frame_0001.png")
I2=read_image("../data/frame_0001.png")

x_batch = []
x_batch.append([I1,I2])
x_batch = np.array(x_batch, np.float32)

[X1,X2] = [x_batch[:,0,:,:,:],x_batch[:,1,:,:,:]]
