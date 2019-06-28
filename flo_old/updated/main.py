from flowlib import *
from utils_flow import *
# from utils import *




F=read_flow("frame_0001.flo")
F=F[np.newaxis,:,:,:]
corr1, corr2 = cor_from_flow(F)

e = compute_essential_matrix(corr1[0],corr2[0],K1=cam_read("frame_0001.cam"))

r,t = return_RT_from_E(e)
