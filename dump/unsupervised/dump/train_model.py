from model import *
from gen_data import *
from generator import *
# from init import *
import os
from model_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] ='1'
#----------------------training ---------------

# def test(model,i):
# 	i=0

# 	X,Y=get_data()
# 	x1=X[i][0]/255.0
# 	x2=X[i][1]/255.0
# 	y=Y[i]

# 	"""
# 	x1=resize(x1,(SHAPE_Y,SHAPE_X))
# 	x2=resize(x2,(SHAPE_Y,SHAPE_X))
# 	y=resize(y,(SHAPE_Y,SHAPE_X))
# 	"""

# 	x1=x1[:SHAPE_Y,:SHAPE_X]
# 	x2=x2[:SHAPE_Y,:SHAPE_X]
# 	y=y[:SHAPE_Y,:SHAPE_X]

# 	x1=x1[np.newaxis,:,:,np.newaxis]
# 	x2=x2[np.newaxis,:,:,np.newaxis]

# 	Y_out=model.predict([X1,X2])[0]

# 	Y_out=np.sqrt(Y_out[:,:,0]*Y_out[:,:,0]+Y_out[:,:,1]*Y_out[:,:,1])
# 	y=np.sqrt(y[:,:,0]*y[:,:,0]+y[:,:,1]*y[:,:,1])

# 	plt.figure("x1")
# 	plt.imshow(x1[0,:,:,0])
# 	plt.figure("x2")
# 	plt.imshow(x2[0,:,:,0])
# 	plt.figure("y")
# 	plt.imshow(y)
# 	plt.figure("Y_out")
# 	plt.imshow(Y_out)

# 	error=np.mean(np.abs(y-Y_out))
# 	return error



batch_size=4
model_base=create_model()
model=compile_model_new(model_base,batch_size,lambda1=0.1)
	#model.load_weights("../models/w2.h5")


# ##########---------------------
# imgen=ImageSequence_fixed()
# # [X1,X2],Y = imgen.__getitem__()
# [X1,X2],Y = imgen.__getitem__()

# # model.load_weights('../data/orig_temp.h5')
# model.fit([X1,X2],Y,epochs=10000)
# # model.fit_generator(imgen,None, epochs=2000)
# # data = np.load('data/unet.npz')
# # X1 = data['X1']/255.0
# # X2 = data['X2']/255.0

# # model.fit([X1,X2],None,batch_size=batch_size,epochs=10)
# model.save_weights("../data/orig_temp.h5")

# y=model.predict([X1,X2])
# y1 = flow_mag(y)
# # %matplotlib
# # plt.figure()
# # plt.imshow(y1[0])

# # np.savez('data/unet_out', y = Y_out)
# # l = np.load('data/unet_out.npz')
# # print(l.files)
# # 

# # Y_out=np.sqrt(Y_out[:,:,0]*Y_out[:,:,0]+Y_out[:,:,1]*Y_out[:,:,1])

# # Y_out=model.predict([X1,X2])[0]
# # Y_out=np.sqrt(Y_out[:,:,0]*Y_out[:,:,0]+Y_out[:,:,1]*Y_out[:,:,1])

##############--------------