import tensorflow as tf
import  keras.backend as  K
from generator import *
from image_warp import *

#########-------------------Mask---------------------------

def mask(i1,i2,o1,o2, alpha1=0.01, alpha2=0.5):

	left_f = tf.square( o1 + image_warp(o2,o2))
	right_f = alpha1*( tf.square(o1) + tf.square(image_warp(o2,o2))) + alpha2

	oxf = tf.less(right_f, left_f) 

	left_b = tf.square( o2 + image_warp(o1,o1) )
	right_b = alpha1*( tf.square(o2) + tf.square(image_warp(o1,o1))) + alpha2

	oxb = tf.less(right_b, right_b)

	return oxf, oxb

#######--------------Occlusion_aware_data_loss-------------

def charbonnier(x, gamma=0.45, e=0.001):

	loss = K.pow([tf.square(x) + tf.square(e)], 0.45)
	return loss[0,:,:,:]

def photometric_error(i1,i2,o1,o2):

	err_f = tf.reduce_sum(tf.subtract(i1, image_warp(i2,o2)),-1)
	err_b = tf.reduce_sum(tf.subtract(i2, image_warp(i1,o1)),-1)
	return err_f, err_b

def flow_error(o1,o2):

	ff = o1 + image_warp(o2,o2)
	fb = o2 + image_warp(o1,o1)
	return ff, fb

def occLoss(i1,i2,o1, occ_punishment =0.1):

	i1 = tf.convert_to_tensor(i1)
	i2 = tf.convert_to_tensor(i2)
	o1 = tf.convert_to_tensor(o1)
	o2 = image_warp(-o1,o1)

	oxf, oxb = mask(i1,i2,o1,o2)
	mask_f = oxf[:,:,:,0]
	mask_b = oxb[:,:,:,1]

	err_f, err_b = photometric_error(i1,i2,o1,o2)

	occ_loss1 = (tf.reduce_sum(tf.boolean_mask(charbonnier(err_f), mask_f)))/(436*1024)
	occ_loss2 = (tf.reduce_sum(tf.boolean_mask(charbonnier(err_b), mask_b)))/(436*1024)

	occ_punish1 = tf.multiply(tf.reduce_sum(tf.cast(mask_f, tf.float32)),occ_punishment)
	occ_punish2 = tf.multiply(tf.reduce_sum(tf.cast(mask_b, tf.float32)),occ_punishment)

	occ_loss = occ_loss1 + occ_loss2 + occ_punish1 + occ_punish2

	return occ_loss
######------------------------------------------------------

# imgen=ImageSequence_fixed()
# [X1,X2],Y = imgen.__getitem__()

# loss = occLoss(X1,X2,Y)

######------------------------------------------------------

# def compile_model(model,lambda_smoothness = 0.05, lambda_flow=0.05, lambda_mse=0.001, occ_punishment = 0.01):

#     i1=model.inputs[0]
#     i2=model.inputs[1]
#     o1=model.outputs[0]
#     o2 = image_warp(-o1,o1)

#     oxf, oxb = mask(i1,i2,o1,o2)
#     mask_f = oxf[:,:,:,0]
#     mask_b = oxb[:,:,:,1]

#     err_f, err_b = photometric_error(i1,i2,o1,o2)
#     flow_f, flow_b = flow_error(o1,o2)

#     ###--------Occlusion_aware_mse_rec-------------------------------------
#     occ_loss1 = (tf.reduce_sum(tf.boolean_mask(charbonnier(err_f), mask_f)))#/(436*1024)
#     occ_loss2 = (tf.reduce_sum(tf.boolean_mask(charbonnier(err_b), mask_b)))#/(436*1024)
#     occ_loss = (occ_loss1 + occ_loss2)*lambda_mse

#     ###--------Occlusion_aware_mse_flow------------------------------------
#     flow_loss1 = tf.reduce_sum(tf.boolean_mask(charbonnier(flow_f), mask_f))
#     flow_loss2 = tf.reduce_sum(tf.boolean_mask(charbonnier(flow_b), mask_f))
#     flow_loss = (flow_loss1 + flow_loss2)*lambda_flow

#     ###--------Punishment_for_occlusion-----------------------------------------
#     occ_punish1 = tf.multiply(tf.reduce_sum(tf.cast(mask_f, tf.float32)),occ_punishment)
#     occ_punish2 = tf.multiply(tf.reduce_sum(tf.cast(mask_b, tf.float32)),occ_punishment)
#     occ_punish = occ_punish1 + occ_punish2

#     ###--------Gradient_smoothness--------------------------------------------
#     ux,uy=grad_xy(o1[:,:,:,:1])
#     vx,vy=grad_xy(o1[:,:,:,1:2])
#     sm_loss_o1 = K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy))
#     ux,uy=grad_xy(o2[:,:,:,:1])
#     vx,vy=grad_xy(o2[:,:,:,1:2])
#     sm_loss_o2 = K.mean(K.abs(ux*ux)+ K.abs(uy*uy)+ K.abs(vx*vx)+ K.abs(vy*vy))
#     sm_loss = (sm_loss_o1 + sm_loss_o2)*lambda_smoothness   

#     ### Reconstruction_loss_ssim_(occlusion_not considered)
#     # I2_rec=image_warp(I1,o1)
#     # I1_rec=image_warp(I2,o2)
#     # re_loss1=DSSIMObjective(kernel_size=50)(I2,I2_rec)
#     # re_loss2=DSSIMObjective(kernel_size=50)(I1,I1_rec)

#     total_loss = sm_loss + occ_loss + occ_punish + flow_loss 

#     model = Model(inputs=[i1,i2], outputs=[o1])
#     model.add_loss(total_loss)
#     model.compile(optimizer=keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0))
   	
#     return model





