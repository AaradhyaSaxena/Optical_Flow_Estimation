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
###################################################

def DSSIM_updated(y_true, y_pred, batch_size=2):
    """Need tf0.11rc to work"""
    y_true = tf.reshape(y_true, [batch_size] + get_shape(y_pred)[1:])
    y_pred = tf.reshape(y_pred, [batch_size] + get_shape(y_pred)[1:])
    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    patches_true = tf.extract_image_patches(y_true, [1, 50, 50, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.extract_image_patches(y_pred, [1, 50, 50, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    u_true = K.mean(patches_true, axis=3)
    u_pred = K.mean(patches_pred, axis=3)
    var_true = K.var(patches_true, axis=3)
    var_pred = K.var(patches_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    ssim = tf.select(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    return K.mean(((1.0 - ssim) / 2))

######################################################