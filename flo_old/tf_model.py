import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from generator import *
from image_warp import *

###---------------------model-----------------------------

# def initialize_parameters():
# 	W1 = tf.get_variable("W1", [5, 5, 6, 12], initializer = tf.contrib.layers.xavier_initializer())
# 	W2 = tf.get_variable("W2", [5, 5, 12, 24], initializer = tf.contrib.layers.xavier_initializer())
# 	W3 = tf.get_variable("W3", [5, 5, 24, 12], initializer = tf.contrib.layers.xavier_initializer())
# 	W3u = tf.get_variable("W3u", [5, 5, 12, 12], initializer = tf.contrib.layers.xavier_initializer())
# 	W4 = tf.get_variable("W4", [5, 5, 12, 6], initializer = tf.contrib.layers.xavier_initializer())
# 	W4u = tf.get_variable("W4u", [5, 5, 6, 6], initializer = tf.contrib.layers.xavier_initializer())
# 	W5 = tf.get_variable("W5", [5, 5, 6, 2], initializer = tf.contrib.layers.xavier_initializer())

# 	warp_params = { W1:'W1',W2:'W2',W3:'W3',W3u:'W3u',W4:'W4',W4u:'W4u',W5:'W5'}

# 	return warp_params

# def model(parameters):
def model():

	# x = tf.placeholder(tf.float32, shape=(4,436,1024,6), name='x')
	x1 = tf.placeholder(tf.float32, shape=(4,436,1024,3), name='x1')
	x2 = tf.placeholder(tf.float32, shape=(4,436,1024,3), name='x2')

	W1 = tf.get_variable("W1", [5, 5, 6, 12], initializer = tf.contrib.layers.xavier_initializer())
	W2 = tf.get_variable("W2", [5, 5, 12, 24], initializer = tf.contrib.layers.xavier_initializer())
	W3 = tf.get_variable("W3", [5, 5, 24, 12], initializer = tf.contrib.layers.xavier_initializer())
	W3u = tf.get_variable("W3u", [5, 5, 12, 12], initializer = tf.contrib.layers.xavier_initializer())
	W4 = tf.get_variable("W4", [5, 5, 12, 6], initializer = tf.contrib.layers.xavier_initializer())
	W4u = tf.get_variable("W4u", [5, 5, 6, 6], initializer = tf.contrib.layers.xavier_initializer())
	W5 = tf.get_variable("W5", [5, 5, 6, 2], initializer = tf.contrib.layers.xavier_initializer())

	# W1 = parameters['W1']
	# W2 = parameters['W2']		
	# W3 = parameters['W3']
	# W3u = parameters['W3u']
	# W4 = parameters['W4']
	# W4u = parameters['W4u']
	# W5 = parameters['W5']

	x = tf.concat([x1,x2],3)
	Z1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding="SAME")
	A1 = tf.nn.relu(Z1)
	P1 = tf.nn.max_pool(A1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

	Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding="SAME")
	A2 = tf.nn.relu(Z2)
	P2 = tf.nn.max_pool(A2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

	Z3 = tf.nn.conv2d(P2,W3, strides=[1,1,1,1], padding="SAME")
	A3 = tf.nn.relu(Z3)
	P3 = tf.nn.conv2d_transpose(A3,W3u, output_shape=[4,218,512,12],strides=[1,2,2,1])#, padding="SAME")

	Z4 = tf.nn.conv2d(P3,W4, strides=[1,1,1,1], padding="SAME")
	A4 = tf.nn.relu(Z4)
	P4 = tf.nn.conv2d_transpose(A4,W4u, output_shape=[4,436,1024,6],strides=[1,2,2,1])#, padding="SAME")

	Z5 = tf.nn.conv2d(P4,W5, strides=[1,1,1,1], padding="SAME")
	F = tf.nn.relu(Z5)

	with tf.variable_scope('image_warp') as scope:

		# x1 = x[:,:,:,:3]
		# x2 = x[:,:,:,3:]
		warped = image_warp(x1,F)
		loss = tf.reduce_mean(tf.square(warped - x2))

	return loss, x1, x2, F, warped



###---------------------run--------------------------------

def run():

	A = ImageSequence()
	im1,im2, _ = A.__getitem__()
	# x_batch = np.concatenate(x1[0][:,:,:,:,0],x1[1][:,:,:,:,0], axis=-1)

	# parameters = initialize_parameters()
	loss, x1, x2, F, warped = model()
	optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

	init = tf.initializers.global_variables()

	with tf.Session() as session:
		# parameters = initialize_parameters()
		session.run(init)
		feed_dict = {x1:im1, x2:im2}

		for _ in range(100):
			loss_val, _ = session.run([loss,optimizer], feed_dict)
			print("loss", loss_val.mean())
			
		loss,x1,x2,F = session.run([loss,x1,x2,F],{x1:im1, x2:im2})
		np.savez('run1', loss= loss,x1 = x1,x2 =x2,F = F)
	

if __name__ == "__main__":
	run()
