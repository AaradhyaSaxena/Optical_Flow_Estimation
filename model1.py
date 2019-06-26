import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from generator import *

###---------------------model-----------------------------

def model():
	x = tf.placeholder(tf.float32, shape=(4,436,1024,6), name='x')

	with tf.variable_scope('warp') as scope:

		# tf.set_random_seed(1)
		W1 = tf.get_variable("W1", [5, 5, 6, 12], initializer = tf.contrib.layers.xavier_initializer())
		W2 = tf.get_variable("W2", [5, 5, 12, 24], initializer = tf.contrib.layers.xavier_initializer())
		W3 = tf.get_variable("W3", [5, 5, 24, 12], initializer = tf.contrib.layers.xavier_initializer())
		W3u = tf.get_variable("W3u", [5, 5, 12, 12], initializer = tf.contrib.layers.xavier_initializer())
		W4 = tf.get_variable("W4", [5, 5, 12, 6], initializer = tf.contrib.layers.xavier_initializer())
		W4u = tf.get_variable("W4u", [5, 5, 6, 6], initializer = tf.contrib.layers.xavier_initializer())
		W5 = tf.get_variable("W5", [5, 5, 6, 2], initializer = tf.contrib.layers.xavier_initializer())

		parameters = { W1='W1',W2='W2',W3='W3',W3u='W3u',W4='W4',W4u='W4u'W5='W5'}

		W1 = parameters['W1']
		W2 = parameters['W2']		
		W3 = parameters['W3']
		W4 = parameters['W4']
		W5 = parameters['W5']

		Z1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding="SAME")
		A1 = tf.nn.relu(Z1)
		P1 = tf.nn.max_pool(A1, ksize=[1,4,4,1], strides=[1,4,4,1], padding="VALID")

		Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding="SAME")
		A2 = tf.nn.relu(Z2)
		P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding="VALID")

		Z3 = tf.nn.conv2d(P2,W3, strides=[1,1,1,1], padding="SAME")
		A3 = tf.nn.relu(Z3)
		P3 = tf.nn.conv2d_transpose(A3,W3u, strides=[1,2,2,1], padding="SAME")

		Z4 = tf.nn.conv2d(P3,W4, strides=[1,1,1,1], padding="SAME")
		A4 = tf.nn.relu(Z4)
		P4 = tf.nn.conv2d_transpose(A4,W4u, strides=[1,2,2,1], padding="SAME")

		Z5 = tf.nn.conv2d(P4,W5, strides=[1,1,1,1], padding="SAME")
		F = tf.nn.relu(Z5)

		

###---------------------run--------------------------------

A = ImageSequence()

x1, _ = A.__getitem__()
x_batch = np.concatenate(x1[0][:,:,:,:,0],x1[1][:,:,:,:,0], axis=-1)

x, f_pred, loss = model()

optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variable_initializer()
with tf.session() as session:
	session.run(init)

	feed_dict = {x = x_batch}
	for _ in range(100):
		loss_val, _ = session.run([loss,optimizer], feed_dict)
		prin("loss", loss_val.mean())

	y_pred_batch = session.run(f_pred,{x:x_batch})


