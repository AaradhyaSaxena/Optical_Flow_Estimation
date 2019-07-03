import tensorflow as tf
import  keras.backend as  K
from generator import *
from image_warp import *

imgen=ImageSequence_new()
[X1,X2],Y = imgen.__getitem__()



def mask(x1,x2,y):

	left = tf.square( y + image_warp((x1), y) )
