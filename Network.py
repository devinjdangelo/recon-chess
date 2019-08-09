import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model




class ReconChessNet(Model):
	#Implements Tensorflow NN for ReconBot
	def __init__(self):
	    super(MyModel, self).__init__()
	    self.conv1 = Conv2D(16, 2, strides=(2,2) activation=tf.nn.leaky_relu)
	    self.conv2 = Conv2D(32, 2, strides=(1,1) activation=tf.nn.leaky_relu)
	    self.conv3 = Conv2D(64, 2, strides=(2,2) activation=tf.nn.leaky_relu)
	    self.conv4 = Conv2D(128, 1, strides=(1,1) activation=tf.nn.leaky_relu)
	    self.flatten = Flatten()
	    
	    self.d1 = Dense(128, activation='relu')
	    self.d2 = Dense(10, activation='softmax')

	def call(self, x):
	    x = self.conv1(x)
	    x = self.flatten(x)
	    x = self.d1(x)
	    return self.d2(x)
