import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, LSTM, Reshape, Masking
from tensorflow.keras import Model
from tensorflow.keras.initializers import TrucatedNormal


class ReconChessNet(Model):
	#Implements Tensorflow NN for ReconBot
	def __init__(self):
		super(MyModel, self).__init__()
	    self.conv1 = Conv2D(16, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TrucatedNormal,data_format='channels_first')
	    self.conv2 = Conv2D(32, 2, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TrucatedNormal,data_format='channels_first')
	    self.conv3 = Conv2D(64, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TrucatedNormal,data_format='channels_first')
	    self.conv4 = Conv2D(128, 1, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TrucatedNormal,data_format='channels_first')
	    self.flatten = Flatten(data_format='channels_first')
	    self.lstm = LSTM(256,stateful=True)

	    self.pir_d = Dense(6*6)
	    self.pir = Reshape((6,6))

	    self.pis_d = Dense(8*8)
	    self.pis = Reshape((8,8))

	    self.pim_d = Dense(8*8)
	    self.pim = Reshape((8,8))

	def call(self,x):
		pass

	def get_lstm(self, x):
		#compute down to the intermediate lstm layer
	    x = self.conv1(x)
	    x = self.conv2(x)
	    x = self.conv3(x)
	    x = self.conv4(x)
	    x = self.flatten(x)
	    x = self.lstm(x)
	    return x

	def get_pir(self,x,mask):
		#compute recon policy
		x = self.get_lstm(x)
		x = self.pir_d(x)
		x = x * mask
		x = tf.keras.activations.softmax(x)
		x = self.pir(x)
		return x

	def get_pis(self,x,mask):
		#compute piece selection policy
		lstm = self.get_lstm(x)
		x = self.pis_d(lstm)
		x = x * mask
		x = tf.keras.activations.softmax(x)
		x = self.pis(x)
		return lstm,x

	def get_pim(self,lstm,mask):
		#compute piece movement policy
		x = self.pim_d(lstm)
		x = x * mask
		x = tf.keras.activations.softmax(x)
		x = self.pim(x)
		return x
		


