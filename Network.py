import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, LSTM, Reshape, Masking
from tensorflow.keras import Model
from tensorflow.keras.initializers import TrucatedNormal
from tensorflow.keras.preprocessing import pad_sequences
import numpy


class ReconChessNet(Model):
	#Implements Tensorflow NN for ReconBot
	def __init__(self):
	    super(MyModel, self).__init__()
	    self.convshape = Reshape((8,8))
	    self.conv1 = Conv2D(16, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TrucatedNormal,data_format='channels_first')
	    self.conv2 = Conv2D(32, 2, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TrucatedNormal,data_format='channels_first')
	    self.conv3 = Conv2D(64, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TrucatedNormal,data_format='channels_first')
	    self.conv4 = Conv2D(128, 1, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TrucatedNormal,data_format='channels_first')
	    self.flatten = Flatten(data_format='channels_first')
	    self.lstm = LSTM(256,stateful=True)

	    self.pir = Dense(6*6)

	    self.pis = Dense(8*8)

	    self.pim = Dense(8*8)

	    self.v = Dense(1)

	def call(self,x):
		pass

	def get_lstm(self, x):
		#compute down to the intermediate lstm layer
		seq_lens = [len(seq) for seq in x]
		mask = np.zeros((len(seq_lens,max(seq_lens))),dtype=np.bool)
		for idx,l in enumerate(seq_lens):
			mask[idx,:l] = 1

		x = pad_sequences(x,padding='post')
		x = self.convshape(x)
	    x = self.conv1(x)
	    x = self.conv2(x)
	    x = self.conv3(x)
	    x = self.conv4(x)
	    x = self.flatten(x)
	    x = self.lstm(x,mask=mask)
	    return x

	def get_pir(self,lstm,mask):
		#compute recon policy
		x = self.pir(lstm)
		x = x * mask
		x = tf.keras.activations.softmax(x)
		return x

	def get_pis(self,x,mask):
		#compute piece selection policy
		lstm = self.get_lstm(x)
		x = self.pis(lstm)
		x = x * mask
		x = tf.keras.activations.softmax(x)
		return lstm,x

	def get_pim(self,lstm,mask):
		#compute piece movement policy
		x = self.pim(lstm)
		x = x * mask
		x = tf.keras.activations.softmax(x)
		return x

	def get_v(self,lstm):
		x = self.v(lstm)
		return x

	def loss(self,inputs,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip):
		lstm = self.get_lstm(inputs)
		pir = self.get_pir(lstm)
		pis = self.get_pis(lstm)
		pim = self.get_pim(lstm)

		prob_f = lambda t,idx : pir[:,idx] if t_i%2!=0 else pis[:,idx]+pim[:,idx]
		lg_prob_new = [prob_f(t,i) for t,i in enumerate(a_taken)]

		rt = tf.math.exp(newlgprob - lg_prob_old)
		pg_losses1 = -GAE * rt
                pg_losses2 = -GAE * np.clip(rt,1-clip,1+clip)
                pg_loss = tf.math.reduce_mean(tf.math.maximum(pg_losses1,pg_losses2))




