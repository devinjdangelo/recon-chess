import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, LSTM, Reshape, Masking
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class ReconChessNet(Model):
	#Implements Tensorflow NN for ReconBot
	def __init__(self):
		super(ReconChessNet, self).__init__()
		self.mask = Masking(mask_value=0)
		self.convshape = Reshape((13,8,8))
		#channels first should work on GPU but not cpu for testing
		self.conv1 = Conv2D(16, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last')
		self.conv2 = Conv2D(32, 2, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last')
		self.conv3 = Conv2D(64, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last')
		self.conv4 = Conv2D(128, 1, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last')
		self.flatten = Flatten(data_format='channels_first')
		self.lstm = LSTM(256,stateful=True)

		self.pir = Dense(6*6)

		self.pim = Dense(8*8*8*8)

		self.v = Dense(1)

	def call(self,x):
		pass

	def get_lstm(self, x):
		#compute down to the intermediate lstm layer
		batch_size=len(x)
		x = pad_sequences(x,padding='post')
		x = x.reshape((batch_size,-1,13,8,8))
		time_steps = x.shape[1]
		x = tf.cast(x,tf.float32) 
		x = self.mask(x)
		x = self.convshape(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.flatten(x)
		x = tf.reshape(x,shape=(batch_size,time_steps,-1))
		x = self.lstm(x)
		return x

	def get_pir(self,lstm,mask):
		#compute recon policy
		x = self.pir(lstm)
		x = tf.keras.activations.softmax(x)
		x = tf.reshape(x,shape=(-1,6,6))
		x = x * mask
		x = x / tf.math.reduce_sum(x)
		return x

	def get_pim(self,lstm,mask):
		#compute piece movement policy
		x = self.pim(lstm)
		x = tf.keras.activations.softmax(x)
		x = tf.reshape(x,shape=(-1,8,8,8,8))
		x = x * mask
		x = x / tf.math.reduce_sum(x)
		return x

	def get_v(self,lstm):
		x = self.v(lstm)
		return x

	def loss(self,inputs,masks,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip):
		lstm = self.get_lstm(inputs)
		pir = self.get_pir(lstm,masks[0])
		pim = self.get_pim(lstm,masks[1])
		vpred = self.get_v(lstm)

		prob_f = lambda t,idx : tf.math.log(pir[t,idx[0],idx[1]]) if t%2==0 else tf.math.log(pim[t,idx[0],idx[1],idx[2],idx[3]])
		lg_prob_new = tf.stack([prob_f(t,i) for t,i in enumerate(a_taken)])

		rt = tf.math.exp(lg_prob_new - lg_prob_old)
		pg_losses1 = -GAE * rt
		pg_losses2 = -GAE * tf.clip_by_value(rt,1-clip,1+clip)
		pg_loss = tf.math.reduce_mean(tf.math.maximum(pg_losses1,pg_losses2))

		vpredclipped = old_v_pred + tf.clip_by_value(vpred-old_v_pred,-clip,clip)
		vf_losses1 = tf.square(vpred - returns)
		vf_losses2 = tf.square(vpredclipped - returns)

		vf_loss = 0.5 * tf.math.reduce_mean(tf.math.maximum(vf_losses1,vf_losses2))

		
		#want to count p=0 as 0 entropy (certain that it won't be picked)
		#so set p=0 -> p=1 so ln(p) = 0 
		pir_e = pir + tf.cast(tf.math.logical_not(masks[0]),tf.float32)
		pim_e = pim + tf.cast(tf.math.logical_not(masks[1]),tf.float32)
		e_f = lambda x : -tf.reduce_sum(x*tf.math.log(x))
		entropy = e_f(pir_e) + e_f(pim_e)

		loss = pg_loss - entropy + vf_loss

		return loss,pg_loss,entropy,vf_loss


if __name__=="__main__":
	from ReconBot import ReconBot
	agent = ReconBot()
	starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	agent.obs = agent._fen_to_obs(starting_fen)
	net = ReconChessNet()

	masks = []
	masks.append(np.random.randint(0,high=2,size=(2,6,6),dtype=np.int32))
	masks.append(np.random.randint(0,high=2,size=(2,8,8,8,8),dtype=np.int32))

	masks[0][0,3,3] = 1
	masks[1][0,1,1,2,2] = 1

	lg_prob_old = np.array([-0.1625,-0.6934],dtype=np.float32) #0.85, 0.5
	a_taken = [(3,3),(1,1,2,2)]

	GAE = np.array([0.5,-0.5],dtype=np.float32)
	old_v_pred = np.array([0.75,0.7],dtype=np.float32)
	returns = GAE + old_v_pred

	clip = 0.2

	loss = net.loss([agent.obs,agent.obs],masks,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip)
	loss = [l.numpy() for l in loss]
	print(loss)
