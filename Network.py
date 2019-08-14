import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, LSTM, Reshape, Masking
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

class ReconChessNet(Model):
	#Implements Tensorflow NN for ReconBot
	def __init__(self,name):
		self.netname = name

		super(ReconChessNet, self).__init__()
		self.mask = Masking(mask_value=0)
		#self.convshape = Reshape((13,8,8))
		#channels first should work on GPU but not cpu for testing
		self.conv1 = Conv2D(16, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last')
		self.conv2 = Conv2D(32, 2, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last')
		self.conv3 = Conv2D(64, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last')
		self.conv4 = Conv2D(128, 1, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last')
		self.flatten = Flatten(data_format='channels_first')
		#stateful is conveinient for sequential action sampling,
		#but problematic because it requires fixed batch size
		#could maintain two layers (one for action and one for training)
		#and copy weights between, or could just not use stateful and
		#manually keep track of state.
		self.lstm_stateful = LSTM(256,stateful=True,name='stateful_lstm',trainable=False)
		self.lstm = LSTM(256,return_sequences=True,name='training_lstm')

		self.pir = Dense(6*6)

		self.pim = Dense(8*8*8*8)

		self.v = Dense(1)

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

	def call(self,x):
		pass

	def get_lstm(self, x, inference=True):

		batch_size=len(x)
		x = pad_sequences(x,padding='post')

		x = tf.reshape(x,(batch_size,-1,13,8,8))
		time_steps = x.shape[1]
		x = tf.reshape(x,shape=(batch_size*time_steps,13,8,8))
		#print(tf.reduce_mean(x,axis=[1,2,3],keepdims=True).numpy())
		x = tf.cast(x,tf.float32) 
		x = self.mask(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.flatten(x)
		x = tf.reshape(x,shape=(batch_size,time_steps,-1))
		if inference:
			x = self.lstm_stateful(x)
		else:
			x = self.lstm(x)

		return x

	def get_pir(self,lstm):
		#compute recon policy
		x = self.pir(lstm)
		x = tf.keras.activations.softmax(x)
		x = tf.reshape(x,(-1,36))
		#x = tf.reshape(x,shape=(-1,6,6))
		return x

	def get_pim(self,lstm,mask):
		#compute piece movement policy
		x = self.pim(lstm)
		x = tf.keras.activations.softmax(x)
		x = x * mask
		if self.netname=='train':
			print(mask.shape)
			#print(tf.reduce_sum(mask,axis=1,keepdims=True).numpy())
		x = tf.reshape(x,(-1,4096))
		scale = tf.math.reduce_sum(x,axis=1,keepdims=True)
		#if self.netname=='train':
			#print(scale.numpy())
		#don't rescale masked time steps, otherwise div by 0
		scale = tf.where(scale>0,scale,tf.ones_like(scale))
		x = x / scale
		#x = tf.reshape(x,shape=(-1,8,8,8,8))
		return x

	def get_v(self,lstm):
		x = self.v(lstm)
		return x

	def loss(self,inputs,mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip):
		lstm = self.get_lstm(inputs,inference=False)
		lstm = tf.reshape(lstm,(-1,256))
		pir = self.get_pir(lstm)
		mask_padded = pad_sequences(mask,padding='post')
		print(len(mask),[m.shape for m in mask],mask_padded.shape)
		mask_padded = tf.cast(tf.reshape(mask_padded,(-1,4096)),tf.float32)
		pim = self.get_pim(lstm,mask_padded)
		vpred = self.get_v(lstm)

		lg_prob_old = tf.reshape(pad_sequences(lg_prob_old,padding='post',dtype=np.float32),(-1,))
		a_taken = tf.reshape(pad_sequences(a_taken,padding='post'),(-1,))
		GAE = tf.reshape(pad_sequences(GAE,padding='post',dtype=np.float32),(-1,))
		old_v_pred = tf.reshape(pad_sequences(old_v_pred,padding='post',dtype=np.float32),(-1,))
		returns = tf.reshape(pad_sequences(returns,padding='post',dtype=np.float32),(-1,))

		GAE = (GAE - tf.math.reduce_mean(GAE)) / (tf.math.reduce_std(GAE)+1e-8)

		prob_f = lambda t,idx : tf.math.log(pir[t,idx]) if t%2==0 else tf.math.log(pim[t,idx])
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
		
		mask_padded = tf.cast(mask_padded,dtype=tf.bool)
		pim_e = pim + tf.cast(tf.math.logical_not(mask_padded),tf.float32)
		e_f = lambda x : -tf.reduce_sum(x*tf.math.log(x))
		entropy = e_f(pir) + e_f(pim_e)

		loss = pg_loss - entropy + vf_loss

		return loss,pg_loss,entropy,vf_loss

	def grad(self,inputs,mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip):
		with tf.GradientTape() as tape:
			loss,pg_loss,entropy,vf_loss = self.loss(inputs,mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip)
		return loss,pg_loss,entropy,vf_loss,tape.gradient(loss,self.trainable_variables)

	def update(self,inputs,mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip):
		loss,pg_loss,entropy,vf_loss,grads = self.grad(inputs,mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip)
		grads,g_n = tf.clip_by_global_norm(grads,0.5)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

		return loss.numpy(),loss.numpy(),loss.numpy(),loss.numpy(),loss.numpy()

	def sample_pir(self,x):
		#return recon actions given batch of obs
		lstm = self.get_lstm(x)
		x = self.get_pir(lstm)
		x = tf.reshape(x,[-1,36])
		x = tf.math.log(x)
		actions = tf.random.categorical(x,1).numpy()[:,0]
		probs = [x[i,action].numpy() for i,action in enumerate(actions)]

		value = self.get_v(lstm).numpy()[:,0]
		return actions,probs,value

	def sample_pim(self,x,mask):
		#return move actions given batch of obs
		lstm = self.get_lstm(x)
		x = self.get_pim(lstm,mask)
		x = tf.reshape(x,[-1,8*8*8*8])
		x = tf.math.log(x)
		actions = tf.random.categorical(x,1).numpy()[:,0]
		probs = [x[i,action].numpy() for i,action in enumerate(actions)]

		value = self.get_v(lstm).numpy()[:,0]
		return actions,probs,value



if __name__=="__main__":
	from ReconBot import ReconBot
	agent = ReconBot()
	starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	agent.obs = agent._fen_to_obs(starting_fen)
	net = ReconChessNet()

	for i in range(1000):
		mask = np.random.randint(0,high=2,size=(2,8,8,8,8),dtype=np.int32)
		mask = np.reshape(mask,(2,-1))

		lg_prob_old = np.array([[-0.1625],[-0.6934]],dtype=np.float32) #0.85, 0.5
		a_taken = [[random.randint(0,35)],[random.randint(0,4095)]]
		mask[:,a_taken[1]]=1

		GAE = np.array([[0.5],[-0.5]],dtype=np.float32)
		old_v_pred = np.array([[0.75],[0.7]],dtype=np.float32)
		returns = GAE + old_v_pred

		clip = 0.2

		loss = net.update([agent.obs,agent.obs],mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip)
		loss = [l.numpy() for l in loss]
		print(loss)

