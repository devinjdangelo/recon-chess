import tensorflow as tf
from math import ceil
import nvidia_smi

from tensorflow.keras.layers import Dense, Flatten, Conv2D, LSTM, Reshape, Masking
from tensorflow.keras import Model
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

def gpu_usage():
	nvidia_smi.nvmlInit()

	handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
	# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

	info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
	usage = info.used

	nvidia_smi.nvmlShutdown()
	return usage


class ReconChessNet(Model):
	#Implements Tensorflow NN for ReconBot
	def __init__(self,name,max_batch_size,learning_rate,use_cpu):

		self.use_cpu = use_cpu

		if self.use_cpu:
			tf.config.set_visible_devices([], 'GPU')

		self.entropy_weight = .08

		self.netname = name
		self.max_batch_size = max_batch_size

		super(ReconChessNet, self).__init__()
		#self.mask = Masking(mask_value=0)
		#self.convshape = Reshape((13,8,8))
		#channels first should work on GPU but not cpu for testing
		#if not use_cpu:
		if not self.use_cpu:
			self.conv1 = Conv2D(16, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_first',name='conv1')
			self.conv2 = Conv2D(32, 2, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_first',name='conv2')
			self.conv3 = Conv2D(64, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_first',name='conv3')
			self.conv4 = Conv2D(128, 1, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_first',name='conv4')
		else:
			self.conv1 = Conv2D(16, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last',name='conv1')
			self.conv2 = Conv2D(32, 2, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last',name='conv2')
			self.conv3 = Conv2D(64, 2, strides=(2,2), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last',name='conv3')
			self.conv4 = Conv2D(128, 1, strides=(1,1), activation=tf.nn.leaky_relu, kernel_initializer=TruncatedNormal,data_format='channels_last',name='conv4')

		self.flatten = Flatten(data_format='channels_first',name='flatten')
		#stateful is conveinient for sequential action sampling,
		#but problematic because it requires fixed batch size
		#could maintain two layers (one for action and one for training)
		#and copy weights between, or could just not use stateful and
		#manually keep track of state.
		self.lstm_stateful = LSTM(256,stateful=True,name='stateful_lstm',trainable=False)
		self.lstm = LSTM(256,return_sequences=True,name='training_lstm')

		self.pir = Dense(6*6,name='pir')

		self.pim = Dense(8*8*8*8,name='pim')

		self.v = Dense(1,name='v')

		self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

		# print('initialized ',name,' usage ',gpu_usage())

	def call(self,x):
		pass

	def get_lstm(self, x, inference=True,mask=None):

		batch_size=len(x)
		x = pad_sequences(x,padding='post')

		x = tf.reshape(x,(batch_size,-1,13,8,8))
		time_steps = x.shape[1]
		x = tf.reshape(x,shape=(batch_size*time_steps,13,8,8))
		#convert to channels last for cpu
		if self.use_cpu:
			x = tf.transpose(x,[0,2,3,1])
		x = tf.cast(x,tf.float32) 
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.flatten(x)
		x = tf.reshape(x,shape=(batch_size,time_steps,-1))
		if inference:
			x = self.lstm_stateful(x)
		else:
			mask = tf.reshape(mask,(batch_size,time_steps))
			mask = tf.cast(mask,tf.bool)
			x = self.lstm(x,mask=mask)

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
		#if self.netname=='train':
			#print(mask.shape)
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

		lg_prob_old = tf.reshape(pad_sequences(lg_prob_old,padding='post',dtype=np.float32),(-1,))
		#vector reporting 1 for real data and 0 for padded 0's
		actual_timesteps = tf.where(lg_prob_old<0,tf.ones_like(lg_prob_old),tf.zeros_like(lg_prob_old))
		a_taken = tf.reshape(pad_sequences(a_taken,padding='post'),(-1,))

		GAE = tf.reshape(pad_sequences(GAE,padding='post',dtype=np.float32),(-1,))

		old_v_pred = tf.reshape(pad_sequences(old_v_pred,padding='post',dtype=np.float32),(-1,))
		returns = tf.reshape(pad_sequences(returns,padding='post',dtype=np.float32),(-1,))
		mask_padded = pad_sequences(mask,padding='post')
		mask_padded = tf.cast(tf.reshape(mask_padded,(-1,4096)),tf.float32)
		# print(mask_padded.shape,old_v_pred.shape,len(inputs))

		lstm = self.get_lstm(inputs,inference=False,mask=actual_timesteps)

		#every other starting from first
		lstm_pir = lstm[:,::2,:]
		#every other starting from second
		lstm_pim = lstm[:,1::2,:]
		
		lstm_pir = tf.reshape(lstm_pir,(-1,256))
		lstm_pim = tf.reshape(lstm_pim,(-1,256))
		lstm = tf.reshape(lstm,(-1,256))	

		pir = self.get_pir(lstm_pir)
		pir = pir*tf.expand_dims(actual_timesteps[::2],axis=-1)
		pim = self.get_pim(lstm_pim,mask_padded)

		a_taken_pir = a_taken[::2]
		a_taken_pim = a_taken[1::2]

		pir_taken = tf.stack([pir[i,j] for i,j in enumerate(a_taken_pir)])
		pim_taken = tf.stack([pim[i,j] for i,j in enumerate(a_taken_pim)])

		pir_taken = tf.expand_dims(pir_taken,-1)
		pim_taken = tf.expand_dims(pim_taken,-1)
		lg_prob_new = tf.concat([pir_taken,pim_taken],-1)
		lg_prob_new = tf.reshape(lg_prob_new,(-1,))
		lg_prob_new = tf.where(actual_timesteps>0,lg_prob_new,tf.ones_like(lg_prob_new))
		lg_prob_new = tf.math.log(lg_prob_new)
		vpred = self.get_v(lstm)

		
		rt = tf.math.exp(lg_prob_new - lg_prob_old)
		pg_losses1 = -GAE * rt
		pg_losses2 = -GAE * tf.clip_by_value(rt,1-clip,1+clip)

		pg_loss = tf.math.reduce_mean(tf.math.maximum(pg_losses1,pg_losses2))


		vpredclipped = old_v_pred + tf.clip_by_value(vpred-old_v_pred,-clip,clip)
		# print(gpu_usage())
		vf_losses1 = tf.square(vpred - returns)
		vf_losses2 = tf.square(vpredclipped - returns)

		vf_loss = 0.5 * tf.math.reduce_mean(tf.math.maximum(vf_losses1,vf_losses2))

		#want to count p=0 as 0 entropy (certain that it won't be picked)
		#so set p=0 -> p=1 so ln(p) = 0 
		
		mask_padded = tf.cast(mask_padded,dtype=tf.bool)

		pim_e = pim + tf.cast(tf.math.logical_not(mask_padded),tf.float32)
		pir_e = tf.where(pir>0,pir,tf.ones_like(pir))

		e_f = lambda x : tf.reduce_mean(-tf.reduce_sum(x*tf.math.log(x),axis=1))

		entropy = e_f(pir_e) + e_f(pim_e)

		loss = pg_loss - self.entropy_weight*entropy + vf_loss
		#loss = pg_loss + vf_loss

		return loss,pg_loss,entropy,vf_loss

	def grad(self,inputs,mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip):
		with tf.GradientTape() as tape:
			loss,pg_loss,entropy,vf_loss = self.loss(inputs,mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip)
		return loss,pg_loss,entropy,vf_loss,tape.gradient(loss,self.trainable_variables)

	def update(self,inputs,mask,lg_prob_old,a_taken,GAE,old_v_pred,returns,clip):

		total_batch_size = len(inputs)
		n_iters = ceil(total_batch_size/self.max_batch_size)
		batch_size_per_iter = self.max_batch_size

		accumulate_gradients = []
		accumulate_loss = []
		accumulate_pg_loss = []
		accumulate_entropy = []
		accumulate_vf_loss = []

		for i in range(n_iters):
			# print('iter ',i,' memory usage: ',gpu_usage())
			#EAGER EXECUTION BUG MEMORY LEAK
			#https://github.com/tensorflow/tensorflow/issues/19671
			#set seed is workarond
			tf.random.set_seed(1)
			if i==n_iters-1:
				start = i*batch_size_per_iter
				end = total_batch_size
			else:
				start = i*batch_size_per_iter
				end = (i+1)*batch_size_per_iter

			# print('sending ',end-start,' games to gpu ')

			
			loss,pg_loss,entropy,vf_loss,grads = self.grad(inputs[start:end],mask[start:end],lg_prob_old[start:end],
															a_taken[start:end],GAE[start:end],old_v_pred[start:end],
															returns[start:end],clip)

			weight = (end-start) / total_batch_size
			accumulate_loss.append(loss.numpy()*weight)
			accumulate_pg_loss.append(pg_loss.numpy()*weight)
			accumulate_entropy.append(entropy.numpy()*weight)
			accumulate_vf_loss.append(vf_loss.numpy()*weight)

			accumulate_gradients.append([g*weight for g in grads])
			if i>0:
				accumulate_gradients = [[t[0] + t[1] for t in zip(accumulate_gradients[0],accumulate_gradients[1])]]

		assert len(accumulate_gradients)==1
		grads = accumulate_gradients[0]
		grads,g_n = tf.clip_by_global_norm(grads,0.5)
		self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

		loss,pg_loss,entropy,vf_loss = np.sum(accumulate_loss),np.sum(accumulate_pg_loss), \
										np.sum(accumulate_entropy),np.sum(accumulate_vf_loss)


		return loss,pg_loss,entropy,vf_loss,g_n.numpy()

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

	def sample_final_v(self,x):
		lstm = self.get_lstm(x)
		v = self.get_v(lstm)

		return v.numpy()[:,0]


