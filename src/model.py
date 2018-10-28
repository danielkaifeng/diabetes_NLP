
import tensorflow as tf

import numpy as np


def my_sigmoid_loss(labels, logits):
		#relu_logits = tf.nn.relu(logits)
		#neg_abs_logits = -tf.abs(logits)		
		#res = tf.add(relu_logits - logits * labels, tf.log1p(tf.exp(neg_abs_logits)))

		gamma = 1.15 *labels - 0.15
		#gamma = 2 *labels - 1
		res = 1 - tf.log1p( gamma*logits/(1+ tf.abs(logits)) )
		return res

def leaky_relu(x, leak=0.001, name='leaky_relu'):
		return tf.maximum(x, x * leak, name=name)

def RNN_net(net, num_layers=1):
		cell = []
		hidden_size = 64 
		for i in range(num_layers):
				cell.append(tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size))
				#cell.append(tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True))

		cell = tf.nn.rnn_cell.MultiRNNCell(cell)
		#cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = (1-dropout))

		#initial_state = cell.zero_state(batch_size, tf.float32)

		input_list = tf.unstack(net, axis=1)

		rnn_output, _ = tf.nn.static_rnn(cell, input_list, dtype=tf.float32)
		rnn_output = rnn_output[-1]

		w1 = tf.get_variable("w1", [hidden_size, 128])
		b1 = tf.get_variable("b1", [128])
		feature = tf.nn.xw_plus_b(rnn_output, w1, b1)

		logits = tf.layers.dense(feature, units=5)
		return logits




def batch_norm(x, phase_train, scope='bn'):
	n_out = x.get_shape().as_list()[-1]
	with tf.variable_scope(scope):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
									 name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
									  name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train, mean_var_with_update,
							lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed



class build_network():
	def __init__(self):
		self.x = tf.placeholder(tf.int32, shape=[None, 47880], name='x')
		self.xn = tf.placeholder(tf.float32, shape=[None, 217], name='xn')
		self.labels = tf.placeholder(tf.float32, [None, 5], name='y')
		self.is_train = tf.placeholder(tf.bool, name='is_train')

		self.dropout = tf.placeholder(tf.float32, name='dropout')

		with tf.name_scope("cal_loss") as scope:
				logits = self.network(self.x, self.dropout, self.is_train, self.xn) 
				
				#base = np.array([126,78,1.6,1.3,2.7])
				base = np.array([120, 70, 2, 2, 3])

				top = np.array([260,130,30, 6, 15])
				low = np.array([60, 30, 0.05, 0.02, 0.02])
				self.mid = logits

				logits = tf.nn.relu(logits)
				logits += low
	
				#logits = tf.maximum(logits, low)
				self.logits = tf.minimum(logits, top)

				opt = tf.train.AdamOptimizer(0.001)
				#opt2 = tf.train.AdamOptimizer(0.01)

				alpha = np.array([100, 100, 1, 1, 1])
				#x = tf.clip_by_value(self.labels / alpha, 1e-4, 1 - 1e-4)
				#y = tf.clip_by_value(self.logits / alpha, 1e-4, 1 - 1e-4)
				#x = self.logits / alpha
				#y = self.labels / alpha
				x = self.logits
				y = self.labels

				#mse = tf.sqrt(tf.reduce_mean(tf.square(self.mid)))
				self.loss2 = tf.reduce_mean(tf.squared_difference(x, y))
				#self.acc2 = tf.reduce_mean(np.array([1,1,2,2,1]) * tf.square(tf.log1p(logits) - tf.log1p(self.labels)))
				self.loss = tf.reduce_mean(tf.square(tf.log1p(logits) - tf.log1p(self.labels)))

				#tf.losses.mean_squared_error(labels, predictions)
				self.optimizer = opt.minimize(self.loss)

				emb_var_list = [v for v in tf.global_variables()
				                    if v.name == "cal_loss/embedding_layer/emb:0"]

				print emb_var_list
				#self.optimizer2 = opt2.minimize(self.loss, var_list=emb_var_list)
		
		
	def network(self, x, dropout, is_train, xn):
		kernel_size = [11, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
		kernel_size2 = [7, 5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
		filters = [32, 32, 16, 16, 8, 8, 16, 16, 12, 12, 12, 12]

		vocabulary_size = 180
		embedding_size = 64

		with tf.variable_scope("embedding_layer") as scope:
			embeddings = tf.Variable(
			    tf.random_uniform([5193, embedding_size], -1.0, 1.0), name='emb')

		self.vec = embeddings
		emb = tf.nn.embedding_lookup(embeddings, x)

		emb = tf.reshape(emb, [-1, 266, vocabulary_size, embedding_size])
		emb2 = tf.transpose(emb, [0,2,1,3])
		emb2 = tf.reshape(emb2, [-1, vocabulary_size, 266*embedding_size])
		emb2 = tf.layers.dense(emb2, 10*embedding_size)
		emb2 = tf.layers.dense(emb2, embedding_size)
		emb2 = tf.layers.dense(emb2, embedding_size)
		emb2 = tf.reshape(emb2, [-1, vocabulary_size * embedding_size])

		net = emb
		print emb.get_shape()
		
		block_num = 8
		for i in range(block_num):
			#std = np.sqrt(2/(filters[i] * 266 * 180/(1+i/2)))
			#std = np.sqrt(2/4096 /(1+i/2))
			std = 0.001

			#with tf.name_scope("conv_block_%d" % i) as scope:
			with tf.name_scope("conv_block") as scope:
				conv1 = tf.layers.conv2d(inputs=net, filters=filters[i], 
											kernel_size=(1, 11),padding="same",  
											activation=leaky_relu, kernel_initializer= tf.truncated_normal_initializer(stddev=std, dtype=tf.float32))

				conv2 = tf.layers.conv2d(inputs=net, filters=filters[i], 
											kernel_size=(1,7), padding="same", 
											activation=leaky_relu, kernel_initializer= tf.truncated_normal_initializer(stddev=std, dtype=tf.float32))

				conv3 = tf.layers.conv2d(inputs=net, filters=filters[i], 
											kernel_size=(1,5) , padding="same", 
											activation=leaky_relu, kernel_initializer= tf.truncated_normal_initializer(stddev=std, dtype=tf.float32))

				conv = tf.concat(axis=3, values=[conv1, conv2, conv3])

				conv = tf.layers.conv2d(inputs=conv, filters=filters[i], 
											kernel_size=(1,5), padding="same", dilation_rate=4, 
											activation=leaky_relu, kernel_initializer= tf.truncated_normal_initializer(stddev=std, dtype=tf.float32))

			
				conv = batch_norm(conv, is_train, scope='bn%d' % i)
				conv = tf.layers.dropout(inputs=conv, rate=dropout)

				#net = tf.layers.max_pooling1d(inputs=net, pool_size=2, strides=1)
				#net = tf.layers.dense(net, filters[i])
				net = conv
				
				if i % 2 == 1:
					net = tf.layers.max_pooling2d(inputs=conv, pool_size=(2,2), strides=(2,2))

		#net = tf.reshape(net, [-1, seq_len/(2**(block_num/1)), filters[i]])
		net = tf.reshape(net, [-1, (vocabulary_size/2**4) * (266/2**4) * filters[i]])

		#logits = RNN_net(emb2)
		dense = tf.layers.dense(emb2, units=64)

		xn = tf.layers.dense(xn, 128)
		xn = tf.layers.dense(xn, 128)
		xn = tf.layers.dense(xn, 64)
		dense = tf.concat(axis=1, values=[dense,xn])

		logits = tf.layers.dense(dense, units=5)

		"""	
		#global_pooling = tf.reduce_mean(conv, (1))
		#x1 = tf.layers.dense(x[:,0:159], 8)
		#x2 = tf.layers.dense(x[:,159:458], 8)

		#x1 = tf.expand_dims(x1,2)
		#x2 = tf.expand_dims(x2,2)
		#x = tf.concat([x1,x2], 2)
		for n in range(2):
			conv1 = tf.layers.conv2d(x, filters = 16, kernel_size=11, padding="same", activation=tf.nn.relu)
			conv2 = tf.layers.conv2d(x, filters = 16, kernel_size=7, padding="same", activation=tf.nn.relu)
			conv3 = tf.layers.conv2d(x, filters = 8, kernel_size=1, padding="same", activation=tf.nn.relu)
			x = tf.concat([conv1, conv2, conv3], 2)
			x = tf.layers.max_pooling1d(x, pool_size=2, strides=1, padding="same")
			x = batch_norm(x, is_train, scope='bn_%d' % n)

			x = tf.nn.relu(x)
			x = tf.layers.dropout(x, rate=self.dropout)
		"""
	
		return logits

