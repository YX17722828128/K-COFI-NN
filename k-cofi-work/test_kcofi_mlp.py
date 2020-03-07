import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

class parameters(object):
	"""docstring for parameters"""
	def __init__(self):
		super(parameters, self).__init__()
		''' parameters for mlp model'''
		self.batch_size = 128		
		self.learning_rate = 0.0005
		self.layers = [64, 32, 16, 8]
		self.max_epoch_number = 150
		self.negative_ratio = 3
		self.item_set_size = 1
		self.verbose = 1 # Show performance per X iterations
		self.patience = 20 # assume that there are 100 iterations, if it can not achieve better performance for consecutive 50 times, iteration process will exit
		
		''' parameters for mlp data'''
		self.train_data = 'ML100K/copy1.train'
		self.test_data = 'ML100K/copy1.test'
		self.n = 943
		self.m = 1682
		self.embedding_size = 20
		self.top_k = 5
		self.rating_types_num = 5
		self.k_gram = [1]
		self.W_embedding_size = 0

		for gram in self.k_gram:
			self.W_embedding_size += (self.rating_types_num + 1 - gram)

parameters = parameters()

class KCOFI_MLP(object):
	"""docstring for KCOFI_MLP"""
	def __init__(self, parameters):
		super(KCOFI_MLP, self).__init__()
		all_gram_data = {'1_gram':{1:{1:[1,2,3,4], 2:[5,6,7,8]}, 2:{1:[9,10,11,12], 2:[13,14]}}}
		# all_gram_data = {'1_gram':[[[],[],[],[],[],[]], [[],[1,2,3,4],[5,6,7,8],[],[],[]], [[],[9,10,11,12],[13,14],[],[],[]]]}
		with tf.name_scope('input'):
			self.user = tf.placeholder(tf.int32, shape = [None, 1]) # "None" means the "batch size" and can be any number, "1" is the "size of data"
			self.item = tf.placeholder(tf.int32, shape = [None, 1])
			for gram in parameters.k_gram:
				setattr(self, str(gram)+'_gram_item', tf.placeholder(tf.int32, shape = [None, 1]))
				setattr(self, str(gram)+'_gram_factor', tf.placeholder(tf.float32, shape = [None, 1]))

		with tf.name_scope('embeddings_weights'):
        	# "truncated_normal" generate normal distribution random numbers according to the dimension "shape"(eg: [1683, 20]), "mean" is their mean value and "stddev" is their standard deviation
			# shape = (?, 1, layers[0]/2), the reason it becomes half is that the following "concat" operations will make "ui_concat" has shape(?, 1, layers[0])
			self.user_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.n+1, int(parameters.layers[0]/2)], mean = 0.0, stddev = 0.01), name = 'user_embeddings', dtype = tf.float32)
			self.item_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.m+1, int(parameters.layers[0]/2)-parameters.W_embedding_size], mean = 0.0, stddev = 0.01), name = 'item_embeddings', dtype = tf.float32)
			self.gram_item_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.m+1, parameters.W_embedding_size], mean = 0.0, stddev = 0.01), name = 'gram_item_embeddings', dtype = tf.float32)
			self.hidden_layer_W = [] # different "W" value reflect to different layer according to their index(layer position)
			self.hidden_layer_b = [] # different "b" value reflect to different layer according to their index(layer position)
			for index in range(len(parameters.layers)-1):
				# all W-related layers are hidden layers. W_1:shape(64, 32); W_2:shape(32, 16); W_3:shape(16, 8);
				self.hidden_layer_W.append(tf.Variable(tf.truncated_normal(shape = [parameters.layers[index], parameters.layers[index+1]], mean = 0.0, stddev = 0.01), name = 'hidden_layer_W_'+str(index+1), dtype = tf.float32))
				self.hidden_layer_b.append(tf.Variable(tf.zeros(parameters.layers[index+1]), name = 'hidden_layer_b_'+str(index+1))) # [0.0, 0.0, 0.0]
		
			self.h = tf.Variable(tf.truncated_normal(shape = [parameters.layers[-1], 1], mean = 0.0, stddev = 0.01), name = 'h_weights', dtype = tf.float32) # h:shape(8, 1)
			self.bias = tf.Variable(tf.zeros(1), name = 'bias') # 0.0

		self.kcofi_mlp_model()

	def kcofi_mlp_model(self):
		with tf.name_scope('model'):
			self.embedding_user = tf.nn.embedding_lookup(self.user_embeddings, self.user) # "embedding_lookup" gets the "element"(tensor with specific shape, each one here is [20, 1]) corresponding to "index"(here the index is user)
			self.embedding_item = tf.nn.embedding_lookup(self.item_embeddings, self.item) # the form is like [ [[...]] , [[...]] ]
			self.embedding_total_gram_items = tf.zeros([1, parameters.W_embedding_size])
			for gram in parameters.k_gram:
				self.embedding_gram_items = tf.divide(tf.reduce_sum(tf.nn.embedding_lookup(self.gram_item_embeddings, getattr(self, str(gram)+'_gram_item')), 0), getattr(self, str(gram)+'_gram_factor'))
				self.embedding_total_gram_items = tf.add(self.embedding_total_gram_items, self.embedding_gram_items)
			
		with tf.Session(config = tf_config) as sess:
			sess.run(tf.global_variables_initializer())
			input_dict = {}
			input_dict[self.user] = [[1]]
			input_dict[self.item] = [[1]]
			for gram in parameters.k_gram:
				input_dict[getattr(self, str(gram)+'_gram_item')] = [[1],[2],[3]]
				input_dict[getattr(self, str(gram)+'_gram_factor')] = [[2.0]]
			print(sess.run(self.embedding_gram_items, feed_dict = input_dict))
			print(sess.run(self.embedding_total_gram_items, feed_dict = input_dict))
			# print(sess.run(self.embedding_gram_item, feed_dict = {self.gram_item : [[1],[2],[3]]}))
			# print(sess.run(tf.reduce_sum(self.embedding_gram_item, 0), feed_dict = {self.user : [[1]], self.item : [[1]], self.gram_item : [[1],[2],[3]]}))
			# print(sess.run(tf.divide(tf.reduce_sum(self.embedding_gram_item, 0), 2.0), feed_dict = {self.user : [[1]], self.item : [[1]], self.gram_item : [[1],[2],[3]]}))


KCOFI_MLP = KCOFI_MLP(parameters)