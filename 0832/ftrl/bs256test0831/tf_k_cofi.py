import random, os, math, copy
import tensorflow as tf
from time import time
import numpy as np
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

class data_set(object):
	"""docstring for data_set"""
	def __init__(self, train_data, test_data, k_gram, rating_types_num, m, n):
		super(data_set, self).__init__()
		self.train_pairs, self.train_dict, self.all_users, self.all_items, self.g_avg, self.bu, self.bi = self.read_train_data(train_data, rating_types_num, m, n) # 读取训练数据并算出初始的g_avg，biasU，biasV
		self.test_pairs, self.test_dict = self.read_test_data(test_data, rating_types_num) # 读取测试数据
		self.test_unobserved_dict = self.get_test_unobserved_dict() # 这个函数没用到，可以忽略
		self.u_gram_items_dict = {}
		'''
		key是user，value是二维列表[array0, array1, ...]，内部array的个数取决于有多少类划分，在k_gram算法中，若k=[1]，则有5个array；若k=[2]，则有4个array；若k=[1,2]，则有9个array
		若rating为{1,2,3,4,5}，k=1，则有5个array，array0由若干个item_id组成，表示用户user评分为1的所有物品
		'''
		self.u_gram_factors_dict = {} # 上述每个内部的array，对应着这里的一个normalization_factor
		self.max_inner_length = 0 # 上面每个用户，在每个评分上的物品个数是不同的，为了数据对齐，找到最多item的array并获得其包含的item个数，在所有缺失的数据后面补值为max_item(也就是物品数量m+1)
		self.all_gram_data = self.get_k_gram_data(train_data, k_gram, rating_types_num, m)	# 这个函数统计好上面的三个 self.u_gram_items_dict，self.u_gram_factors_dict，self.max_inner_length；这里返回的self.all_gram_data没有在函数外用到，可以忽略
		
	def read_train_data(self, train_data, rating_types_num, m, n):
		train_pairs = []
		train_dict = dict()
		all_users = set()
		all_items = set()
		train_num = 0
		rating_sum = 0
		user_rating_sum = [0.0]*(n+1)
		user_rating_num = [0]*(n+1)
		item_rating_sum = [0.0]*(m+1)
		item_rating_num = [0]*(m+1)
		bu = [0.0]*(n+1)
		bi = [0.0]*(m+1)
		for line in open(train_data, 'r', encoding = 'utf-8'):
			record = line.split()
			if rating_types_num == 5:
				user, item, rating = int(record[0]), int(record[1]), int(record[2])
			elif rating_types_num == 10:
				user, item, rating = int(record[0]), int(record[1]), int(float(record[2])*2)
			train_pairs.append([user, item, rating])
			if user not in train_dict:
				train_dict[user] = set()
				train_dict[user].add(item)
			else:
				train_dict[user].add(item)
			all_users.add(user)
			all_items.add(item)
			rating_sum += float(record[2])
			user_rating_sum[user] += float(record[2])
			item_rating_sum[item] += float(record[2])
			user_rating_num[user] += 1
			item_rating_num[item] += 1
			train_num += 1
		g_avg = rating_sum / train_num
		for u in range(1, n+1):
			if user_rating_num[u] > 0:
				bu[u] = (user_rating_sum[u] - g_avg * user_rating_num[u]) / user_rating_num[u]
		for i in range(1, m+1):
			if item_rating_num[i] > 0:
				bi[i] = (item_rating_sum[i] - g_avg * item_rating_num[i]) / item_rating_num[i]
		return train_pairs, train_dict, all_users, all_items, g_avg, bu, bi

	def read_test_data(self, test_data, rating_types_num):
		test_pairs = []
		test_dict = dict()
		for line in open(test_data, 'r', encoding = 'utf-8'):
			record = line.split()
			if rating_types_num == 5:
				user, item, rating = int(record[0]), int(record[1]), int(record[2])
			elif rating_types_num == 10:
				user, item, rating = int(record[0]), int(record[1]), int(float(record[2])*2)
			test_pairs.append([user, item, rating])
			if user not in test_dict:
				test_dict[user] = set()
				test_dict[user].add(item)
			else:
				test_dict[user].add(item)
		return test_pairs, test_dict

	def get_test_unobserved_dict(self):
		test_unobserved_dict = dict()
		for user in self.test_dict.keys():
			if user in self.train_dict.keys():
				observed_items = self.train_dict[user]
				unobserved_items = self.all_items - observed_items
				test_unobserved_dict[user] = unobserved_items
		return test_unobserved_dict

	def get_k_gram_data(self, train_data, k_gram, rating_types_num, m):
		all_gram_data = {}
		for gram in k_gram:
			all_gram_data[str(gram)+'_'+'gram'] = {}
		''' user_rating_id_set = {user_id_1:rating_id_set_1, user_id_2:rating_id_set_2, user_id_3:......} '''
		''' rating_id_set = {rating_1:id_set_1, rating_2:id_set_2, rating_3:......} '''
		''' this is the 1_gram type '''
		user_rating_id_set = {}
		if rating_types_num == 5:
			for line in open(train_data, 'r', encoding = 'utf-8'):
				record = line.split()
				user, item, rating = int(record[0]), int(record[1]), int(record[2])
				if user not in user_rating_id_set.keys():
					user_rating_id_set[user] = {rating : [item]}
				else:
					if rating in user_rating_id_set[user].keys():
						user_rating_id_set[user][rating].append(item)
					else:
						user_rating_id_set[user][rating] = [item]
		elif rating_types_num == 10:
			for line in open(train_data, 'r', encoding = 'utf-8'):
				record = line.split()
				user, item, rating = int(record[0]), int(record[1]), int(record[2])*2
				if user not in user_rating_id_set.keys():
					user_rating_id_set[user] = {rating : [item]}
				else:
					if rating in user_rating_id_set[user].keys():
						user_rating_id_set[user][rating].append(item)
					else:
						user_rating_id_set[user][rating] = [item]
		if 1 in k_gram:
			all_gram_data['1_gram'] = copy.deepcopy(user_rating_id_set)

		for gram in k_gram:
			if gram == 2:
				if rating_types_num == 5:
					for user in self.all_users:
						all_gram_data['2_gram'][user] = {}
						for rating in range(1, 5):
							all_gram_data['2_gram'][user][rating] = []
							if rating in user_rating_id_set[user].keys():
								all_gram_data['2_gram'][user][rating] += copy.deepcopy(user_rating_id_set[user][rating])
							if rating+1 in user_rating_id_set[user].keys():
								all_gram_data['2_gram'][user][rating] += copy.deepcopy(user_rating_id_set[user][rating+1])
				elif rating_types_num == 10:
					for user in self.all_users:
						all_gram_data['2_gram'][user] = {}
						for rating in range(1, 8, 2):
							all_gram_data['2_gram'][user][rating] = []
							if rating in user_rating_id_set[user].keys():
								all_gram_data['2_gram'][user][rating] += copy.deepcopy(user_rating_id_set[user][rating])
							if rating+1 in user_rating_id_set[user].keys():
								all_gram_data['2_gram'][user][rating] += copy.deepcopy(user_rating_id_set[user][rating+1])
							if rating+2 in user_rating_id_set[user].keys():
								all_gram_data['2_gram'][user][rating] += copy.deepcopy(user_rating_id_set[user][rating+2])
							if rating+3 in user_rating_id_set[user].keys():
								all_gram_data['2_gram'][user][rating] += copy.deepcopy(user_rating_id_set[user][rating+3])
			elif gram == 5:
				for user in self.all_users:
					all_gram_data['5_gram'][user] = []
					for rating in range(1, rating_types_num+1):
						if rating in user_rating_id_set[user].keys():
							all_gram_data['5_gram'][user] += copy.deepcopy(user_rating_id_set[user][rating])

		max_inner_length = 0
		if k_gram == [1]:
			for user in self.all_users:
				u_items_set = []
				u_factors_set = []
				for g in range(rating_types_num):
					if g+1 in user_rating_id_set[user].keys():
						items_set = all_gram_data['1_gram'][user][g+1]
						temp_inner_length = len(items_set)
						if max_inner_length < temp_inner_length:
							max_inner_length = temp_inner_length
						u_items_set.append(items_set)
						u_factors_set.append([temp_inner_length])
					else:
						u_items_set.append([m+1]*max_inner_length)
						u_factors_set.append([max_inner_length])
				self.u_gram_items_dict[user] = u_items_set
				self.u_gram_factors_dict[user] = u_factors_set
		elif k_gram == [2]:
			for user in self.all_users:
				u_items_set = []
				u_factors_set = []
				for g in range(rating_types_num-1):
					if g+1 in user_rating_id_set[user].keys():
						items_set = all_gram_data['2_gram'][user][g+1]
						temp_inner_length = len(items_set)
						if max_inner_length < temp_inner_length:
							max_inner_length = temp_inner_length
						u_items_set.append(items_set)
						u_factors_set.append([temp_inner_length])
					else:
						u_items_set.append([m+1]*max_inner_length)
						u_factors_set.append([max_inner_length])
				self.u_gram_items_dict[user] = u_items_set
				self.u_gram_factors_dict[user] = u_factors_set
		elif k_gram == [5]:
			for user in self.all_users:
				u_items_set = []
				u_factors_set = []
				for g in range(1):
					if g+1 in user_rating_id_set[user].keys():
						items_set = all_gram_data['5_gram'][user]
						temp_inner_length = len(items_set)
						if max_inner_length < temp_inner_length:
							max_inner_length = temp_inner_length
						u_items_set.append(items_set)
						u_factors_set.append([temp_inner_length])
					else:
						u_items_set.append([m+1]*max_inner_length)
						u_factors_set.append([max_inner_length])
				self.u_gram_items_dict[user] = u_items_set
				self.u_gram_factors_dict[user] = u_factors_set
		elif k_gram == [1, 2]:
			for user in self.all_users:
				u_items_set = []
				u_factors_set = []
				for g in range(rating_types_num):
					if g+1 in user_rating_id_set[user].keys():
						items_set = all_gram_data['1_gram'][user][g+1]
						temp_inner_length = len(items_set)
						if max_inner_length < temp_inner_length:
							max_inner_length = temp_inner_length
						u_items_set.append(items_set)
						u_factors_set.append([temp_inner_length])
					else:
						u_items_set.append([m+1]*max_inner_length)
						u_factors_set.append([max_inner_length])
				for g in range(rating_types_num-1):
					if g+1 in user_rating_id_set[user].keys():
						items_set = all_gram_data['2_gram'][user][g+1]
						temp_inner_length = len(items_set)
						if max_inner_length < temp_inner_length:
							max_inner_length = temp_inner_length
						u_items_set.append(items_set)
						u_factors_set.append([temp_inner_length])
					else:
						u_items_set.append([m+1]*max_inner_length)
						u_factors_set.append([max_inner_length])
				self.u_gram_items_dict[user] = u_items_set
				self.u_gram_factors_dict[user] = u_factors_set
		elif k_gram == [1, 5]:
			for user in self.all_users:
				u_items_set = []
				u_factors_set = []
				for g in range(rating_types_num):
					if g+1 in user_rating_id_set[user].keys():
						items_set = all_gram_data['1_gram'][user][g+1]
						temp_inner_length = len(items_set)
						if max_inner_length < temp_inner_length:
							max_inner_length = temp_inner_length
						u_items_set.append(items_set)
						u_factors_set.append([temp_inner_length])
					else:
						u_items_set.append([m+1]*max_inner_length)
						u_factors_set.append([max_inner_length])
				for g in range(1):
					if g+1 in user_rating_id_set[user].keys():
						items_set = all_gram_data['5_gram'][user]
						temp_inner_length = len(items_set)
						if max_inner_length < temp_inner_length:
							max_inner_length = temp_inner_length
						u_items_set.append(items_set)
						u_factors_set.append([temp_inner_length])
					else:
						u_items_set.append([m+1]*max_inner_length)
						u_factors_set.append([max_inner_length])
				self.u_gram_items_dict[user] = u_items_set
				self.u_gram_factors_dict[user] = u_factors_set
		self.max_inner_length = max_inner_length
		return all_gram_data	

class parameters(object):
	"""docstring for parameters"""
	def __init__(self, learning_rate, alpha):
		super(parameters, self).__init__()
		''' parameters for mlp model'''
		self.batch_size = 256
		self.learning_rate = learning_rate
		# self.layers = [64, 32, 16, 8]
		# self.max_epoch_number = 20
		self.max_epoch_number = 100
		self.item_set_size = 1
		self.verbose = 1 # Show performance per X iterations
		self.patience = 20 # assume that there are 100 iterations, if it can not achieve better performance for consecutive 50 times, iteration process will exit
		
		''' parameters for mlp data'''
		self.train_data = 'ML100K/copy1.train'
		self.test_data = 'ML100K/copy1.test'
		self.n = 943
		self.m = 1682
		# self.train_data = 'ML1M/copy1.train'
		# self.test_data = 'ML1M/copy1.test'
		# self.n = 6040 
		# self.m = 3952 
		self.embedding_size = 20
		# self.top_k = 5
		self.rating_types_num = 5
		self.k_gram = [1] # (100K): MAE = 0.7194 , RMSE = 0.9181 (bs:1): 
		# self.k_gram = [2] # (100K): MAE = 0.7190 , RMSE = 0.9180 (bs:1): 
		# self.k_gram = [5] # (100K): MAE = 0.7157 , RMSE = 0.9154 (bs:1): 
		self.W_embedding_size = 0
		for gram in self.k_gram: # 在k_gram算法中，若k=[1]，则有5个array；若k=[2]，则有4个array；若k=[1,2]，则有9个array。这里提到的“5”、“4”、“9”就是 self.W_embedding_size
			self.W_embedding_size += (self.rating_types_num + 1 - gram)
		# self.user_embeddings_size = 32
		# self.item_embeddings_size = 32
		# self.gram_item_embeddings_size = 0
		self.alpha = alpha

class KCOFI_MLP(object):
	"""docstring for KCOFI_MLP"""
	def __init__(self, parameters, data_set):
		super(KCOFI_MLP, self).__init__()
		with tf.name_scope('input'):
			self.user = tf.placeholder(tf.int32, shape = [None, 1]) # "None" means the "batch size" and can be any number, "1" is the "size of data"
			self.item = tf.placeholder(tf.int32, shape = [None, 1])
			for w in range(parameters.W_embedding_size):
				setattr(self, 'gram_items_'+str(w+1), tf.placeholder(tf.int32, shape = [None, 1, data_set.max_inner_length]))
				setattr(self, 'gram_factors_'+str(w+1), tf.placeholder(tf.float32, shape = [None, 1, 1]))
			self.rating = tf.placeholder(tf.float32, shape = [None, 1]) 
			self.train_num = tf.placeholder(tf.int32, shape = None)
			'''
			假设一个batch为128个
			self.user用于放入128个用户，self.item用于放入128个物品，self.rating用于放入128个评分，self.train_num即为传入的用户个数（即可以是128也可以是train_pairs_number%128）；
			若W_embedding_size=5，rating为{1,2,3,4,5}，则self.gram_items_1到self.gram_items_5分别存放这个user评分为1到5的物品集，这5个物品集的大小均为max_inner_length，其中物品号为m+1的表示补齐的无用数据，注意每次的用户不止一个而是128个
			若W_embedding_size=5，rating为{1,2,3,4,5}，则self.gram_factors_1到self.gram_factors_5分别存放这个user评分为1到5的物品集对应的归一化因子，该因子由实际评分的物品个数决定，注意每次的用户不止一个而是128个
			'''

		with tf.name_scope('embeddings_weights'):
        	# "truncated_normal" generate normal distribution random numbers according to the dimension "shape"(eg: [1683, 20]), "mean" is their mean value and "stddev" is their standard deviation
			# shape = (?, 1, layers[0]/2), the reason it becomes half is that the following "concat" operations will make "ui_concat" has shape(?, 1, layers[0])
			self.user_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.n+1, parameters.embedding_size], mean = 0.0, stddev = 0.01), name = 'user_embeddings', dtype = tf.float32)
			self.item_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.m+1, parameters.embedding_size], mean = 0.0, stddev = 0.01), name = 'item_embeddings', dtype = tf.float32)
			for w in range(parameters.W_embedding_size):
				temp_var = tf.Variable(tf.truncated_normal(shape = [parameters.m+1, parameters.embedding_size], mean = 0.0, stddev = 0.01), name = 'gram_item_embeddings_'+str(w+1), dtype = tf.float32)
				zero_vec = tf.constant(0.0, tf.float32, [1, parameters.embedding_size])
				temp_var = tf.concat([temp_var, zero_vec], 0)
				setattr(self, 'gram_item_embeddings_'+str(w+1), temp_var)
			self.g_avg = tf.Variable(data_set.g_avg, name = 'g_avg')
			self.bu = tf.Variable(data_set.bu, name = 'bu')
			self.bi = tf.Variable(data_set.bi, name = 'bi')
			'''
			若W_embedding_size=5，rating为{1,2,3,4,5}
			分别为上述的self.user，self.item，self.gram_items_1到self.gram_items_5设置对应的向量，其中embedding_size即为d=20，并设置self.g_avg，self.bu，self.bi为tensorflow的变量
			变量self.bu，self.bi均为一个列表，self.bu=[0.0, 7.0, 9.0]代表user1的biasU为7.0，user2的biasU为9.0
			'''

		self.kcofi_mlp_model(parameters)

	def kcofi_mlp_model(self, parameters): # 根据论文prediction rule设置self.output，根据objective function设置self.loss
		with tf.name_scope('model'):
			self.embedding_user = tf.nn.embedding_lookup(self.user_embeddings, self.user) # "embedding_lookup" gets the "element"(tensor with specific shape, each one here is [20, 1]) corresponding to "index"(here the index is user)
			self.embedding_item = tf.nn.embedding_lookup(self.item_embeddings, self.item) # the form is like [ [[...]] , [[...]] ]
			self.embedding_normalized_gram_items = []
			self.embedding_total_gram_items = []
			for w in range(parameters.W_embedding_size):
				temp_tensor = tf.reduce_sum(tf.nn.embedding_lookup(getattr(self, 'gram_item_embeddings_'+str(w+1)), getattr(self, 'gram_items_'+str(w+1))), 2)
				self.embedding_total_gram_items.append(temp_tensor)
				self.embedding_normalized_gram_items.append( tf.divide(temp_tensor, getattr(self, 'gram_factors_'+str(w+1)) ) )
				# self.embedding_normalized_gram_items.append( tf.divide(temp_tensor, tf.sqrt(getattr(self, 'gram_factors_'+str(w+1))) ) )
			self.embedding_normalized_gram_items = tf.reduce_sum(self.embedding_normalized_gram_items, 0)
			
			self.embedding_bu = tf.nn.embedding_lookup(self.bu, self.user)
			self.embedding_bi = tf.nn.embedding_lookup(self.bi, self.item)

			self.output = tf.add(tf.add(tf.add(tf.reduce_sum(tf.matmul(tf.add(self.embedding_user, self.embedding_normalized_gram_items), tf.reshape(self.embedding_item, [self.train_num, parameters.embedding_size, 1])), 1), self.embedding_bi), self.embedding_bu), self.g_avg)
		
		with tf.name_scope('loss'):
			self.loss = 0.5*tf.reduce_sum(tf.square(self.rating - self.output)) + 0.5*parameters.alpha*tf.reduce_sum(tf.square(self.embedding_user)) + 0.5*parameters.alpha*tf.reduce_sum(tf.square(self.embedding_item)) \
			+ 0.5*parameters.alpha*tf.reduce_sum(tf.square(self.embedding_bu)) + 0.5*parameters.alpha*tf.reduce_sum(tf.square(self.embedding_bi)) + 0.5*parameters.alpha*tf.reduce_sum(tf.square(self.embedding_total_gram_items))

		with tf.name_scope('optimizer'):
			self.optimizer = tf.train.FtrlOptimizer(learning_rate = parameters.learning_rate).minimize(self.loss)

def get_train_instances(train_pairs, u_gram_items_dict, u_gram_factors_dict, max_inner_length, W_embedding_size, m):
	'''
	处理输入数据的格式，每个batch含128个元素，每个元素分别是对应的user，item，rating，gram_items，factors
	在main函数中，会进一步根据W_embedding_size将gram_items和factors处理成类似gram_items_1到gram_items_5及其对应的factors_1到factors_5的形式
	'''
	users_input, items_input, ratings_input, gram_items_input, factors_input = [], [], [], [], []
	for [u, i, r] in train_pairs:
	    users_input.append(u)
	    items_input.append(i)
	    ratings_input.append(r)
	    # gram_items_input.append(u_gram_items_dict[u])
	    # factors_input.append(u_gram_factors_dict[u])
	    u_gram_items = u_gram_items_dict[u]
	    gram_items_input.append(u_gram_items)
	    u_gram_factors = u_gram_factors_dict[u]
	    modify_u_gram_factors = []
	    for k in range(len(u_gram_items)):
	    	if i in u_gram_items[k] and u_gram_factors[k][0] != 1:
	    		modify_u_gram_factors.append([math.sqrt(u_gram_factors[k][0]-1)])
	    	else:
	    		modify_u_gram_factors.append([math.sqrt(u_gram_factors[k][0])])
	    factors_input.append(modify_u_gram_factors)
	for i in range(len(gram_items_input)):
	    for j in range(W_embedding_size):
	        for k in range(max_inner_length-len(gram_items_input[i][j])):
	            gram_items_input[i][j].append(m+1)
	return np.array(users_input), np.array(items_input), np.array(ratings_input), np.array(gram_items_input), np.array(factors_input)

def get_test_input(test_pairs, u_gram_items_dict, u_gram_factors_dict, max_inner_length, W_embedding_size, m): # 与上面函数的功能一致
    users_input, items_input, ratings_input, gram_items_input, factors_input = [], [], [], [], []
    for [u, i, r] in test_pairs:
    	users_input.append(u)
    	items_input.append(i)
    	ratings_input.append(r)
    	# gram_items_input.append(u_gram_items_dict[u])
    	# factors_input.append(u_gram_factors_dict[u])
    	u_gram_items = u_gram_items_dict[u]
    	gram_items_input.append(u_gram_items)
    	u_gram_factors = u_gram_factors_dict[u]
    	modify_u_gram_factors = []
    	for k in range(len(u_gram_items)):
    		if i in u_gram_items[k] and u_gram_factors[k][0] != 1:
    			modify_u_gram_factors.append([math.sqrt(u_gram_factors[k][0]-1)])
    		else:
    			modify_u_gram_factors.append([math.sqrt(u_gram_factors[k][0])])
    	factors_input.append(modify_u_gram_factors)
    for i in range(len(gram_items_input)):
    	for j in range(W_embedding_size):
    		for k in range(max_inner_length-len(gram_items_input[i][j])):
	    		gram_items_input[i][j].append(m+1)
    return np.array(users_input), np.array(items_input), np.array(ratings_input), np.array(gram_items_input), np.array(factors_input), len(test_pairs)

def evaluation(model, sess, parameters, test_users_input, test_items_input, test_labels_input, test_gram_items_input, test_factors_input, test_pairs_num):
	'''
	predict_rating_list是test数据中对应的所有预测评分，test_labels_input是test数据中的真实评分
	'''
	mae, rmse = 0.0, 0.0
	predict_rating_list = []
	batch_number = test_pairs_num // parameters.batch_size + 1
	for i in range(batch_number):
	    start = i * parameters.batch_size
	    end = np.min([test_pairs_num, (i+1)*parameters.batch_size])
	    users_batch = test_users_input[start : end]
	    items_batch = test_items_input[start : end]
	    ratings_batch = test_labels_input[start : end]
	    gram_items_batch = test_gram_items_input[start : end]
	    factors_batch = test_factors_input[start: end]
	    input_dict = {}
	    input_dict[model.user] = users_batch[:,None]
	    input_dict[model.item] = items_batch[:,None]
	    input_dict[model.rating] = ratings_batch[:,None]
	    input_dict[model.train_num] = len(users_batch[:,None])
	    gram_items_list = np.split(gram_items_batch, parameters.W_embedding_size, 1)
	    gram_factors_list = np.split(factors_batch, parameters.W_embedding_size, 1)
	    for w in range(parameters.W_embedding_size):
	        input_dict[getattr(model, 'gram_items_'+str(w+1))] = gram_items_list[w]
	        input_dict[getattr(model, 'gram_factors_'+str(w+1))] = gram_factors_list[w]
	    predict_rating = sess.run(model.output, feed_dict = input_dict)
	    for rating in predict_rating:
	    	if rating[0] > 5.0:
	    		predict_rating_list.append(5.0)
	    	elif rating[0] < 1.0:
	    		predict_rating_list.append(1.0)
	    	else:
	        	predict_rating_list.append(rating[0])
	for k in range(len(test_labels_input)):
	    mae += abs(float(predict_rating_list[k]) - test_labels_input[k]) / test_pairs_num
	    rmse += ((float(predict_rating_list[k]) - test_labels_input[k])**2) / test_pairs_num
	rmse = rmse**0.5
	return mae, rmse

def main(model, data_set, parameters, users_input, items_input, ratings_input, gram_items_input, factors_input, train_pairs_number):
	with tf.Session(config = tf_config) as sess:
		sess.run(tf.global_variables_initializer())
		test_users_input, test_items_input, test_labels_input, test_gram_items_input, test_factors_input, test_pairs_num = get_test_input(data_set.test_pairs, data_set.u_gram_items_dict, data_set.u_gram_factors_dict, data_set.max_inner_length, parameters.W_embedding_size, parameters.m)

		best_mae, best_rmse, best_iter = 10.0, 10.0, -1

		patience_count = 0
		tf.set_random_seed(1) # seed operations contain "operation level" and "graph level", here is the graph level, all variables defined later can generate the same random number across sessions.
		
		for epoch in range(parameters.max_epoch_number):		

			# shuffled_indexs = np.random.permutation(np.arange(train_pairs_number)) # "permutation" randomly disrupt the sequence of incoming list and return the shuffled list
			# users_input = users_input[shuffled_indexs]
			# items_input = items_input[shuffled_indexs]
			# ratings_input = ratings_input[shuffled_indexs]
			# gram_items_input = gram_items_input[shuffled_indexs]
			# factors_input = factors_input[shuffled_indexs]

			batch_number = train_pairs_number // parameters.batch_size + 1 # calculate how many batches should be counted according to the batch size

			losses = [] # record train loss for each time
			start_time = time()
			for i in range(batch_number):
			    start = i * parameters.batch_size
			    end = np.min([train_pairs_number, (i+1)*parameters.batch_size]) # if the number of elements in last batch is less than batch size, just process the remaining elements

			    users_batch = users_input[start : end]
			    items_batch = items_input[start : end]
			    ratings_batch = ratings_input[start : end]
			    gram_items_batch = gram_items_input[start : end]
			    factors_batch = factors_input[start : end]
			    input_dict = {}
			    input_dict[model.user] = users_batch[:,None]
			    input_dict[model.item] = items_batch[:,None]
			    input_dict[model.rating] = ratings_batch[:,None]
			    input_dict[model.train_num] = len(users_batch[:,None])
			    gram_items_list = np.split(gram_items_batch, parameters.W_embedding_size, 1)
			    gram_factors_list = np.split(factors_batch, parameters.W_embedding_size, 1)
			    for w in range(parameters.W_embedding_size):
			    	input_dict[getattr(model, 'gram_items_'+str(w+1))] = gram_items_list[w]
			    	input_dict[getattr(model, 'gram_factors_'+str(w+1))] = gram_factors_list[w]
			    # because we set user the shape=[None,1], so we should transform [user1, user2, ...] to [ [user1], [user2], ...] using [:,None], otherwise throw out error.
			    _, batch_loss = sess.run([model.optimizer, model.loss], feed_dict = input_dict)
			    losses.append(batch_loss) # batch loss is the log loss of ([all_ratings_in_batch] ,[all_outputs_in_batch]), the result is a float type value and is appended to "losses"
			end_time = time()
			train_time = end_time - start_time
			
			if epoch % parameters.verbose == 0:
				start_time = time()
				(mae, rmse) = evaluation(model, sess, parameters, test_users_input, test_items_input, test_labels_input, test_gram_items_input, test_factors_input, test_pairs_num)
				end_time = time()
				eval_time = end_time- start_time
				# print('Iteration %d: train_time = %.4f s, MAE = %.4f, RMSE = %.4f[%.1f s]\n'% (epoch+1, train_time, mae, rmse, eval_time))
				if rmse < best_rmse: 
				    best_mae, best_rmse, best_iter = mae, rmse, epoch+1
				    patience_count = 0
				else:
				    patience_count += 1

				if patience_count > parameters.patience:
				    break
		print("End. Best Iteration %d: MAE = %.4f , RMSE = %.4f " % (best_iter, best_mae, best_rmse))

if __name__ == '__main__':	
	parser = argparse.ArgumentParser(description = 'manual to this script')
	parser.add_argument('--alpha', type = float, default = 0.0)
	parser.add_argument('--learning_rate', type = float, default = 0.0)
	args = parser.parse_args()
	parameters = parameters(args.learning_rate, args.alpha)
	data_set = data_set(parameters.train_data, parameters.test_data, parameters.k_gram, parameters.rating_types_num, parameters.m, parameters.n)
	users_input, items_input, ratings_input, gram_items_input, factors_input = get_train_instances(data_set.train_pairs, data_set.u_gram_items_dict, data_set.u_gram_factors_dict, data_set.max_inner_length, parameters.W_embedding_size, parameters.m)
	train_pairs_number = len(users_input)
	KCOFI_MLP = KCOFI_MLP(parameters, data_set)
	main(KCOFI_MLP, data_set, parameters, users_input, items_input, ratings_input, gram_items_input, factors_input, train_pairs_number) # 把layers的最后一层1设置为5