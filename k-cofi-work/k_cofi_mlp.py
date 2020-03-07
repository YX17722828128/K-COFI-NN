import random, os, math, copy
import tensorflow as tf
from time import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

class data_set(object):
	"""docstring for data_set"""
	def __init__(self, train_data, test_data, k_gram, rating_types_num):
		super(data_set, self).__init__()
		self.train_pairs, self.train_dict, self.all_users, self.all_items = self.read_train_data(train_data, rating_types_num)
		self.test_pairs, self.test_dict = self.read_test_data(test_data, rating_types_num)
		self.test_unobserved_dict = self.get_test_unobserved_dict()
		self.all_gram_data = self.get_k_gram_data(train_data, k_gram, rating_types_num)
		self.u_gram_items_dict = {}
		self.u_gram_factors_dict = {}

	def read_train_data(self, train_data, rating_types_num):
		train_pairs = []
		train_dict = dict()
		all_users = set()
		all_items = set()
		for line in open(train_data, 'r', encoding = 'utf-8'):
			record = line.split()
			if rating_types_num == 5:
				user, item, rating = int(record[0]), int(record[1]), int(record[2])
			elif rating_types_num == 10:
				user, item, rating = int(record[0]), int(record[1]), int(record[2])*2
			train_pairs.append([user, item, rating])
			if user not in train_dict:
				train_dict[user] = set()
				train_dict[user].add(item)
			else:
				train_dict[user].add(item)
			all_users.add(user)
			all_items.add(item)
		return train_pairs, train_dict, all_users, all_items

	def read_test_data(self, test_data, rating_types_num):
		test_pairs = []
		test_dict = dict()
		for line in open(test_data, 'r', encoding = 'utf-8'):
			record = line.split()
			if rating_types_num == 5:
				user, item, rating = int(record[0]), int(record[1]), int(record[2])
			elif rating_types_num == 10:
				user, item, rating = int(record[0]), int(record[1]), int(record[2])*2
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

	def get_k_gram_data(self, train_data, k_gram, rating_types_num):
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
		return all_gram_data	

class parameters(object):
	"""docstring for parameters"""
	def __init__(self):
		super(parameters, self).__init__()
		''' parameters for mlp model'''
		self.batch_size = 128		
		self.learning_rate = 0.001
		self.layers = [64, 32, 16, 8]
		self.max_epoch_number = 150
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
		self.k_gram = [1, 2]
		self.W_embedding_size = 0
		for gram in self.k_gram:
			self.W_embedding_size += (self.rating_types_num + 1 - gram)
		self.user_embeddings_size = 21
		self.item_embeddings = 21
		self.gram_item_embeddings = 22

class KCOFI_MLP(object):
	"""docstring for KCOFI_MLP"""
	def __init__(self, parameters, data_set, max_inner_length):
		super(KCOFI_MLP, self).__init__()
		with tf.name_scope('input'):
			self.user = tf.placeholder(tf.int32, shape = [None, 1]) # "None" means the "batch size" and can be any number, "1" is the "size of data"
			self.item = tf.placeholder(tf.int32, shape = [None, 1])
			for gram in parameters.k_gram:
				setattr(self, 'gram_'+str(gram)+'_item', tf.placeholder(tf.int32, shape = [None, 1, max_inner_length]))
				setattr(self, 'gram_'+str(gram)+'_factor', tf.placeholder(tf.float32, shape = [None, 1, 1]))
			self.rating = tf.placeholder(tf.float32, shape = [None, 1]) 
			self.train_num = tf.placeholder(tf.int32, shape = None)

		with tf.name_scope('embeddings_weights'):
        	# "truncated_normal" generate normal distribution random numbers according to the dimension "shape"(eg: [1683, 20]), "mean" is their mean value and "stddev" is their standard deviation
			# shape = (?, 1, layers[0]/2), the reason it becomes half is that the following "concat" operations will make "ui_concat" has shape(?, 1, layers[0])
			self.user_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.n+1, parameters.user_embeddings_size], mean = 0.0, stddev = 0.01), name = 'user_embeddings', dtype = tf.float32)
			self.item_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.m+1, parameters.item_embeddings], mean = 0.0, stddev = 0.01), name = 'item_embeddings', dtype = tf.float32)
			self.gram_item_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.m+1, parameters.gram_item_embeddings], mean = 0.0, stddev = 0.01), name = 'gram_item_embeddings', dtype = tf.float32)
			self.hidden_layer_W = [] # different "W" value reflect to different layer according to their index(layer position)
			self.hidden_layer_b = [] # different "b" value reflect to different layer according to their index(layer position)
			for index in range(len(parameters.layers)-1):
				# all W-related layers are hidden layers. W_1:shape(64, 32); W_2:shape(32, 16); W_3:shape(16, 8);
				self.hidden_layer_W.append(tf.Variable(tf.truncated_normal(shape = [parameters.layers[index], parameters.layers[index+1]], mean = 0.0, stddev = 0.01), name = 'hidden_layer_W_'+str(index+1), dtype = tf.float32))
				self.hidden_layer_b.append(tf.Variable(tf.zeros(parameters.layers[index+1]), name = 'hidden_layer_b_'+str(index+1))) # [0.0, 0.0, 0.0]
		
			self.h = tf.Variable(tf.truncated_normal(shape = [parameters.layers[-1], 1], mean = 0.0, stddev = 0.01), name = 'h_weights', dtype = tf.float32) # h:shape(8, 1)
			self.bias = tf.Variable(tf.zeros(1), name = 'bias') # 0.0

		self.kcofi_mlp_model(parameters)

	def kcofi_mlp_model(self, parameters):
		with tf.name_scope('model'):
			self.embedding_user = tf.nn.embedding_lookup(self.user_embeddings, self.user) # "embedding_lookup" gets the "element"(tensor with specific shape, each one here is [20, 1]) corresponding to "index"(here the index is user)
			self.embedding_item = tf.nn.embedding_lookup(self.item_embeddings, self.item) # the form is like [ [[...]] , [[...]] ]
			self.embedding_total_gram_items = []
			for gram in parameters.k_gram:
				self.embedding_gram_items = tf.divide(tf.reduce_sum(tf.nn.embedding_lookup(self.gram_item_embeddings, getattr(self, 'gram_'+str(gram)+'_item')), 2), getattr(self, 'gram_'+str(gram)+'_factor'))
				self.embedding_total_gram_items.append(self.embedding_gram_items)
			self.embedding_total_gram_items = tf.reduce_sum(self.embedding_total_gram_items, 0)

			self.ui_concat = tf.concat([self.embedding_user, self.embedding_item, self.embedding_total_gram_items], 2) # combine into shape(?, 1, 64)

			self.a = tf.reduce_mean(self.ui_concat, 1) # then turn [ [[]] , [[]] ... ] into [ [] , [] ... ] and get the shape(?, 64)
			for index in range(len(parameters.layers)-1): # execute matrix-like calculation (shape): (?,64)*(64,32)*(32,16)*(16,8)=(?,8)
				self.a = tf.nn.relu(tf.add(tf.matmul(self.a, self.hidden_layer_W[index]), self.hidden_layer_b[index])) # choose from sigmoid, tanh, relu.

			# self.output = tf.add(tf.to_float(tf.argmax(tf.sigmoid(tf.add(tf.matmul(self.a, self.h), self.bias)), axis = 1, output_type = tf.int32)), 1.0) # sigmoid(x) = 1 / (1 + e^(-x)), shape(?, rating_types_num) # tf.while_loop
		
		# with tf.name_scope('loss'):
		# 	self.loss = tf.reduce_mean(tf.square(tf.reduce_mean(self.rating, 1) - self.output))	

			self.output = tf.multiply(tf.add(tf.matmul(self.a, self.h), self.bias), parameters.rating_types_num)

			self.output = tf.where(self.output>=1.0, self.output, tf.ones([self.train_num, 1]))

			self.output = tf.where(self.output<=5.0, self.output, 5*tf.ones([self.train_num, 1]))
		
		with tf.name_scope('loss'):
			self.loss = tf.reduce_mean(tf.square(self.rating - self.output))

		with tf.name_scope('optimizer'):
			self.optimizer = tf.train.GradientDescentOptimizer(learning_rate = parameters.learning_rate).minimize(self.loss)

def get_train_instances(train_pairs, all_gram_data, k_gram, rating_types_num):
    users_input, items_input, ratings_input, gram_items_input, factors_input = [], [], [], [], []
    max_inner_length = 0
    temp_length = 0
    u_gram_items_dict = {}
    u_gram_factors_dict = {}
    for [u, i, r] in train_pairs:
    	users_input.append(u)
    	items_input.append(i)
    	ratings_input.append(r)
    	if rating_types_num == 5: 
    		inner_gram_items_input = []
    		inner_factors_input = []
    		for gram in k_gram:
	        	if gram == 1:
	        		item_set = all_gram_data['1_gram'][u][r]
	        		temp_length = len(item_set)
	        		inner_gram_items_input.append(item_set)
	        	elif gram == 2:
	        		if r != 5 and r != 1:
	        			item_set = all_gram_data['2_gram'][u][r-1] + all_gram_data['2_gram'][u][r]
	        			temp_length = len(item_set)
	        			inner_gram_items_input.append(item_set)
	        		elif r == 1:
	        			item_set = all_gram_data['2_gram'][u][r]
	        			temp_length = len(item_set)
	        			inner_gram_items_input.append(item_set)
	        		elif r == 5:
	        			item_set = all_gram_data['2_gram'][u][r-1]
	        			temp_length = len(item_set)
	        			inner_gram_items_input.append(item_set)
	        	elif gram == 5:
	        		item_set = all_gram_data['5_gram'][u]
	        		temp_length = len(item_set)
	        		inner_gram_items_input.append(item_set)
	        	
	        	if r in item_set:
	        		factor = math.sqrt(len(item_set) - 1)
	        	else:
	        		factor = math.sqrt(len(item_set))
	        	inner_factors_input.append([factor])
    		gram_items_input.append(inner_gram_items_input)
    		factors_input.append(inner_factors_input)
    	elif rating_types_num == 10:
    		inner_gram_items_input = []
    		inner_factors_input = []
	    	for gram in k_gram:
	    		if gram == 1:
	    			item_set = all_gram_data['1_gram'][u][r]
	    			temp_length = len(item_set)
	    			inner_gram_items_input.append(item_set)
	    		elif gram == 2:
        			if r in [1, 2]:
        				item_set = all_gram_data['2_gram'][u][1]
        				temp_length = len(item_set)
        				inner_gram_items_input.append(item_set)
        			elif r in [9, 10]:
        				item_set = all_gram_data['2_gram'][u][7]
        				temp_length = len(item_set)
        				inner_gram_items_input.append(item_set)
        			else:
        				if r in [3, 4]:
        					item_set = all_gram_data['2_gram'][u][1] + all_gram_data['2_gram'][u][3]
        					temp_length = len(item_set)
        					inner_gram_items_input.append(item_set)
        				elif r in [5, 6]:
        					item_set = all_gram_data['2_gram'][u][3] + all_gram_data['2_gram'][u][5]
        					temp_length = len(item_set)
        					inner_gram_items_input.append(item_set)
        				elif r in [7, 8]:
        					item_set = all_gram_data['2_gram'][u][5] + all_gram_data['2_gram'][u][7]
        					temp_length = len(item_set)
        					inner_gram_items_input.append(item_set)
	    		elif gram == 5:
	        			item_set = all_gram_data['5_gram'][u]
	        			temp_length = len(item_set)
	        			inner_gram_items_input.append(item_set)

	    		if r in item_set:
	    			factor = math.sqrt(len(item_set) - 1)
	    		else:
	    			factor = math.sqrt(len(item_set))
	    		inner_factors_input.append([factor])	
    		gram_items_input.append(inner_gram_items_input)
	    	factors_input.append(inner_factors_input)
    	if max_inner_length < temp_length:
    		max_inner_length = temp_length
    	u_gram_items_dict[u] = inner_gram_items_input
    	u_gram_factors_dict[u] = inner_factors_input
    for i in range(len(gram_items_input)):
    	for j in range(len(k_gram)):
    		for k in range(max_inner_length-len(gram_items_input[i][j])):
	    		gram_items_input[i][j].append(0)
    return max_inner_length, np.array(users_input), np.array(items_input), np.array(ratings_input), np.array(gram_items_input), np.array(factors_input), u_gram_items_dict, u_gram_factors_dict

def get_test_input(test_pairs, u_gram_items_dict, u_gram_factors_dict):
    users_input, items_input, labels_input, gram_items_input, factors_input = [], [], [], [], []
    for [u, i, r] in test_pairs:
        users_input.append(u)
        items_input.append(i)
        labels_input.append(r)
        gram_items_input.append(u_gram_items_dict[u])
        factors_input.append(u_gram_factors_dict[u])
    users_input = np.array(users_input)
    items_input = np.array(items_input)
    labels_input = np.array(labels_input)
    gram_items_input = np.array(gram_items_input)
    factors_input = np.array(factors_input)
    return users_input, items_input, labels_input, gram_items_input, factors_input, len(test_pairs)

def evaluation(model, sess, parameters, test_users_input, test_items_input, test_labels_input, test_gram_items_input, test_factors_input, test_pairs_num):
    mae, rmse = 0.0, 0.0
    predict_rating_list = []
    test_pairs_number = len(test_users_input)
    batch_number = test_pairs_number // parameters.batch_size + 1
    for i in range(batch_number):
        start = i * parameters.batch_size
        end = np.min([test_pairs_number, (i+1)*parameters.batch_size])
        users_batch = test_users_input[start : end]
        items_batch = test_items_input[start : end]
        input_dict = {}
        input_dict[model.user] = users_batch[:,None]
        input_dict[model.item] = items_batch[:,None]
        for j in range(len(parameters.k_gram)):
            input_dict[getattr(model, 'gram_'+str(parameters.k_gram[j])+'_item')] = test_gram_items_input[:,j][start : end][:,None]
            input_dict[getattr(model, 'gram_'+str(parameters.k_gram[j])+'_factor')] = test_factors_input[:,j][start : end][:,None]
        input_dict[model.train_num] = len(users_batch[:,None])
        predict_rating = sess.run(model.output, feed_dict = input_dict)
        for rating in predict_rating:
        	predict_rating_list.append(rating[0])
    for k in range(len(test_labels_input)):
    	mae += abs(float(predict_rating_list[k]) - test_labels_input[k]) / test_pairs_num
    	rmse += ((float(predict_rating_list[k]) - test_labels_input[k])**2) / test_pairs_num
    rmse = rmse**0.5
    print(mae, rmse)
    return mae, rmse

def main(model, data_set, parameters, users_input, items_input, ratings_input, gram_items_input, factors_input, train_pairs_number):
	with tf.Session(config = tf_config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.assign(model.gram_item_embeddings[0], np.zeros([parameters.gram_item_embeddings])))
		test_users_input, test_items_input, test_labels_input, test_gram_items_input, test_factors_input, test_pairs_num = get_test_input(data_set.test_pairs, data_set.u_gram_items_dict, data_set.u_gram_factors_dict)
		print("initialization Completed")
		print()

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

			for i in range(len(parameters.k_gram)):
				setattr(model, 'gram_'+str(parameters.k_gram[i])+'_items_input', gram_items_input[:,i])
				setattr(model, 'gram_'+str(parameters.k_gram[i])+'_factors_input', factors_input[:,i])

			batch_number = train_pairs_number // parameters.batch_size + 1 # calculate how many batches should be counted according to the batch size

			losses = [] # record train loss for each time
			start_time = time()
			for i in range(batch_number-1):
			    start = i * parameters.batch_size
			    end = np.min([train_pairs_number, (i+1)*parameters.batch_size]) # if the number of elements in last batch is less than batch size, just process the remaining elements

			    users_batch = users_input[start : end]
			    items_batch = items_input[start : end]
			    ratings_batch = ratings_input[start : end]
			    input_dict = {}
			    input_dict[model.user] = users_batch[:,None]
			    input_dict[model.item] = items_batch[:,None]
			    input_dict[model.rating] = ratings_batch[:,None]
			    for gram in parameters.k_gram:
			    	input_dict[getattr(model, 'gram_'+str(gram)+'_item')] = getattr(model, 'gram_'+str(gram)+'_items_input')[start : end][:,None]
			    	input_dict[getattr(model, 'gram_'+str(gram)+'_factor')] = getattr(model, 'gram_'+str(gram)+'_factors_input')[start : end][:,None]
			    input_dict[model.train_num] = len(users_batch[:,None])
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
				print('Iteration %d: train_time = %.4f s, MAE = %.4f, RMSE = %.4f[%.1f s]\n'% (epoch+1, train_time, mae, rmse, eval_time))
				if rmse < best_rmse: 
				    best_mae, best_rmse, best_iter = mae, rmse, epoch+1
				    patience_count = 0
				else:
				    patience_count += 1

				if patience_count > parameters.patience:
				    break
		print("End. Best Iteration %d: MAE = %.4f , RMSE = %.4f " % (best_iter, best_mae, best_rmse))

if __name__ == '__main__':	
	parameters = parameters()
	data_set = data_set(parameters.train_data, parameters.test_data, parameters.k_gram, parameters.rating_types_num)
	max_inner_length, users_input, items_input, ratings_input, gram_items_input, factors_input, u_gram_items_dict, u_gram_factors_dict = get_train_instances(data_set.train_pairs, data_set.all_gram_data, parameters.k_gram, parameters.rating_types_num)
	data_set.u_gram_items_dict = u_gram_items_dict
	data_set.u_gram_factors_dict = u_gram_factors_dict
	train_pairs_number = len(users_input)
	KCOFI_MLP = KCOFI_MLP(parameters, data_set, max_inner_length)
	main(KCOFI_MLP, data_set, parameters, users_input, items_input, ratings_input, gram_items_input, factors_input, train_pairs_number) # 把layers的最后一层1设置为5