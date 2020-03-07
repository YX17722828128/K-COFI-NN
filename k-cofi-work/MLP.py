import random, os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import tensorflow as tf
from time import time
import numpy as np
from data_set import data_set
from evaluate import evaluation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True

class data_set(object):
	"""docstring for data_set"""
	def __init__(self, train_data, test_data):
		super(data_set, self).__init__()
		self.train_pairs, self.train_dict, self.all_users, self.all_items = self.read_train_data(train_data)
		self.test_pairs, self.test_dict = self.read_test_data(test_data)
		self.test_unobserved_dict = self.get_test_unobserved_dict()

	def read_train_data(self, train_data):
		train_pairs = []
		train_dict = dict()
		all_users = set()
		all_items = set()
		for line in open(train_data, 'r', encoding = 'utf-8'):
			record = line.split()
			user, item = int(record[0]), int(record[1])
			train_pairs.append([user, item])
			if user not in train_dict:
				train_dict[user] = set()
				train_dict[user].add(item)
			else:
				train_dict[user].add(item)
			all_users.add(user)
			all_items.add(item)
		return train_pairs, train_dict, all_users, all_items

	def read_test_data(self, test_data):
		test_pairs = []
		test_dict = dict()
		for line in open(test_data, 'r', encoding = 'utf-8'):
			record = line.split()
			user, item = int(record[0]), int(record[1])
			test_pairs.append([user, item])
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

class parameters(object):
	"""docstring for parameters"""
	def __init__(self):
		super(parameters, self).__init__()
		self.batch_size = 128
		self.embedding_size = 20
		self.train_data = 'ML100K-copy1-train'
		self.test_data = 'ML100K-copy1-test'
		self.learning_rate = 0.0005
		self.n = 943
		self.m = 1682
		self.layers = [64, 32, 16, 8]
		self.max_epoch_number = 150
		self.negative_ratio = 3
		self.item_set_size = 1
		self.top_k = 5
		self.verbose = 1 # Show performance per X iterations
		self.patience = 20 # assume that there are 100 iterations, if it can not achieve better performance for consecutive 50 times, iteration process will exit

class MLP(object):
	"""docstring for MLP"""
	def __init__(self, parameters):
		super(MLP, self).__init__()
		with tf.name_scope('input'):
			self.user = tf.placeholder(tf.int32, shape = [None, 1]) # "None" means the "batch size" and can be any number, "1" is the "size of data"
			self.item = tf.placeholder(tf.int32, shape = [None, 1])
			self.rating = tf.placeholder(tf.float32, shape = [None, 1]) # 1 or -1

		with tf.name_scope('embeddings_weights'):
        	# "truncated_normal" generate normal distribution random numbers according to the dimension "shape"(eg: [1683, 20]), "mean" is their mean value and "stddev" is their standard deviation
			# shape = (?, 1, layers[0]/2), the reason it becomes half is that the following "concat" operations will make "ui_concat" has shape(?, 1, layers[0])
			self.user_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.n+1, int(parameters.layers[0]/2)], mean = 0.0, stddev = 0.01), name = 'user_embeddings', dtype = tf.float32)
			self.item_embeddings = tf.Variable(tf.truncated_normal(shape = [parameters.m+1, int(parameters.layers[0]/2)], mean = 0.0, stddev = 0.01), name = 'item_embeddings', dtype = tf.float32)
			self.hidden_layer_W = [] # different "W" value reflect to different layer according to their index(layer position)
			self.hidden_layer_b = [] # different "b" value reflect to different layer according to their index(layer position)
			for index in range(len(parameters.layers)-1):
				# all W-related layers are hidden layers. W_1:shape(64, 32); W_2:shape(32, 16); W_3:shape(16, 8);
				self.hidden_layer_W.append(tf.Variable(tf.truncated_normal(shape = [parameters.layers[index], parameters.layers[index+1]], mean = 0.0, stddev = 0.01), name = 'hidden_layer_W_'+str(index+1), dtype = tf.float32))
				self.hidden_layer_b.append(tf.Variable(tf.zeros(parameters.layers[index+1]), name = 'hidden_layer_b_'+str(index+1))) # [0.0, 0.0, 0.0]
		
			self.h = tf.Variable(tf.truncated_normal(shape = [parameters.layers[-1], 1], mean = 0.0, stddev = 0.01), name = 'h_weights', dtype = tf.float32) # h:shape(8, 1)
			self.bias = tf.Variable(tf.zeros(1), name = 'bias') # 0.0

		self.mlp_model()

	def mlp_model(self):
		with tf.name_scope('model'):
			self.embedding_user = tf.nn.embedding_lookup(self.user_embeddings, self.user) # "embedding_lookup" gets the "element"(tensor with specific shape, each one here is [20, 1]) corresponding to "index"(here the index is user)
			self.embedding_item = tf.nn.embedding_lookup(self.item_embeddings, self.item) # the form is like [ [[...]] , [[...]] ]
            
			self.ui_concat = tf.concat([self.embedding_user, self.embedding_item], 2) # combine two shape(?, 1, 32) into shape(?, 1, 64)
            
			self.a = tf.reduce_mean(self.ui_concat, 1) # then turn [ [[]] , [[]] ... ] into [ [] , [] ... ] and get the shape(?, 64)
			for index in range(len(parameters.layers)-1): # execute matrix-like calculation (shape): (?,64)*(64,32)*(32,16)*(16,8)=(?,8)
				self.a = tf.nn.relu(tf.add(tf.matmul(self.a, self.hidden_layer_W[index]), self.hidden_layer_b[index])) # choose from sigmoid, tanh, relu.

			self.output = tf.sigmoid(tf.add(tf.matmul(self.a, self.h), self.bias)) # sigmoid(x) = 1 / (1 + e^(-x)), shape(?, 1)

		with tf.name_scope('loss'):
			self.loss = tf.losses.log_loss(self.rating, self.output) # log_loss(labels, predictions) # labels = 1 or -1，depends on the classification

		with tf.name_scope('optimizer'):
			self.optimizer = tf.train.AdamOptimizer(learning_rate = parameters.learning_rate).minimize(self.loss)
		
def get_train_instances(train_pairs, negative_ratio, n, m, all_users, all_items):
    users_input, items_input, ratings_input = [], [], []
    # positive instance
    for (u, i) in train_pairs:
        users_input.append(u)
        items_input.append(i)
        ratings_input.append(1.0)
    # negative instance
    negative_instance_number = negative_ratio * len(train_pairs)
    k = 0
    while k < negative_instance_number:
        u = random.randint(1, n)
        j = random.randint(1, m)
        if u not in all_users or j not in all_items:
            continue
        if (u, j) in train_pairs:
            continue
        users_input.append(u)
        items_input.append(j)
        ratings_input.append(0.0)
        k += 1
    return np.array(users_input), np.array(items_input), np.array(ratings_input)

def displayResult(varname, metric):
    count = len(metric)
    for k in range(1, count):
        print("%s@%d: %.4f " % (varname, k, metric[k]))

def main(model, data_set, parameters):
	with tf.Session(config = tf_config) as sess:
		sess.run(tf.global_variables_initializer())
		print("initialization Completed")
		print()

		start_time = time()
		(precisions, recalls, F1s, ndcgs, one_calls, eval_loss) = evaluation(model, sess, parameters.top_k, data_set.test_unobserved_dict, data_set.test_dict, data_set.all_items)
		ndcgs = np.mean(np.array(ndcgs), axis = 0)
		precisions = np.mean(np.array(precisions), axis = 0)
		recalls = np.mean(np.array(recalls), axis = 0)
		F1s = np.mean(np.array(F1s), axis = 0)
		one_calls = np.mean(np.array(one_calls), axis = 0)
		end_time = time()

		print('Init prediction completed [%.1f s]: eval_loss = %.4f, NDCG@%d= %.4f' % (end_time-start_time, np.mean(eval_loss), parameters.top_k, ndcgs[parameters.top_k]))
		displayResult("Precision",precisions)
		displayResult("Recall",recalls)
		displayResult("F1",F1s)
		displayResult("NDCG",ndcgs)
		displayResult("1-call",one_calls)
		print()

		best_prec, best_rec, best_f1, best_ndcg, best_1call, best_iter = precisions, recalls, F1s, ndcgs, one_calls, -1
		best_ndcg_5 = ndcgs[parameters.top_k]

		patience_count = 0

		tf.set_random_seed(1) # seed operations contain "operation level" and "graph level", here is the graph level, all variables defined later can generate the same random number across sessions.

		for epoch in range(parameters.max_epoch_number):

			users_input, items_input, ratings_input = get_train_instances(data_set.train_pairs, parameters.negative_ratio, parameters.n, parameters.m, data_set.all_users, data_set.all_items)
			train_pairs_number = len(users_input)

			shuffled_indexs = np.random.permutation(np.arange(train_pairs_number)) # "permutation" randomly disrupt the sequence of incoming list and return the shuffled list
			users_input = users_input[shuffled_indexs]
			items_input = items_input[shuffled_indexs]
			ratings_input = ratings_input[shuffled_indexs]

			batch_number = train_pairs_number // parameters.batch_size + 1 # calculate how many batches should be counted according to the batch size

			losses = [] # record train loss for each time
			start_time = time()
			for i in range(batch_number):
			    start = i * parameters.batch_size
			    end = np.min([train_pairs_number, (i+1)*parameters.batch_size]) # if the number of elements in last batch is less than batch size, just process the remaining elements

			    users_batch = users_input[start : end]
			    items_batch = items_input[start : end]
			    ratings_batch = ratings_input[start : end]
				# because we set user the shape=[None,1], so we should transform [user1, user2, ...] to [ [user1], [user2], ...] using [:,None], otherwise throw out error.
			    _, batch_loss = sess.run([model.optimizer, model.loss], feed_dict = {model.user : users_batch[:,None], model.item : items_batch[:,None], model.rating : ratings_batch[:,None]})
			    losses.append(batch_loss) # batch loss is the log loss of ([all_ratings_in_batch] ,[all_outputs_in_batch]), the result is a float type value and is appended to "losses"
			end_time = time()
			train_time = end_time - start_time

			if epoch % parameters.verbose == 0:
			    start_time = time()
			    (precisions, recalls, F1s, ndcgs, one_calls, eval_loss) = evaluation(model, sess, parameters.top_k, data_set.test_unobserved_dict, data_set.test_dict, data_set.all_items)
			    ndcgs = np.mean(np.array(ndcgs), axis = 0)
			    precisions = np.mean(np.array(precisions), axis = 0)
			    recalls = np.mean(np.array(recalls), axis = 0)
			    F1s = np.mean(np.array(F1s), axis = 0)
			    one_calls = np.mean(np.array(one_calls), axis = 0)
			    end_time = time()
			    eval_time = end_time- start_time

			    print('Iteration %d: train_loss = %.4f[%.1f s], eval_loss = %.4f[%.1f s], NDCG@%d = %.4f'% (epoch+1, np.mean(losses), train_time, np.mean(eval_loss), eval_time, parameters.top_k, ndcgs[parameters.top_k]))
			    displayResult("Precision", precisions)
			    displayResult("Recall", recalls)
			    displayResult("F1", F1s)
			    displayResult("NDCG" ,ndcgs)
			    displayResult("1-call", one_calls)
			    print()

			    if ndcgs[parameters.top_k] > best_ndcg_5: # evaluation() has insured that the real index starts from 1, eg: ndcgs[5] is the fifth ndcg value(ndcg@5)
			        best_prec, best_rec, best_f1, best_ndcg, best_1call, best_iter = precisions, recalls, F1s, ndcgs, one_calls, epoch+1
			        best_ndcg_5 = ndcgs[parameters.top_k]
			        patience_count = 0
			    else:
			        patience_count += 1

			    if patience_count > parameters.patience:
			        break

		print("End. Best Iteration %d: NDCG@%d = %.4f " % (best_iter, parameters.top_k, best_ndcg_5))
		displayResult("Precision", best_prec)
		displayResult("Recall", best_rec)
		displayResult("F1", best_f1)
		displayResult("NDCG", best_ndcg)
		displayResult("1-call", best_1call)

if __name__ == '__main__':
	parameters = parameters()
	MLP = MLP(parameters)
	data_set = data_set(parameters.train_data, parameters.test_data)
	main(MLP, data_set, parameters)