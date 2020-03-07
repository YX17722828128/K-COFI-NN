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

# data_set = data_set('ML100K-copy1-train', 'ML100K-copy1-test')