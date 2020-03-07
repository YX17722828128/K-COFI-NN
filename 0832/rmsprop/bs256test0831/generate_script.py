alpha = [0.001, 0.005, 0.01, 0.05]
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
for al in alpha:
	for rate in learning_rate:
			name = 'alpha_'+str(al)+'@rate_'+str(rate)
			f = open(name+'.bat', 'a')
			f.write('python tf_k_cofi.py --alpha='+str(al)+' --learning_rate='+str(rate)+' >test_results/'+name+'.txt\npause')
			f.close()