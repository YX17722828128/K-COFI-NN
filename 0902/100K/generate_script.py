k_gram = [1, 2, 5]
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
data_set = 'ML100K'
n = 943
m = 1682
# data_set = 'ML1M'
# n = 6040
# m = 3952
for k in k_gram:
	for rate in learning_rate:
		for no in range(1, 2):
			name = 'k_'+str(k)+'@r_'+str(rate)+'@'+data_set+'_copy'+str(no)
			f = open(name+'.bat', 'a')
			f.write('python tf_k_cofi_mlp.py --n='+str(n)+' --m='+str(m)+' --learning_rate='+str(rate)+' --k_gram='+str(k)+' --train_data='+data_set+'/copy'+str(no)+'.train --test_data='+data_set+'/copy'+str(no)+'.test >kcofi_results/'+name+'.txt\npause')
			f.close()
			