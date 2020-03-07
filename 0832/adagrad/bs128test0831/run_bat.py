import psutil,subprocess,time,os

execute_files = []
for file in os.listdir():
	if 'bat' in file:
		execute_files.append(file)

index = 0
child = subprocess.Popen(execute_files[index], shell = False)
time.sleep(2)
while True:
    f = open('test_results/'+execute_files[index].replace('bat', 'txt'), 'r', encoding = 'utf-8')
    data = f.read()
    f.close()
    if len(data) > 0:
        print('file: ' + execute_files[index] + ' finished.')
        index += 1
        child = subprocess.Popen(execute_files[index], shell = False)
        print('execute file: ' + execute_files[index])
        if index == len(execute_files):
            break
    time.sleep(20)