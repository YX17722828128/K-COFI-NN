import os, re

files = os.listdir()
folders = []
for file in files:
	if '.' not in file:
		folders.append(file)

txt_names = []
for folder in folders:
	txt_names = os.listdir(folder+'/bs128test0831/test_results')
	break

COUNTS = []
MAES = []
RMSES = []
for folder in folders:
	inner_c = []
	inner_m = []
	inner_r = []
	for txt in txt_names:
		f = open(folder+'/bs128test0831/test_results/'+txt, 'r', encoding = 'utf-8')
		data = f.read()
		f.close()
		iteration_time = re.findall(r'Iteration\s+(.*?):', data)[0]
		MAE = re.findall(r'MAE = (.*?) ,', data)[0]
		RMSE = re.findall(r'RMSE = (.*?) ', data)[0]
		inner_c.append(iteration_time)
		inner_m.append(MAE)
		inner_r.append(RMSE)
	COUNTS.append(inner_c)
	MAES.append(inner_m)
	RMSES.append(inner_r)
	inner_c = []
	inner_m = []
	inner_r = []
	for txt in txt_names:
		f = open(folder+'/bs256test0831/test_results/'+txt, 'r', encoding = 'utf-8')
		data = f.read()
		f.close()
		iteration_time = re.findall(r'Iteration\s+(.*?):', data)[0]
		MAE = re.findall(r'MAE = (.*?) ,', data)[0]
		RMSE = re.findall(r'RMSE = (.*?) ', data)[0]
		inner_c.append(iteration_time)
		inner_m.append(MAE)
		inner_r.append(RMSE)
	COUNTS.append(inner_c)
	MAES.append(inner_m)
	RMSES.append(inner_r)

FIX_COUNTS = []
FIX_MAES = []
FIX_RMSES = []
for a in range(len(RMSES[0])):
	inner_c = []
	inner_m = []
	inner_r = []
	for b in range(len(RMSES)):
		inner_c.append(COUNTS[b][a])
		inner_m.append(MAES[b][a])
		inner_r.append(RMSES[b][a])
	FIX_COUNTS.append(inner_c)
	FIX_MAES.append(inner_m)
	FIX_RMSES.append(inner_r)

A = 0
B = 0
MIN_COUNT = 0
MIN_MAE = 0.0
MIN_RMSE = 10.0
for a in range(len(FIX_RMSES)):
	for b in range(len(FIX_RMSES[a])):
		if float(FIX_RMSES[a][b]) < MIN_RMSE:
			A = a
			B = b
			MIN_COUNT = int(FIX_COUNTS[a][b])
			MIN_MAE = float(FIX_MAES[a][b])
			MIN_RMSE = float(FIX_RMSES[a][b])

temp = 0
for folder in folders:
	for file in ['bs128test0831', 'bs256test0831']:
		if temp == B:
			print(folder, file)
		temp += 1
print(txt_names[A])
print(MIN_COUNT)
print(MIN_MAE)			
print(MIN_RMSE)

def compare():
	max_disparity = 0.0
	for folder in folders:
		print('*******************************')
		for txt in txt_names:
			f = open(folder+'/bs128test0831/test_results/'+txt)
			data_128 = f.read()
			RMSE_128 = re.findall(r'RMSE = (.*?) ', data_128)[0]
			f.close()
			f = open(folder+'/bs256test0831/test_results/'+txt)
			data_256 = f.read()
			RMSE_256 = re.findall(r'RMSE = (.*?) ', data_256)[0]
			f.close()
			if float(RMSE_128) < 1.0 and float(RMSE_256) < 1.0:
				disparity = abs(float(RMSE_128)-float(RMSE_256))
				print(RMSE_128, RMSE_256, disparity)
				if disparity > max_disparity:
					max_disparity = disparity
	print('max disparity '+str(max_disparity))
# compare()

import xlwt
workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('my worksheet')
row_num = 0
for folder in folders:
	for txt in txt_names:
		f = open(folder+'/bs128test0831/test_results/'+txt)
		data_128 = f.read()
		iteration_time = re.findall(r'Iteration\s+(.*?):', data_128)[0]
		MAE_128 = re.findall(r'MAE = (.*?) ,', data_128)[0]
		RMSE_128 = re.findall(r'RMSE = (.*?) ', data_128)[0]
		alpha = re.findall(r'alpha_(.*?)@', txt)[0]
		gamma = re.findall(r'@rate_(.*?)\.txt', txt)[0]
		f.close()
		worksheet.write(row_num, 0, label = '100K')
		worksheet.write(row_num, 1, label = '128')
		worksheet.write(row_num, 2, label = folder)
		worksheet.write(row_num, 3, label = alpha)
		worksheet.write(row_num, 4, label = gamma)
		worksheet.write(row_num, 5, label = iteration_time)
		worksheet.write(row_num, 6, label = MAE_128)
		worksheet.write(row_num, 7, label = RMSE_128)
		row_num += 1
	row_num += 1
	for txt in txt_names:
		f = open(folder+'/bs256test0831/test_results/'+txt)
		data_256 = f.read()
		iteration_time = re.findall(r'Iteration\s+(.*?):', data_256)[0]
		MAE_256 = re.findall(r'MAE = (.*?) ,', data_256)[0]
		RMSE_256 = re.findall(r'RMSE = (.*?) ', data_256)[0]
		alpha = re.findall(r'alpha_(.*?)@', txt)[0]
		gamma = re.findall(r'@rate_(.*?)\.txt', txt)[0]
		f.close()
		worksheet.write(row_num, 0, label = '100K')
		worksheet.write(row_num, 1, label = '256')
		worksheet.write(row_num, 2, label = folder)
		worksheet.write(row_num, 3, label = alpha)
		worksheet.write(row_num, 4, label = gamma)
		worksheet.write(row_num, 5, label = iteration_time)
		worksheet.write(row_num, 6, label = MAE_256)
		worksheet.write(row_num, 7, label = RMSE_256)
		row_num += 1
	row_num += 1
workbook.save('excel_test.xls')