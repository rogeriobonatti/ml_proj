import os

for i in range(1000):
	str_num= '%04d' %i
	filename_new= '/data/datasets/rbonatti/data_processed/2/train/'+str_num
	filename_old= '/data/datasets/rbonatti/data_processed/2/train/'+str(i)
	os.rename(filename_old, filename_new)