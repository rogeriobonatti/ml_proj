import os

for i in range(1):
	str_num= '%05d' %i
	filename_new= '/data/datasets/rbonatti/data_processed/1/valid/'+str_num+'.png'
	filename_old= '/data/datasets/rbonatti/data_processed/1/valid/'+str(i)+'_orig.png'
	os.rename(filename_old, filename_new)