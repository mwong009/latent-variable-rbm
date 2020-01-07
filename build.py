import sys
import numpy as np
import csv
import gzip
import pickle
import h5py
import pandas as pd
from utils import *

def build_dataset(data_file="santander.csv", evaluation=False):

	print("... loading " + data_file)
	data = pd.read_csv(data_file, delimiter=',')

	uid = data.uid.values
	n_samples = uid.shape[0] # number of sampels
	indicies = np.random.permutation(n_samples)
	data = data.loc[indicies]

	# prepare df
	#data_x = data.loc[:,'choice_aval':'choice_recibo']
	data_x = data.loc[:,'age':'e_acc']
	data_y = data.loc[:,'choice']

	data_x = data_x.values
	for i in range(data_x.shape[1]):
		data_x[:,i] =  scale_to_unit_interval(data_x[:,i])

	train_set_x = data_x[indicies[:int(n_samples*0.7)]]
	for i in range(train_set_x.shape[1]):
		train_set_x[:,i] =  train_set_x[:,i]
	train_set_y = data_y.values[indicies[:int(n_samples*0.7)]]

	valid_set_x = data_x[indicies[int(n_samples*0.7):]]
	for i in range(valid_set_x.shape[1]):
		valid_set_x[:,i] = valid_set_x[:,i]
	valid_set_y = data_y.values[indicies[int(n_samples*0.7):]]

	test_set_x = data_x[indicies[int(n_samples*0.9):]]
	for i in range(test_set_x.shape[1]):
		test_set_x[:,i] = test_set_x[:,i]
	test_set_y = data_y.values[indicies[int(n_samples*0.9):]]

	print("... compressing")
	# h5f = h5py.File(data_file + '.h5', 'w')
	h5f = h5py.File(data_file + '10.h5', 'w')
	h5f.create_dataset('train_set_x', data=train_set_x[:int(n_samples*0.1)], compression="gzip")
	h5f.create_dataset('train_set_y', data=train_set_y[:int(n_samples*0.1)], compression="gzip")
	h5f.create_dataset('valid_set_x', data=valid_set_x, compression="gzip")
	h5f.create_dataset('valid_set_y', data=valid_set_y, compression="gzip")
	h5f.create_dataset('test_set_x', data=test_set_x, compression="gzip")
	h5f.create_dataset('test_set_y', data=test_set_y, compression="gzip")
	h5f.close()
	print("... compression done!")



if __name__ == '__main__':
	build_dataset()
