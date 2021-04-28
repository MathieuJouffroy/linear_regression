import numpy as np
import csv
import os
from typing import List, Tuple

def get_data(dataset):
	mileage = []
	price = []
	if (os.path.isfile(dataset)):
		with open(dataset, 'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			# You cannot loop over the reader more than once (it is a generator)
			# cannot do multiple list comprehension on a generator without re-declaring it.
			for row in reader:
				mileage.append(row[0])
				price.append(row[1])
		mileage.pop(0)
		price.pop(0)
		# Ensure both list are correct with eval
		for i in range(len(mileage)):
			mileage[i] = eval(mileage[i])
			price[i] = eval(price[i])
	return mileage, price

def	get_thetas(model):
	theta = []
	if (os.path.isfile(model)):
		with open(model, 'r') as csvfile:
			file = csv.reader(csvfile, delimiter=',')
			for row in file:
				theta = [float(row[0]), float(row[1])]
				break
	else:
		theta = [0.0, 0.0]
	return (theta)

def feature_scaling_normalization(args, X):
	if args.method == 'vectorization':
		x_min = X.min()
		x_max = X.max()
		mean = X.mean()
		X = (X - mean) / (x_max - x_min)
	elif args.method == 'for_loop':
		x = []
		x_min = min(X)
		x_max = max(X)
		mean = sum(X) / len(X)
		for xi in X:
			x.append((xi - mean) / (x_max - x_min))
		X = x
	return X

def	feature_scaling_standardization(args, X):
	if args.method == 'vectorization':
		mean = X.mean()
		std = X.std()
		X = (X - mean) / std
	elif args.method == 'for_loop':
		x = []
		mean = sum(X) / len(X)
		variance = sum([((xi - mean) ** 2) for xi in X]) / len(X)
		std = variance ** 0.5
		for xi in X:
			x.append((xi - mean) / std)
		X = x
	return X

def save_model(theta, alpha, n_iters, file='./model.csv'):
	theta0, theta1 = np.float(theta[0]), np.float(theta[1])
	if os.path.isfile(file):
		print ("append")
		with open(file, 'a+') as f:
			writer = csv.writer(f)
			writer.writerow([theta0, theta1, alpha, n_iters])
			f.close()
	else:
		print ("write")
		with open(file, 'w+') as f:
			writer = csv.writer(f, delimiter=',')
			writer.writerow([theta0, theta1, alpha, n_iters])
			f.close()