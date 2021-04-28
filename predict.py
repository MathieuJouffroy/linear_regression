from utils import get_thetas, get_data, feature_scaling_normalization, feature_scaling_standardization
from linear_regression import LinearRegression
import matplotlib.pyplot as plt
import sys
import argparse

def get_mileage():
	while (1):
		try:
			mileage = input(f'Enter a mileage: ')
		except EOFError:
			sys.exit('EOF on input. Exit..')
		except:
			sys.exit('Error on input. Exit.')
		try:
			mileage = float(mileage)
			if mileage >= 0:
				break
			else:
				print(f'Cannot predict the price for a negative mileage, input a positive number.\n')
		except ValueError:
			print (f'Input for mileage not valid, try again.\n')
	return (mileage)

def feature_scaling_input(args, mileages, mileage):
	if args.feature_scaling == 'normalization':
		x_min = min(mileages)
		x_max = max(mileages)
		mean = sum(mileages) / len(mileages)
		fs_mileage = (mileage - mean) / (x_max - x_min)
	elif args.feature_scaling == 'standardization':
		mean = sum(mileages) / len(mileages)
		variance = sum([((xi - mean) ** 2) for xi in mileages]) / len(mileages)
		std = variance ** 0.5
		fs_mileage = (mileage - mean) / std
	return (fs_mileage)

def main():
	parser = argparse.ArgumentParser(description='LR Gradient Descent')
	parser.add_argument('--feature_scaling', choices={"normalization", "standardization"}, help='Feature scaling method: Can either be normalization or standardization')
	args = parser.parse_args()

	mileages, price = get_data('./data.csv')
	theta = get_thetas('model.csv')
	mileage = get_mileage()
	model = LinearRegression()
	model.set_weigths(theta)

	if args.feature_scaling:
		mileage = feature_scaling_input(args, mileages, mileage)

	price = model.predict_(mileage)
	if price < 0:
		print(f'This car has no more value ({price:.3f}).')
	else:
		print(f'Estimated price: {price:.3f} $')

if __name__ == "__main__":
	main()