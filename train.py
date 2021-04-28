import numpy as np
import argparse
from utils import *
from linear_regression import LinearRegression
from visu import *
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser(description='LR Gradient Descent')
	parser.add_argument("--dataset", default='./data.csv', type=str, help='Dataset to learn from')
	parser.add_argument('--method', default='vectorization', choices={"vectorization", "for_loop"},  help='Method for computing gradient descent. It can either be vectorization or for_loop')
	parser.add_argument('--feature_scaling', choices={"normalization", "standardization"}, help='Feature scaling method: Can either be normalization or standardization')
	parser.add_argument('-a', '--accuracy', action="store_true", help='Accuracy and metrics')
	parser.add_argument('-v', '--visual', action="store_true", help='Visualize the model and cost function')
	parser.add_argument('--n_iters', default=200, type=int, help='Number of iterations for the Gradient Descent')
	parser.add_argument('--alpha', default=0.5, type=float, help='Learning Rate for your model')
	args = parser.parse_args()

	mileage, price = get_data('./data.csv')
	model = LinearRegression(args.alpha, args.n_iters)
	model.set_weigths()

	plt.scatter(mileage, price, marker='x', c='red')
	plt.xlim(min(mileage) - 10000, max(mileage)+10000)
	plt.show()

	if args.method == 'vectorization':
		# Transform data into float vectors 
		mileage = np.array(mileage, dtype='float64')
		price = np.array(price, dtype='float64')
		mileage = mileage[:, np.newaxis]
		price = price[:, np.newaxis]

	# Feature Scaling
	if args.feature_scaling:
		if args.feature_scaling == 'normalization':
			fs_mileage = feature_scaling_normalization(args, mileage)
		elif args.feature_scaling == 'standardization':
			fs_mileage = feature_scaling_standardization(args, mileage)
	else:
		fs_mileage = mileage

	# Run gradient Descent
	if args.method == 'vectorization':
		theta, J_history, t0_hist, t1_hist = model.vec_gradient_descent(fs_mileage, price)
	elif args.method == 'for_loop':
		theta, J_history, t0_hist, t1_hist = model.gradient_descent(fs_mileage, price)
		
	save_model(theta, model.alpha, model.n_iters)
	
	if args.accuracy:
		model.set_weigths(theta)
		display_metrics(args, model, fs_mileage, price)
	if args.visual:
		show_model(args, mileage, fs_mileage, price, theta)
		show_cost_fct(J_history, t0_hist, t1_hist)

if __name__ == "__main__":
	main() 