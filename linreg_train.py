import numpy as np
import argparse
from utils import *
from visualization import *
from lin_regression import LinearRegression

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Linear Regression Gradient Descent')
    parser.add_argument("--dataset", default='./data.csv', type=str, help='dataset to learn from')
    parser.add_argument('--feature_scaling', default='normalization', choices={"normalization", "standardization"}, help='feature scaling method')
    parser.add_argument('--method', default='vectorization', choices={"vectorization", "for_loop"},  help='method for computing Gradient Descent')
    parser.add_argument('--n_iters', default=300, type=int, help='number of iterations for the Gradient Descent')
    parser.add_argument('--alpha', default=0.5, type=float, help='learning Rate for your model (must be a float)')
    parser.add_argument('-a', action="store_true", help='accuracy and metrics')
    parser.add_argument('-v', action="store_true", help='visualize the model and cost function')
    args = parser.parse_args()

    # Load data and initialize the model
    mileage, price = get_data(args.dataset)
    model = LinearRegression(args.alpha, args.n_iters)
    model.set_weigths()

    if args.method == 'vectorization':
        # Transform data into 2-D float vectors
        mileage = np.array(mileage, dtype='float64')
        price = np.array(price, dtype='float64')
        mileage = mileage[:, np.newaxis]
        price = price[:, np.newaxis]

    # Feature scaling
    if args.feature_scaling == 'normalization':
        fs_mileage = feature_scaling_normalization(args, mileage)
    elif args.feature_scaling == 'standardization':
        fs_mileage = feature_scaling_standardization(args, mileage)

    # Run gradient descent
    if args.method == 'vectorization':
        theta, J_history, t0_hist, t1_hist = model.vec_gradient_descent(fs_mileage, price)
    elif args.method == 'for_loop':
        theta, J_history, t0_hist, t1_hist = model.gradient_descent(fs_mileage, price)
    
    # Save model
    save_model(theta, model.alpha, model.n_iters, args.feature_scaling)
    
    if args.a:
        model.set_weigths(theta)
        display_metrics(args, model, fs_mileage, price)
    if args.v:
        show_model(args, mileage, fs_mileage, price, theta)
        show_cost_fct(J_history, t0_hist, t1_hist)

if __name__ == "__main__":
    main() 