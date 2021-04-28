import numpy as np
import matplotlib.pyplot as plt

def r2_score(model, mileages, prices):
	price_average = sum(prices) / len(prices)
	
	# sst = total sum of squares =  Σ(Yi – mean of Y)² 
	# it is a measure of the price variability
	sst = sum(map(lambda price: pow(price - price_average, 2), prices))
	
	# ssr = sum of square due to regression = Σ(Ŷi – mean of Y)² 
	ssr = sum(map(lambda mileage, price: pow(price - model.predict_(mileage), 2), mileages, prices))
	
	# r2_score = 1 - (sum of squares of residuals/total sum of squares )
	# r2_score = coefficient of determination
	# the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
	# “(total variance explained by model) / total variance.”
	# An R2 score near 1 means that the model is able to predict the data very well.
	return float(1 - (ssr / sst))

def	display_metrics(args, model, mileage, price):
	if args.method == 'vectorization':
		mse = model.vec_mse(mileage, price)
	elif args.method == 'for_loop':
		y_pred = []
		for xi in mileage:
			y_pred.append(model.predict_(xi))
		print (y_pred)
		mse = model.for_loop_mse(y_pred, price)
	r2score = r2_score(model, mileage, price)
	print (f"Mean Squarred error: {mse}")
	print (f"R2_score: {r2score}")
	print (f"Weight: {float(model._theta[1])}")
	print (f"Biais:{float(model._theta[0])}")

def show_model(args, X, X_fs, y, theta):
	plt.scatter(X, y, marker='x', c='red')
	if args.method == 'for_loop':
		line = [theta[1] * x + theta[0] for x in X_fs]
		plt.plot(X, line, c='navy', label=f"weight={theta[1]:.3f}, biais={theta[0]:.3f}")
	elif args.method == 'vectorization':
		plt.plot(X, theta[1] * X_fs + theta[0], c='navy', label=f"weight={float(theta[1]):.3f}, biais={float(theta[0]):.3f}")
	plt.title('Linear Regression')
	plt.xlabel('Mileage')
	plt.ylabel('Price')
	plt.legend()
	plt.grid()
	plt.show()
	
def show_cost_fct(J_history, theta0_hist, theta1_hist):
	fig1 = plt.figure(1)
	plt.plot(theta0_hist, label='$\\theta_{0}$', linestyle='--', color='blue')
	plt.plot(theta1_hist, label='$\\theta_{1}$', linestyle='-', color='navy')
	plt.xlabel('Iterations')
	plt.ylabel('$\\theta$', color='blue')
	fig1.legend()
	plt.show()
	fig2 = plt.figure(2)
	plt.plot(J_history, color='navy')
	plt.title('Cost function J(theta0, theta1)')
	plt.xlabel('iterations')
	plt.autoscale(axis='both')
	plt.show()