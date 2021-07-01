import numpy as np
import matplotlib.pyplot as plt
import math

def r2_score(model, mileages, prices):
    """
    Calculates the R2 score : the proportion of the variance in the dependent variable
    that is predictable from the independent variable(s). Also called the coefficient of determination.
    Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
    An R2 score near 1 means that the model is able to predict the data very well.
    R2 = (1 - (ssr/sst))
        Where sst = total sum of squares =  Σ(Yi – mean of Y)²
              ssr = sum of square errors (due to regression) = Σ(Ŷi – mean of Y)² 
        Note the Ŷ denotes the prediction from the model.
    
    Arguments:
    model -- LinearRegression model (Class)
    mileages -- numpy.ndarray or list containing the mileages (independent variable)
    prices -- numpy.ndarray or list containing the prices (dependent variable)
    
    Returns:
    r2 score -- float 
        (coefficient of determination) regression score function.
    """

    price_average = sum(prices) / len(prices)
    sst = sum(map(lambda price: pow(price - price_average, 2), prices))
    ssr = sum(map(lambda mileage, price: pow(price - model.predict_(mileage), 2), mileages, prices))
    return float(1 - (ssr / sst))

def	display_metrics(args, model, mileage, price):
    """
    Display the values for the weigh, the biais, the mean squarred error (mse),
    the root mse and the r2 score.
    
    Arguments:
    args -- argument parser containing the method for computing.
    model -- LinearRegression model (Class) 
    mileage -- numpy.ndarray or list containing the mileages (independent variable)
    prices -- numpy.ndarray or list containing the prices (dependent variable)
    
    Returns:
    r2 score -- float 
        (coefficient of determination) regression score function.
    """

    if args.method == 'vectorization':
        MSE = model.vec_mse(mileage, price)
    elif args.method == 'for_loop':
        y_pred = []
        for xi in mileage:
            y_pred.append(model.predict_(xi))
        MSE = model.for_loop_mse(y_pred, price)

    r2score = r2_score(model, mileage, price)
    print (f"\nWeight\t\t\t: {float(model._theta[1])}")
    print (f"Biais\t\t\t: {float(model._theta[0])}\n")
    print (f"Mean Squarred Error(MSE): {MSE}")
    print(f"Root MSE\t\t: {math.sqrt(MSE)}")
    print (f"R2 Score\t\t: {r2score}")

def show_model(args, X, X_fs, y, theta):
    """
    Display a scatter plot representing the data with a regression line.
    The regression line is the outcome of our training. It is the line
    that best fits our data.
    
    Arguments:
    args -- argument parser containing the method for computing.
    X -- input variable
    X_fs -- normalized input variable
    y -- output variable
    theta -- θ₀,θ₁ : Biais and Weight
    """

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
    """
    Display the values for the weigh, the biais, the mean squarred error (mse),
    the root mse and the r2 score.
    
    Arguments:
    J_history -- history for the cost function (at each iteration)
    theta0_hist -- history for the biais term (at each iteration)
    theta1_hist -- history for the weight term (at each iteration)
    """

    fig1 = plt.figure(1)
    plt.plot(theta0_hist, label='$\\theta_{0}$', linestyle='--', color='blue')
    plt.plot(theta1_hist, label='$\\theta_{1}$', linestyle='-', color='navy')
    plt.xlabel('Iterations')
    plt.ylabel('$\\theta$', color='blue')
    fig1.legend()

    fig2 = plt.figure(2)
    plt.plot(J_history, color='navy')
    plt.title('Cost function J(theta0, theta1)')
    plt.xlabel('iterations')
    plt.autoscale(axis='both')
    plt.show()