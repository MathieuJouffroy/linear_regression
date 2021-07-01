import os
import csv
import numpy as np

def get_data(dataset):
    """
    Returns the input variable and the output variable from our dataset.
    The input variable is the 1st column and the output variable is the 2nd column.
    Note that you cannot loop over the reader more than once (it is a generator).
    You cannot do multiple list comprehension on a generator without re-declaring it.
    
    Arguments:
    dataset -- csv file containing the dataset
    
    Returns:
    mileage -- list of int (input variable)
    price -- list of int (output variable)
    """

    mileage = []
    price = []
    if (os.path.isfile(dataset)):
        with open(dataset, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                mileage.append(row[0])
                price.append(row[1])
        mileage.pop(0)
        price.pop(0)
        for i in range(len(mileage)):
            mileage[i] = eval(mileage[i])
            price[i] = eval(price[i])
    return mileage, price

def	get_thetas_and_fs_method(model):
    """
    Returns the weights (weight and biais) theta and the method used
    for feature scaling our features. The weights where updated while fitting
    our model. Note that we should used the same feature scaling method for
    the training and for the prediction.

    Arguments:
    model -- LinearRegression model (Class) 
    
    Returns:
    theta -- θ₀,θ₁ : Biais and Weight 
    fs_method -- feature scaling method
    """

    theta = []
    if (os.path.isfile(model)):
        with open(model, 'r') as csvfile:
            file = csv.reader(csvfile, delimiter=',')
            for row in file:
                theta = [float(row[0]), float(row[1])]
                fs_method = row[4]
    else:
        theta = [0.0, 0.0]
        fs_method = 'normalization'
    return (theta, fs_method)

def feature_scaling_normalization(args, X):
    """
    Returns the normalized input variable X.

    Arguments:
    args -- argument parser containing the method used for the computations
    X -- input variable
    
    Returns:
    X -- normalized input variable
    """

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
    """
    Returns the standardized input variable X.

    Arguments:
    args -- argument parser containing the method used for the computations
    X -- input variable
    
    Returns:
    X -- standardized input variable
    """

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

def save_model(theta, alpha, n_iters, method, file='./model.csv'):
    """
    Saves the model in a CSV file containing 5 columns.
    The first 2 columns are the biais and the weights. The 3rd column
    is the learning rate alpha, the 4th column is the number of iterations.
    And the last column is the feature scaling method used for training.

    Arguments:
    theta -- θ₀,θ₁ : Biais and Weight after fitting the model
    alpha -- learning rate used for training
    n_iters -- number of iterations for the gradient descent
    method -- feature scaling method used for training.
    """

    theta0, theta1 = np.float(theta[0]), np.float(theta[1])
    with open(file, 'w+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([theta0, theta1, alpha, n_iters, method])
        f.close()