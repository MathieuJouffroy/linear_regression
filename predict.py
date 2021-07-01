import sys
import argparse
import matplotlib.pyplot as plt
from lin_regression import LinearRegression
from utils import get_data, get_thetas_and_fs_method, feature_scaling_normalization, feature_scaling_standardization


def get_mileage():
    """ Prompt the user to input a mileage and returns the result as a float. """
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

def feature_scaling_input(method, mileages, mileage):
    """
    Returns the feature scaled mileage. It is very important that the feature 
    scaling method used is the same as the one used for training. 

    Arguments:
    args -- argument parser containing the method used for the computations
    method -- feature scaling method used for training
    mileages -- input variable (list of floats)
    mileage -- user input (int)
    
    Returns:
    fs_mileage -- feature scaled mileage
    """
    
    if method == 'normalization':
        x_min = min(mileages)
        x_max = max(mileages)
        mean = sum(mileages) / len(mileages)
        fs_mileage = (mileage - mean) / (x_max - x_min)
    elif method == 'standardization':
        mean = sum(mileages) / len(mileages)
        variance = sum([((xi - mean) ** 2) for xi in mileages]) / len(mileages)
        std = variance ** 0.5
        fs_mileage = (mileage - mean) / std
    return (fs_mileage)

def main():
    mileages, price = get_data('./data.csv')
    theta, method = get_thetas_and_fs_method('model.csv')
    mileage = get_mileage()
    mileage = feature_scaling_input(method, mileages, mileage)
    
    model = LinearRegression()
    model.set_weigths(theta)
    price = model.predict_(mileage)
    
    if price < 0:
        print(f'This car has no more value ({price:.3f}).')
    else:
        print(f'Estimated price: {price:.3f} $')

if __name__ == "__main__":
    main()