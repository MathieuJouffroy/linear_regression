import numpy as np

class LinearRegression():
    """ Linear Regression
    Parameters
    ----------
    alpha: float
      Learning rate (between 0.0 and 1.0)
    n_iters: int
      Number of iterations over the whole training dataset

    Attributes
    ----------
    _theta: list[float]
      θ₀,θ₁ weights after fitting the model
    """

    def __init__(self, alpha=0.01, n_iters=20):
        """ Set the values for our learning rate alpha and for the number of iterations. """
        self.alpha = alpha
        self.n_iters = n_iters

    def set_weigths(self, thetas=[0.0, 0.0]):
        """ Set the weights for our model. """
        self._theta = thetas

    def predict_(self, x):
        """ Linear regression function with a single explanatory variable X. """
        return self._theta[0] + (x * self._theta[1])

    def vec_mse(self, X, Y):
        """ Returns the vectorized Mean Squarred Error. """
        m = X.shape[0]
        X = np.c_[np.ones(m), X]
        Y_pred = X @ self._theta
        return np.square(Y_pred - Y).sum() / m

    def for_loop_mse(self, y_pred, y):
        """ Returns the Mean Squarred Error using a for loop. """
        error = 0.0
        m = len(y)
        for i in range (m):
            error += (y[i] - y_pred[i])**2
        return error / float(m)

    def vec_cost_elem(self, X, Y, theta):
        """ Vectorized individual losses (for each point) for the linear regression. """
        m = X.shape[0]
        Y_pred = X @ theta
        errors = Y_pred - Y
        return (1/(2*m))*np.transpose(errors)@(errors)

    def vec_cost(self, X, Y, theta):
        """ Returns the vectorized linear regression cost given a single explanatory variable X. """
        # hθ(x_i) = θ0 + (θ1 * x)
		# 1/2m Σ (hθ(x_i) - y_i )² = 1/2m Σ (θ.T(x_i) - y_i )²
		#                          = 1/2m (Xθ - Y).T (Xθ - Y)
        return np.sum(self.vec_cost_elem(X, Y, theta))

    def for_loop_cost(self, x, y, theta0, theta1):
        """ Returns the linear regression cost given a single explanatory variable X using a for loop. """
        loss = 0.0
        m = len(x)
        for xi, yi in zip(x, y):
            loss += (yi - (theta1*xi + theta0))**2
        return loss / m

    def vec_gradient_descent(self, X, Y):
        """
        Vectorized gradient descent for simple linear regression.
        Calculate the predictions given the hypothesis (linear regression function).
        The calculates the errors given our predictions and the output variable.
        Then we calculate the derivatives to know in which direction we should take
        a step (towards a minimum). And finally we update the weights and the biais term
        and cache in the cost at each iteration.
        
        Arguments:
        X -- input variable
        Y -- output variable
        
        Returns:
        theta -- θ₀,θ₁ : Biais and Weight after fitting the model
        J_history -- history for the cost function (at each iteration)
        t0_hist -- history for the biais term (at each iteration)
        t1_hist -- history for the weight term (at each iteration)
        """

        J_history = []
        t1_hist = []
        t0_hist = []
        m = X.shape[0]
        theta = np.array(self._theta, dtype='float64')
        theta = theta[:, np.newaxis]
        # 2-D Vectors, addind intercept term (bias)
        X = np.c_[np.ones(m), X]

        for iter in range(self.n_iters):
            t0_hist.append(float(theta[0]))
            t1_hist.append(float(theta[1]))
            # calculate derivatives
            errors = (X @ theta) - Y
            gradient = (X.T @ errors)
            # update theta
            theta -= (self.alpha/m) * gradient
            cost = self.vec_cost(X, Y, theta)
            J_history.append(cost)
            # Log Progress:
            if iter % 20 == 0:
                print (f'iter:{iter:d}	cost:{cost:.2f}	weight:{theta[1]}	bias:{theta[0]}')

        return (theta, J_history, t0_hist, t1_hist)

    def gradient_descent(self, x, y):
        """
        Gradient descent for simple linear regression using a for loop.
        Calculate the predictions given the hypothesis (linear regression function).
        The calculates the errors given our predictions and the output variable.
        Then we calculate the derivatives to know in which direction we should take
        a step (towards a minimum). And finally we update the weights and the biais term
        and cache in the cost at each iteration.
        
        Arguments:
        X -- input variable
        Y -- output variable
        
        Returns:
        theta -- θ₀,θ₁ : Biais and Weight after fitting the model
        J_history -- history for the cost function (at each iteration)
        t0_hist -- history for the biais term (at each iteration)
        t1_hist -- history for the weight term (at each iteration)
        """

        theta0 = self._theta[0]
        theta1 = self._theta[1]
        J_history = []
        t1_hist = []
        t0_hist = []
        m = len(x)

        for iter in range(self.n_iters):
            t0_hist.append(theta0)
            t1_hist.append(theta1)
            d_theta0 = 0
            d_theta1 = 0
            # calculate derivatives
            for xi, yi in zip(x, y):
                y_pred = theta0 + theta1 * xi
                errors =  y_pred - yi
                d_theta0 += errors
                d_theta1 += errors * xi
            # update thetas
            theta0 -= d_theta0 / m * self.alpha
            theta1 -= d_theta1 / m * self.alpha
            cost = self.for_loop_cost(x, y, theta0, theta1)
            J_history.append(cost)
            # Log Progress:
            if iter % 20 == 0:
                print (f'epoch:{iter:d}	cost:{cost:.2f}	weight:{theta1} bias:{theta0}')

        theta = list([theta0, theta1])
        return (theta, J_history, t0_hist, t1_hist)
