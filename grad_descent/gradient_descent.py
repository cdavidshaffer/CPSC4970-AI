import numpy as np
import sklearn.datasets
import pandas as pd
from sklearn.preprocessing import StandardScaler


def squared_error(X, y, alpha):
    """"Returns the squared error for linear regression.
    X -- the feature matrix
    y -- the target (column) vector
    alpha -- the parameters (column) vector"""
    return (((X @ alpha) - y).T @ ((X @ alpha) - y))[0,0] / len(y)


def grad_squared_error_stoch(x, y, alpha):
    """"Returns the gradient (column) vector of the squared error function for linear regression.
    This version is for stochastic gradient descent.
    x -- a feature vector
    y -- a single target value
    alpha -- the parameters (column) vector"""
    return 2 * x * (alpha.T @ x - y)


def grad_squared_error(X, y, alpha):
    """"Returns the gradient (column) vector of the squared error function for linear regression.
    This version is for batch gradient descent.
    X -- the feature matrix
    y -- the target (column) vector
    alpha -- the parameters (column) vector"""
    return 2 / len(y) * X.T @ (X @ alpha - y)


def l2_regularization_function(alpha):
    tmp = np.copy(alpha)
    tmp[0, 0] = 0
    return np.linalg.norm(tmp, ord=2)


def grad_l2_regularization_function(alpha):
    tmp = 2 * alpha
    tmp[0, 0] = 0
    return tmp


def l1_regularization_function(alpha):
    tmp = np.copy(alpha)
    tmp[0, 0] = 0
    return np.linalg.norm(tmp, ord=1)


def grad_l1_regularization_function(alpha):
    tmp = np.zeros_like(alpha)
    for i in range(1, len(alpha)):
        if alpha[i, 0] > 0:
            tmp[i, 0] = 1
        elif alpha[i, 0] < 0:
            tmp[i, 0] = -1
    return tmp


def stoc_gradient_descent(X, y, initial_alpha,
                          cost_function, grad_cost_function,
                          regularization_function, grad_regularization_function,
                          learning_rate=0.1, momentum_coef=0.9, regularization_coef=0,
                          max_epochs=1000, norm_tolerance=0.001):
    alpha = initial_alpha
    previous_delta = np.zeros_like(initial_alpha)
    total_grad = np.ones_like(initial_alpha)  # force us into the loop
    epoch = 0
    while np.linalg.norm(total_grad, ord=1) > norm_tolerance and epoch < max_epochs:
        total_grad = np.zeros_like(initial_alpha)
        # for each sample...
        for i in range(len(y)):
            grad = 1 / len(y) * grad_cost_function(X[i:i + 1].T, y[i:i + 1].T, alpha)
            if regularization_coef > 0:
                grad = grad + regularization_coef * grad_regularization_function(alpha)
            total_grad = total_grad + grad
            delta = learning_rate * grad + momentum_coef * previous_delta
            alpha = alpha - delta
            previous_delta = delta
        if epoch % 100 == 0:
            print(f"{epoch}:  norm(gradient): {np.linalg.norm(total_grad, ord=1)}  "
                  f"MSE: {cost_function(X, y, alpha)}   "
                  f"MSE+reg: {cost_function(X, y, alpha) + regularization_coef * regularization_function(alpha)}")
        epoch += 1
    if epoch >= max_epochs:
        print("WARNING: FAILED TO CONVERGE")
    print(f"final:  norm(gradient): {np.linalg.norm(total_grad)}  "
          f"MSE: {cost_function(X, y, alpha)}   "
          f"MSE+reg: {cost_function(X, y, alpha) + regularization_coef * regularization_function(alpha)}")
    return alpha


def batch_gradient_descent(X, y, initial_alpha,
                           cost_function, grad_cost_function,
                           regularization_function, grad_regularization_function,
                           learning_rate=0.1, momentum_coef=0.9, regularization_coef=0,
                           max_epochs=1000, norm_tolerance=0.001):
    alpha = initial_alpha
    previous_delta = np.zeros_like(initial_alpha)
    grad = np.ones_like(initial_alpha)  # force us into the loop
    epoch = 0
    while np.linalg.norm(grad, ord=1) > norm_tolerance and epoch < max_epochs:
        grad = grad_cost_function(X, y, alpha)
        if regularization_coef > 0:
            grad = grad + regularization_coef * grad_regularization_function(alpha)
        delta = learning_rate * grad \
                + momentum_coef * previous_delta
        alpha = alpha - delta
        previous_delta = delta
        if epoch % 10000 == 0:
            print(f"{epoch}:  norm(gradient): {np.linalg.norm(grad, ord=1)}  "
                  f"MSE: {cost_function(X, y, alpha)}   "
                  f"MSE+reg: {cost_function(X, y, alpha) + regularization_coef * regularization_function(alpha)}")
        epoch += 1
    if epoch >= max_epochs:
        print("WARNING: FAILED TO CONVERGE")
    else:
        print(f"Converged after {epoch} epochs.")
    print(f"final:  norm(gradient): {np.linalg.norm(grad)}  "
          f"MSE: {cost_function(X, y, alpha)}   "
          f"MSE+reg: {cost_function(X, y, alpha) + regularization_coef * regularization_function(alpha)}")
    return alpha


def stoch_main():
    # df = sklearn.datasets.load_diabetes()
    # X = df.data
    # y = df.target.reshape(-1, 1)  # turn y into a column vector
    df = pd.read_csv('../data/housing.csv')
    df = df.dropna()
    del df['ocean_proximity']
    X = df.iloc[:, :-1].values
    y = df['median_house_value'].values
    y = y.reshape(-1, 1)
    X = StandardScaler().fit_transform(X)
    y = StandardScaler().fit_transform(y)
    X = np.insert(X, 0, 1, axis=1)

    params = stoc_gradient_descent(X, y, np.zeros((len(X[0]), 1)),
                                   squared_error, grad_squared_error_stoch,
                                   l1_regularization_function, grad_l1_regularization_function,
                                   learning_rate=2, momentum_coef=0.95, regularization_coef=0,
                                   max_epochs=1000, norm_tolerance=1e-20)
    print(params.T)


def batch_main():
    # df = sklearn.datasets.load_diabetes()
    # X = df.data
    # y = df.target.reshape(-1, 1)  # turn y into a column vector
    df = pd.read_csv('../data/housing.csv')
    print(df.columns)
    del df['ocean_proximity']
    X = df.iloc[:,:-1]
    y = df['median_house_value'].values
    y = y.reshape(-1, 1)
    X = np.insert(X, 0, 1, axis=1)

    params = batch_gradient_descent(X, y, np.zeros((len(X[0]), 1)),
                                    squared_error, grad_squared_error,
                                    l1_regularization_function, grad_l1_regularization_function,
                                    learning_rate=0.9, momentum_coef=0.99, regularization_coef=0,
                                    max_epochs=10000000, norm_tolerance=1e-6)
    print(params.T)


if __name__ == '__main__':
    stoch_main()
