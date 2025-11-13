import numpy as np
from mse import mse, grad_mse


def gradient_descent(function, grad_function, initial_alpha, norm_tolerance=0.001, learning_rate=0.001):
    iteration = 0
    alpha = initial_alpha
    grad = grad_function(alpha)
    while np.linalg.norm(grad) > norm_tolerance:
        alpha = alpha - (learning_rate * grad)
        print(f"{iteration}:  {grad.T=}    {alpha.T=}   {function(alpha)=}")
        grad = grad_function(alpha)
        iteration += 1
    print(f"final:  {grad.T=}    {alpha.T=}   {function(alpha)=}")
    return alpha


def f(alpha):
    return (alpha[0] - 3) ** 2 + 5


def grad_f(alpha):
    return np.array([2 * (alpha[0] - 3)])


if __name__ == '__main__':
    gradient_descent(mse, grad_mse, np.array([[10], [10]]), norm_tolerance=0.0001, learning_rate=0.025)
