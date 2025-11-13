import pandas as pd
import numpy as np

# Global data needed by mse and grad_mse
df = pd.read_csv('../data/synthetic_linear_1var.csv')
x, y = df['x'].values, df['y'].values.reshape(-1, 1)
X = np.array([np.ones(len(x)), x]).T


def mse(alpha):
    return (X @ alpha - y).T @ (X @ alpha - y) / len(y)


def grad_mse(alpha):
    return 2 / len(y) * X.T @ (X @ alpha - y)