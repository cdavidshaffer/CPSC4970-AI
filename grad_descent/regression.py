from math import sqrt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler


def make_grid():
    # intercepts = np.arange(-1, 1, 0.05)
    # slopes = np.arange(0, 2, 0.05)
    intercepts = np.arange(-20, 20, 1)
    slopes = np.arange(0, 10, 0.1)
    return np.meshgrid(intercepts, slopes)


def mse_vec(d, p_vec):
    return [cost(d, [x, y]) for x, y in zip(p_vec[0], p_vec[1])]


def cost(d, parameters):
    error = 0
    for x, y in zip(d[:, 0], d[:, 1]):
        prediction = parameters[0] + parameters[1] * x
        error += (prediction - y) ** 2
    return sqrt(error) / len(d[:, 0])


def get_xyz_data(d):
    x, y = make_grid()
    z = np.array([mse_vec(d, p) for p in zip(x, y)])
    return x, y, z


def main():
    d = pd.read_csv("../data/synthetic_linear_1var.csv")
    d = d.dropna()
    x, y, z = get_xyz_data(d)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    cset = ax.contour(x, y, z, levels=[0, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], zdir='z', offset=0, cmap=cm.coolwarm)

    ax.set_zlabel('$\sqrt{\mathrm{mse}}$')
    ax.set_xlabel('intercept')
    ax.set_ylabel('slope')
    plt.show()


if __name__ == '__main__':
    main()
