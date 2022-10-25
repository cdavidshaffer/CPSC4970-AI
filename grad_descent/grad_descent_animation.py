import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from matplotlib import cm
from sklearn.preprocessing import StandardScaler

import regression

parameters = [10, 6]
learning_rate = 0.025
momemtum_coef = 0


fig, ((regression_line_axis, contour_plot_axis), (rmse_axis, _)) = plt.subplots(nrows=2, ncols=2,
                                                                                gridspec_kw={'hspace': 0.5})
regression_data_plot = None
contour_plot = None
data_set = None
epoch = 0
previous_delta = [0, 0]


def cost_gradient(d, parameters):
    total = [0, 0]
    for x, y in zip(d[:,0], d[:,1]):
        prediction = parameters[0] + parameters[1] * x
        total[0] += (prediction - y)
        total[1] += (prediction - y) * x
    return 2 * total[0] / len(d[:,0]), 2 * total[1] / len(d[:,0])


def update_parameter_est():
    global parameters
    global epoch
    global previous_delta
    epoch += 1
    grad = cost_gradient(data_set, parameters)
    delta = [learning_rate * grad[0] + momemtum_coef * previous_delta[0],
             learning_rate * grad[1] + momemtum_coef * previous_delta[1]]
    parameters = [parameters[0] - delta[0],
                  parameters[1] - delta[1]]
    previous_delta = delta
    print(epoch, parameters, regression.cost(data_set, parameters))


def init_animation():
    rmse_axis.set_xlim(0, 3000)
    rmse_axis.set_ylim(0, 10)
    return regression_line_plot + regression_data_plot + contour_plot.collections + contour_marker_plot + rmse_plot


def update_animation(frame):
    global parameters
    update_animation_regression_line()
    update_animation_contour_marker()
    update_rmse()
    update_parameter_est()
    return regression_line_plot + regression_data_plot + contour_plot.collections + contour_marker_plot + rmse_plot


def update_animation_regression_line():
    intercept, slope = parameters
    x_data = [data_set[0][0], data_set[-1][0]]
    y_data = [intercept + data_set[0][0] * slope, intercept + data_set[-1][0] * slope]
    regression_line_plot[0].set_data(x_data, y_data)


def update_animation_contour_marker():
    x_data, y_data = contour_marker_plot[0].get_data()
    if not isinstance(x_data, list): x_data = x_data.tolist()
    if not isinstance(y_data, list): y_data = y_data.tolist()
    intercept, slope = parameters
    x_data.append(intercept)
    y_data.append(slope)
    contour_marker_plot[0].set_data(x_data, y_data)


def update_rmse():
    x_data, y_data = rmse_plot[0].get_data()
    if not isinstance(x_data, list): x_data = x_data.tolist()
    if not isinstance(y_data, list): y_data = y_data.tolist()
    rmse = regression.cost(data_set, parameters)
    x_data.append(epoch)
    y_data.append(rmse)
    rmse_axis.set_xlim(min(x_data), max(x_data + [30]))
    rmse_axis.set_ylim(min(y_data)/1.1, max(y_data)*1.1)
    rmse_plot[0].set_data(x_data, y_data)


def create_contour(x, y, z, axis):
    axis.set_xlabel("intercept")
    axis.set_ylabel("slope")
    return axis.contour(x, y, z, levels=[0, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap=cm.coolwarm)


if __name__ == '__main__':
    data_set = pd.read_csv("../data/synthetic_linear_1var.csv")
    data_set = data_set.dropna()
    sc = StandardScaler().fit(data_set)
    data_set = sc.transform(data_set)
    #data_set = data_set.values
    x, y, z = regression.get_xyz_data(data_set)
    contour_plot = create_contour(x, y, z, contour_plot_axis)
    regression_data_plot = regression_line_axis.plot(data_set[:, 0], data_set[:, 1], 'ro')
    regression_line_plot = regression_line_axis.plot([], [], '-')
    contour_marker_plot = contour_plot_axis.plot([], [], 'rx')
    rmse_axis.set_xlabel('iteration')
    rmse_axis.set_ylabel('$\sqrt{\mathrm{mse}}$')
    rmse_axis.set_yscale('log')
    rmse_plot = rmse_axis.plot([], [], '-')
    ani = FuncAnimation(fig, update_animation,
                        init_func=init_animation,
                        interval=20)
    plt.show()
