import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from matplotlib import cm

import regression

parameters = [10, 8]
momentum_coef = 0
learning_rate = 0.064



fig, ((regression_line_axis, contour_plot_axis), (rmse_axis, _)) = plt.subplots(nrows=2, ncols=2,
                                                                                gridspec_kw={'hspace': 0.5})
regression_data_plot = None
contour_plot = None
data_set = None
epoch = 0
data_set_index = 0
last_rmse_update = 0
previous_delta = [0, 0]


def cost_gradient(x, y, S, parameters):
    total = [0, 0]
    prediction = parameters[0] + parameters[1] * x
    # print("\t", x, "\t", y, "\t", parameters, "\t", prediction, "\t", prediction - y)
    total[0] += (prediction - y)
    total[1] += (prediction - y) * x
    return 2 * total[0] / S, 2 * total[1] / S


def update_parameter_est():
    global parameters
    global epoch
    global data_set_index
    global previous_delta
    grad = cost_gradient(data_set['x'][data_set_index], data_set['y'][data_set_index], len(data_set['x']), parameters)
    # print(grad)
    delta = [learning_rate * grad[0] + momentum_coef * previous_delta[0],
             learning_rate * grad[1] + momentum_coef * previous_delta[1]]
    parameters = [parameters[0] - delta[0],
                  parameters[1] - delta[1]]
    previous_delta = delta
    if data_set_index + 1 >= len(data_set['x']):
        print(epoch, parameters, regression.cost(data_set.values, parameters))
        epoch += 1
        data_set_index = 0
    else:
        data_set_index += 1


def init_animation():
    for i in range(1000000): pass
    rmse_axis.set_xlim(0, 3000)
    #rmse_axis.set_ylim(0, 10)
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
    x_data = [0, 10]
    y_data = [intercept, intercept + 10 * slope]
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
    global last_rmse_update
    if epoch == last_rmse_update:
        return
    last_rmse_update = epoch
    x_data, y_data = rmse_plot[0].get_data()
    if not isinstance(x_data, list): x_data = x_data.tolist()
    if not isinstance(y_data, list): y_data = y_data.tolist()
    rmse = regression.cost(data_set.values, parameters)
    x_data.append(epoch)
    y_data.append(rmse)
    rmse_axis.set_xlim(min(x_data), max(x_data + [30]))
    rmse_axis.set_ylim(min(y_data)/1.1, 1.1*max(y_data))
    rmse_plot[0].set_data(x_data, y_data)


def create_contour(x, y, z, axis):
    axis.set_xlabel("intercept")
    axis.set_ylabel("slope")
    return axis.contour(x, y, z, levels=[0, 0.75, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap=cm.coolwarm)


if __name__ == '__main__':
    data_set = pd.read_csv("../data/synthetic_linear_1var.csv")
    data_set = data_set.dropna()
    x, y, z = regression.get_xyz_data(data_set.values)
    contour_plot = create_contour(x, y, z, contour_plot_axis)
    regression_data_plot = regression_line_axis.plot(data_set['x'], data_set['y'], 'ro')
    regression_line_plot = regression_line_axis.plot([], [], '-')
    contour_marker_plot = contour_plot_axis.plot([], [], 'rx')
    rmse_axis.set_xlabel('epoch')
    rmse_axis.set_ylabel('$\sqrt{\mathrm{mse}}$')
    rmse_axis.set_yscale('log')
    rmse_plot = rmse_axis.plot([], [], '-')
    ani = FuncAnimation(fig, update_animation,
                        init_func=init_animation,
                        interval=1)
    plt.show()
