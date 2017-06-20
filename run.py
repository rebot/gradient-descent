import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

error = np.vectorize(lambda x, y: (x - y) ** 2, otypes=[np.float64])

grad_m = np.vectorize(lambda x, y, m, b: 2 *
                      ((m * x + b) - y) * (x), otypes=[np.float64])
grad_b = np.vectorize(lambda x, y, m, b: 2 *
                      ((m * x + b) - y) * (1), otypes=[np.float64])


def gradient_descent_runner(points, init_m, init_b, learning_rate, num_iterations):
    """Gradient descent runner

    Args:
        points: a numpy 2D array containing x- and y-pairs.
        init_m: initial gradient.
        init_b: initial y-intercept.
        learning_rate: float number representing the speed of the learing algorithm
        num_interations: number of iterations before shutting down the optimalisation

    Returns:
        np.array([final_m,final_b])

    """
    x, y = points
    p = np.array([init_m, init_b])
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(
        p[1], p[0], np.mean(error(np.polyval(p, x), y))))
    for _ in range(num_iterations):
        p = np.array([
            p[0] - learning_rate * np.mean(grad_m(x, y, p[0], p[1])),
            p[0] - learning_rate * np.mean(grad_b(x, y, p[0], p[1]))
        ])
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(
        num_iterations, p[1], p[0], np.mean(error(np.polyval(p, x), y))))
    return p

if __name__ == '__main__':
    dataset = pd.read_csv('temperature.csv', encoding='ISO-8859-1')

    dataset.dt = pd.to_datetime(dataset.dt)
    dataset.set_index('dt', inplace=True)
    dataset = dataset.groupby(pd.TimeGrouper("12M")).mean()
    dataset.index = dataset.index.year

    [m, b] = gradient_descent_runner([dataset.index, dataset.LandAverageTemperature], 0, 0, 1e-7, 10000)

    plt.plot(dataset.index, dataset.LandAverageTemperature)
    plt.plot(dataset.index, np.polyval([m, b], dataset.index))

    plt.show()
