import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x, y = points
    df = pd.DataFrame(data=[[init_m, init_b, np.mean(error(np.polyval([init_m, init_b], x), y))]], columns=['m','b','error'])
    print("Starting gradient descent at b = {0:.5f}, m = {1:.5f}, error = {2:.5f}".format(
        df.m[0],df.b[0],df.error[0]))
    for _ in range(num_iterations):
        lr = df.tail(1)
        m = float(lr.m - learning_rate * np.mean(grad_m(x, y, lr.m, lr.b)))
        b = float(lr.b - learning_rate * np.mean(grad_b(x, y, lr.m, lr.b)))
        distance = np.mean(error(np.polyval([m, b], x), y))
        df.loc[df.shape[0]] = [m,b,distance]
    print("After {0} iterations b = {1:.5f}, m = {2:.5f}, error = {3:.5f}".format(
        num_iterations, float(df.tail(1).b),float(df.tail(1).m),float(df.tail(1).error)))
    ax.plot(df.m,df.b,df.error)
    plt.show()
    return [float(df.tail(1).m), float(df.tail(1).b)]

if __name__ == '__main__':
    dataset = pd.read_csv('temperature.csv', encoding='ISO-8859-1')

    dataset.dt = pd.to_datetime(dataset.dt)
    dataset.set_index('dt', inplace=True)
    dataset = dataset.groupby(pd.TimeGrouper("12M")).mean()
    dataset.index = dataset.index.year

    [m, b] = gradient_descent_runner([dataset.index, dataset.LandAverageTemperature], 0, 0, 1e-7, 1000)

    plt.plot(dataset.index, dataset.LandAverageTemperature)
    plt.plot(dataset.index, np.polyval([m, b], dataset.index))

    plt.show()
