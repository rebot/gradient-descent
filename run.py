import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x = np.linspace(1, 20, 30)
y = np.linspace(3, 50, 30) + 5

error = lambda x,y: np.mean(np.vectorize(lambda x, y: (x - y) ** 2, otypes=[np.float64]))

grad_m = np.vectorize(lambda x, y, m, b: 2 * ((m * x + b) - y) * (x), otypes=[np.float64])
grad_b = np.vectorize(lambda x, y, m, b: 2 * ((m * x + b) - y) * (1), otypes=[np.float64])

def gradient_decent_runner(points, initital_b, initial_m, learning_rate, num_iterations):
    x,y = points
    p = np.array([initital_b,initial_m])
    for _ in range(num_iterations):
        p = np.array([
            p[0] + learning_rate * np.mean(grad_m(x,y,p[0],p[1])),
            p[0] + learning_rate * np.mean(grad_b(x,y,p[0],p[1]))
        ])
    plt.plot(x,y,'r',x,np.polyval(p, x),'g')
    plt.show()
    return p

if __name__ == '__main__':
    [b, m] = gradient_decent_runner([x,y],0,0,0.01,1000)
