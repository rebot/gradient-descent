%matplotlib inline
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

matplotlib.rcParams['animation.writer'] = 'avconv'

x = np.linspace(1, 20, 30)
y = np.linspace(3, 50, 30) + 5

distances = np.vectorize(lambda x, y: (x - y) ** 2, otypes=[np.float])
distance = lambda x,y: np.sum(distances(x,y)) / float(len(x))

afg_b = np.vectorize(lambda x, y, b, m: 2 * ((b * x + m) - y) * (x))
afg_m = np.vectorize(lambda x, y, b, m: 2 * ((b * x + m) - y) * (1))

def gradient_decent_runner(points, initital_b, initial_m, learning_rate, num_iterations):
    fig = plt.figure()
    plots = []
    x,y = points
    p = np.array([initital_b,initial_m])
    for _ in range(num_iterations):
        error = distance(y, np.polyval(p, x))
        p = np.array([
            p[0] + learning_rate * (1 if np.sum(afg_b(x,y,p[0],p[1])) < 0 else -1),
            p[0] + learning_rate * (1 if np.sum(afg_m(x,y,p[0],p[1])) < 0 else -1)
        ])
        plots.append((plt.scatter(x,np.polyval(p, x)),))
    anim = animation.ArtistAnimation(fig, plots, interval=50, repeat_delay=3000, blit=True)
    HTML(anim.to_html5_video())
    return p

if __name__ == '__main__':
    [b, m] = gradient_decent_runner([x,y],0,0,0.05,50)
