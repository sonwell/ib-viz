import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from viz.state import WholeParser, InitialParser
import scipy.fftpack as fft
import scipy.signal as signal


def angle(x):
    mx = np.mean(x, axis=0)
    y = x - mx
    v = y[463, :]
    a = np.arctan2((v[0] + v[2])/np.sqrt(2), v[1])
    return [mx[1], a]

def transform(t, plt):
    print(f"{t:0.2f}ms")
    nc = plt.shape[1] // 3
    p1 = plt[:, 0::nc]
    p2 = plt[:, 1::nc]
    return angle(p1) + angle(p2)


def pairs(x):
    yield from zip(x, x[1:])


def continuize(x):
    gap = 2 * np.pi
    current = 0
    c = 0 * x
    c[0] = x[0]
    for i, (a, b) in enumerate(pairs(x)):
        d = abs(a - b)
        if abs(a - b - gap) < d:
            current += gap
        elif abs(a - b + gap) < d:
            current -= gap
        c[i+1] = x[i+1] + current
    return c


def diff(t, x):
    return (x[1:] - x[:-1]) / (t[1:] - t[:-1])


def ave(x):
    return (x[1:] + x[:-1]) / 2


def data(filename):
    if filename[0] == 'r':
        parser, index = WholeParser(filename), 4
    else:
        parser, index = InitialParser(filename), 3
    return np.array([[f[0]] + transform(f[0], f[index]) for f in parser])


if __name__ == '__main__':
    import sys
    d = data(sys.argv[1])
    n, _ = d.shape
    t = d[:, 0]
    y1 = continuize(d[:, 2])
    y2 = continuize(d[:, 4])
    plt.plot(t, y1, t, y2)
    plt.show()
