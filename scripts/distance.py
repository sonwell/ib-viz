import os.path
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as spst
import scipy.fftpack as fft
import scipy.signal as signal
import scipy.optimize as opt
import scipy.stats as stats
from viz.state import WholeParser, InitialParser
from viz.rbf.shape import EuclideanSpace
metric = EuclideanSpace.metric


def dist(x, tree):
    d, a = tree.query(x)
    i = d.argmin()
    return i, a[i]
    r = metric(x, y)
    a = r.argmin()
    return a // r.shape[1], a % r.shape[1]


def angle(x):
    mx = np.mean(x, axis=0)
    y = x - mx
    cov = y.T @ y
    e, v = np.linalg.eig(cov)
    k = np.argsort(np.abs(e))
    w = v[:, k[0]]
    return 1.0 * (np.abs(w[0] - w[2]) > 0.9)


def transform(time, rbc, plt, endo):
    nplt = plt.shape[1] // 3
    if rbc is not None:
        nrbc = rbc.shape[1] // 3
        rbc2 = rbc.reshape((rbc.shape[0] * nrbc, 3), order='F')
        rbc2[:, [0, 2]] %= 1.6
        rtree = spst.cKDTree(rbc2)
    endo2 = endo[:, :]
    endo2[:, [0, 2]] %= 1.6
    etree = spst.cKDTree(endo2)

    data = [time]
    for i in range(nplt):
        x = plt[:, i::nplt]
        a = angle(x)
        x[:, [0, 2]] %= 1.6
        di = float('nan')
        if rbc is not None:
            pi, ri = dist(x, rtree)
            px = x[pi, :].reshape((1, 3))
            rx = rbc2[ri, :].reshape((1, 3))
            di = 10 * metric(px, rx)
        pj, ej = dist(x, etree)
        px = x[pj, :].reshape((1, 3))
        ex = endo2[ej, :].reshape((1, 3))
        dj = 10 * metric(px, ex)
        data += [a, di, dj]
    print(f"{time:.2f}ms", end='\r')
    return np.array(data, dtype=float)


class MockWholeParser(InitialParser):
    def __iter__(self):
        for time, u, p, plt, endo in super().__iter__():
            yield time, u, p, None, plt, endo


def data(filename):
    basename = os.path.basename(filename)
    if basename in ('fast-w2.bin', 'flat.bin', 'slow.bin') or \
            basename[0] == 'r':
        parser = WholeParser(filename)
    else:
        parser = MockWholeParser(filename)
    return basename, np.array([transform(time, rbc, plt, endo)
        for time, _, _, rbc, plt, endo in parser]) # if time <= 55.01])


def main(args):
    print("platelet\tunicycling distance\tmean distance\tpercent difference\tunicycling time")
    for file in args:
        basename, d = data(file)
        np.savetxt(basename.split('.', 1)[0] + '.dat', d, delimiter='\t')
        print(basename + ":")
        n = (d.shape[1] - 2) // 2

        fig, ax = plt.subplots(2, 2, sharex='col')
        fig.set_size_inches(11, 8.5)
        fig.suptitle('Platelets over bumpy wall')

        # Distances
        ax[0][0].set_title('Distance to nearest RBC')
        ax[0][0].set_ylabel('distance (μm)')
        for i in range(n):
            t = d[:, 0]
            a = d[:, 1 + 3 * i]
            r = d[:, 2 + 3 * i]
            l, = ax[0][0].plot(t, r)
            ax[0][0].plot(t, a, color=l.get_color(), linestyle='dashed')
        ax[0][0].set_ylim(0, 3)
        ax[1][0].set_title('Distance to endothelium')
        ax[1][0].set_ylabel('distance (μm)')
        ax[1][0].set_xlabel('time (ms)')
        for i in range(n):
            t = d[:, 0]
            a = d[:, 1 + 3 * i]
            r = d[:, 3 + 3 * i]
            ar = d[:, [1 + 3 * i, 2 + 3 * i]]
            dar = ar - np.mean(ar, axis=0)
            cov = dar.T @ dar
            l, = ax[1][0].plot(t, r)
            ax[1][0].plot(t, a, color=l.get_color(), linestyle='dashed')
            mr = np.mean(r)
            mra = mr if not np.sum(a) else np.sum(r * a) / np.sum(a)
            pe = (mra - mr) / mr
            print(f"    {i:4d}\t{mra:19f}\t{mr:14f}\t{pe * 100:18f}\t{np.mean(a) * 100:15f}")
        ax[1][0].set_ylim(0, 3)

        # Fourier decomp
        N = d[:, 0].shape[0]
        T = 0.0001
        t = fft.fftfreq(N, T)
        tf = fft.fftshift(t)

        ax[0][1].set_title('Time spent at distance')
        x = d[:, [2, 5]]
        if not np.any(np.isnan(x)):
            ax[0][1].hist(x, bins=550, range=(0, 3), cumulative=-1)
            ax[0][1].set_ylabel('timesteps')

        x = d[:, [3, 6]]
        if not np.any(np.isnan(x)):
            ax[1][1].hist(x, bins=550, range=(0, 3), cumulative=-1)
            ax[1][1].set_ylabel('timesteps')
        ax[1][1].set_xlabel('distance (μm)')

        plt.savefig(basename.split('.', 1)[0] + '.pdf')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

