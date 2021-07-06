import numpy as np
from scipy.spatial import Delaunay
from viz.state import WholeParser, InitialParser
from viz.rbf.shape import Sphere
from mayavi import mlab
import matplotlib.pyplot as plt


def periodize(x):
    return (x + 0.8) % 1.6 - 0.8


class StitchedEndo:
    def __init__(self, x, focus):
        self.update(x, focus)

    def _stitch(self, x):
        n, dims = x.shape
        dx = 1.6 * np.ones((n, 1)) @ np.array([[1.0, 0.0]])
        dz = 1.6 * np.ones((n, 1)) @ np.array([[0.0, 1.0]])
        return np.concatenate((x-dx-dz, x-dz, x-dx, x), axis=0)

    def _unstitch(self, k, x):
        n = x.shape[0]
        index = k % n
        grid = k // n
        xp = x[index, :]
        sx = grid % 2 - 1
        sz = grid // 2 - 1

        dx = 1.6 * sx.reshape((-1, 1)) @ np.array([[1.0, 0.0, 0.0]])
        dz = 1.6 * sz.reshape((-1, 1)) @ np.array([[0.0, 0.0, 1.0]])
        return xp + dx + dz

    def _triangulate(self, x):
        nrm = lambda x, y: np.sqrt(np.sum((x - y)**2, axis=1))
        d = Delaunay(x)
        s = d.simplices
        md = 0.05
        a, b, c = s[:, 0], s[:, 1], s[:, 2]
        u, v, w = x[a, :], x[b, :], x[c, :]
        r = (nrm(u, v) < md) * (nrm(u, w) < md) * (nrm(v, w) < md)
        return s[r, :]

    def _nearby(self, x, z):
        nrm = lambda x, y: np.max(np.abs(x-y), axis=1)
        n = x.shape[0]
        keep = nrm(x, z) < 0.8
        indices = np.arange(n)
        return indices[keep]

    def _update(self, x, z):
        zp = periodize(z)
        d = np.array([[1, 0], [0, 0], [0, 1]]) @ (zp - z)
        q = self._stitch(x[:, [0, 2]])
        k = self._nearby(q, zp)
        xp = self._unstitch(k, x)
        t = self._triangulate(xp[:, [0, 2]])
        return xp - d, t

    def update(self, x, focus):
        self.px, self.pt = self._update(x, focus[[0, 2]])
        self.x = x

    def __iter__(self):
        yield self.px, self.pt, {'color': (1, 1, 1)}


class Spheres:
    def __init__(self, x, f, options=None):
        n, m = x.shape
        self.nc = m // 3
        q = Sphere.sample(n)
        self.t = Sphere.triangulate(q)
        self.update(x, f)
        self.options = options

    def update(self, x, f):
        self.x = x
        self.f = f

    def __iter__(self):
        options = self.options
        for i in range(self.nc):
            xc = self.x[:, i::self.nc]
            mx = np.mean(xc, axis=0)
            d = mx[[0, 2]] - self.f[[0, 2]]
            dx = periodize(d) - d
            xc[:, [0, 2]] += dx
            yield xc, self.t, {} if options is None else options(i)


class Plotter:
    def __init__(self, focus):
        self.focus = focus
        fig = mlab.figure(size=(1080, 1080), bgcolor=(1, 1, 1))
        scene = fig.scene
        scene.renderer.set(use_depth_peeling=True)
        self.fig = fig

    def _update(self, center):
        # Gritty plot update details
        mlab.clf(figure=self.fig)  # Clear figure
        rbcs_done = False

        # Redraw objects
        for src in (self.rbcs, self.plts, self.endo):
            if src is None:  # For plots without RBCs, ignore that input
                rbcs_done = True
                continue
            for x, t, kwargs in src:
                if rbcs_done is False:
                    n = x.shape[0]//2
                mlab.triangular_mesh(x[:, 0], x[:, 1], x[:, 2], t, **kwargs)
            rbcs_done = True

        # Dark grey domain frame
        fx, fy, fz = center
        mlab.triangular_mesh(np.array([fx-0.8, fx-0.8, fx+0.8, fx+0.8]),
                             1 * np.array([1.2, 1.2, 1.2, 1.2]),
                             np.array([fz-0.8, fz+0.8, fz-0.8, fz+0.8]),
                             np.array([[0, 1, 2], [1, 3, 2]]),
                             color=(0.2, 0.2, 0.2))
        mlab.triangular_mesh(np.array([fx-0.8, fx-0.8, fx+0.8, fx+0.8]),
                             0 * np.array([1.2, 1.2, 1.2, 1.2]),
                             np.array([fz-0.8, fz+0.8, fz-0.8, fz+0.8]),
                             np.array([[0, 1, 2], [1, 3, 2]]),
                             color=(0.2, 0.2, 0.2))

    def update(self):
        rbc, plt, endo = (yield)
        center = self.focus(rbc, plt, endo)
        if rbc is not None:
            self.rbcs = Spheres(rbc, center, rbc_options)
        else:
            self.rbcs = None
        self.plts = Spheres(plt, center, plt_options)
        self.endo = StitchedEndo(endo, center)
        self._update(center)
        while True:
            rbc, plt, endo = (yield)
            center = self.focus(rbc, plt, endo)
            if rbc is not None:
                self.rbcs.update(rbc, center)
            self.plts.update(plt, center)
            self.endo.update(endo, center)
            self._update(center)


def plt_options(n):
    "Plotting options for platelets"
    c = 0.5 + 0.3 * n
    if n == 0:
        return {'color': (c, c, c), 'opacity': 0.8}
    return {'color': (c, c, c), 'opacity': 0.1}


def rbc_options(n):
    "Plotting options for RBCs"
    c = 0.4 + 0.6 * n / 7
    if n == 3:
        return {'color': (c, 0, 0), 'opacity': 0.8}
    return {'color': (c, 0, 0), 'opacity': 0.1}


def plt_focus(n):
    "Sets the camera focus on a certain platelet"
    def wrapped(rbc, plt, end):
        nc = plt.shape[1] // 3
        k = n % nc
        x = plt[:, k::nc]
        return np.mean(x, axis=0)
    return wrapped


def center(*args):
    return np.array([0.8, 0.6, 0.8])


def savefig(f, fig):
    map = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
    pfig = plt.figure(figsize=(9, 9))
    plt.imshow(map, zorder=4)
    plt.axis('off')
    plt.savefig(f, transparent=True, bbox_inches='tight')
    plt.close(pfig)


def main(filename):
    parser = WholeParser(sys.argv[1])
    focus = plt_focus(0)
    plotter = Plotter(focus)
    updater = plotter.update()
    updater.send(None)
    oriented = False
    rbc = None
    for i, frame in enumerate(parser):
        t, u, p, rbc, plt, endo = frame
        f0 = focus(rbc, plt, endo)
        focal = np.array([f0[0], 0.6, f0[2]])
        updater.send((rbc, plt, endo))
        mlab.view(azimuth=-15, elevation=-45, distance=5.6,
                  focalpoint=focal, roll=0)
        if not oriented:
            input(f"{t:4.2f}ms")
            oriented = True
        else:
            print(f"{t:4.2f}ms")
        savefig(f"frames/flyover{i:03d}.png", plotter.fig)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
