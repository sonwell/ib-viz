import sys
import numpy as np
from mayavi import mlab
from plotting.sph2tri import triangulate
from plotting.reader import parse
import matplotlib.pyplot as plt
from viz.state import SimulationState
from viz.rbf.shape import RBC

N = 80
l = 1.6

class Plotter:
    def __init__(self, argv, prefix='frame'):
        self.reader = SimulationState(1, argv)
        self._view = None
        self.prefix = prefix

    def _initialize(self, x):
        n, m = x.shape
        self.t = RBC.triangulate(RBC.sample(n))
        self.x0 = x
        self._initialize = lambda x: None

    def _draw_rbc_1(self, x, dy):
        mlab.triangular_mesh(x[:, 0]+dy[0], x[:, 1]+dy[1], x[:, 2]+dy[2],
                             self.t, color=(1, 0, 0))

    def repeat(self):
        yield np.array([0, 0, 0])

    def draw_rbc(self, x):
        ym = x.T @ np.ones((x.shape[0],)) / x.shape[0]
        dy = (ym % l) - ym
        for d in self.repeat():
            self._draw_rbc_1(x, dy + l * d)

    def rbc(self, x):
        nc = x.shape[1] // 3
        for i in range(nc):
            y = x[:, i::nc]
            self.draw_rbc(y)

    def plot(self, frame, c):
        t, _, p, x = frame
        mlab.clf()
        self.rbc(x)
        if self._view is None:
            mlab.view(azimuth=90, elevation=-90, distance=4.8,
                      focalpoint=np.array([0.8, 0.8, 0.8]), roll=-45)
        else:
            mlab.view(*self._view[0], roll=self._view[1])
        return t

    def savefig(self, f):
        "Save a screenshot using matplotlib. It handles translucency better."

        map = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
        pfig = plt.figure(figsize=(16, 9))
        plt.imshow(map, zorder=4)
        plt.axis('off')
        plt.savefig(f, transparent=True, bbox_inches='tight')
        plt.close(pfig)

    def __iter__(self):
        c = 0
        for frame in self.reader:
            yield self.plot(frame)
            if self._view is None:
                self._view = mlab.view(), mlab.roll()
            self.savefig(f'frames/{self.prefix}{c:03d}.png')
            c += 1


if __name__ == '__main__':
    p = Plotter(sys.argv[1:])
    fig = mlab.figure(size=(1920, 1080))
    oriented = False
    for frame in p:
        if not oriented:
            input(f't = {frame:.2f} ms')
            oriented = True
        else:
            print(f't = {frame:.2f} ms')
    input('continue?')
