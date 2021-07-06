from numpy import ceil, sin, cos, arcsin, arccos, abs, sqrt, arange, array, \
    sum, pi, mgrid, ones, zeros, concatenate
import scipy.spatial as spst
from mayavi import mlab

class Sphere:
    @staticmethod
    def _parse_point_set(filename):
        """
        Read a plaintext file with

            x y z

        on each line and transform into spherical surface coordinates.

        For node sets, see:
            https://web.maths.unsw.edu.au/~rsw/Sphere/MaxDet/
            https://web.maths.unsw.edu.au/~rsw/Sphere/Energy/
            https://github.com/gradywright/spherepts
        """
        with open(filename) as fd:
            content = fd.read()
        lines = content.strip().split('\n')
        x = np.array([list(map(float, l.strip().split())) for l in lines])
        theta = np.arctan2(x[:, 1], x[:, 0])
        phi = np.arcsin(x[:, 2])
        return np.array([theta, phi]).T

    @staticmethod
    def sample(n):
        "Bauer spiral spherical point sampling"

        z = -1 + (2 * arange(n) + 1) / n
        phi = arcsin(z)
        theta = (sqrt(n * pi) * phi) % (2 * pi)
        return array([theta, phi]).T

    @staticmethod
    def shape(p):
        theta, phi = p[:, 0], p[:, 1]
        x = cos(theta) * cos(phi)
        y = sin(theta) * cos(phi)
        z = sin(phi)
        return array([x, y, z]).T

    @staticmethod
    def triangulate(p):
        """
        Transform Cartesian coordinates to stereographic coordinates and
        triangulate the points there. The convext hull surrounds the
        stereographic pole, so we then triangulate the points in the hull and
        concatenate the triangulations.
        """

        x = Sphere.shape(p)
        dims = x.shape[1]
        ref = x[0, :].reshape((1, dims))
        others = x[1:, :]
        dots = -others @ ref.T
        t = 1 / (1 + dots)
        p = array([[-sin(p[0, 0]), cos(p[0, 0]), 0],
                   [-cos(p[0, 0]) * sin(p[0, 1]),
                    -sin(p[0, 0]) * sin(p[0, 1]),
                     cos(p[0, 1])]]).T
        y = t * others @ p
        tri = spst.Delaunay(y)
        hull = tri.convex_hull + 1
        path = array(list(set(hull.flatten().tolist())) + [0], dtype=int)
        z = x[path, :] @ p
        htri = spst.Delaunay(z)
        hsim = array(path[htri.simplices], dtype=int)
        return concatenate((tri.simplices[:, [0, 2, 1]]+1, hsim), axis=0)

    @staticmethod
    def tangents(p):
        return array([
            [-sin(p[:, 0]), cos(p[:, 0]), 0 * p[:, 1]],
            [-cos(p[:, 0]) * sin(p[:, 1]),
             -sin(p[:, 0]) * sin(p[:, 1]),
             cos(p[:, 1])]])

    class Metric:
        """
        Computes distances and their derivatives between points on the sphere
        in a vectorized manner.
        """

        def _cycle(self, theta):
            s, c = sin(theta), cos(theta)
            return s, c, -s, -c

        def _eval(self, y, x, counts):
            m, k = y.shape
            n, _ = x.shape

            total = 1.0
            continues = True

            for i, c in enumerate(counts):
                cyc = self._cycle(y[:, i])
                ss, cs = cyc[c % 4], cyc[(c+1) % 4]
                sd, cd = sin(x[:, i]), cos(x[:, i])
                total *= cs * cd
                if continues:
                    total += ss * sd
                if c:
                    continues = False
            return total

        def _count(self, d, n):
            counts = [0] * n
            for i in d:
                counts[i] += 1
            return counts

        def __call__(self, y, x, d=None):
            if d is None: d = ()

            m, k = y.shape
            n, _ = x.shape
            i, j = mgrid[0:m, 0:n]
            yr = array([y[i, l].flatten() for l in range(k)]).T
            xr = array([x[j, l].flatten() for l in range(k)]).T
            counts = self._count(d, k)

            v = self._eval(yr, xr, counts)

            if len(d) == 0:
                return sqrt(2*abs(1-v)).reshape((m, n))
            return -v.reshape((m, n))

    metric = Metric()


class RBC(Sphere):
    "Red blood cell"

    @staticmethod
    def shape(p):
        r = lambda x: 0.5 * (0.21 + 2.0 * x - 1.12 * x**2)
        theta, phi = p[:, 0], p[:, 1]
        x = cos(theta) * cos(phi)
        y = sin(theta) * cos(phi)
        r2 = x**2 + y**2
        z = sin(phi) * r(r2)
        return 0.391 * array([x, y, z]).T


class Platelet(Sphere):
    @staticmethod
    def shape(p):
        major = 0.155
        minor = 0.05
        theta, phi = p[:, 0], p[:, 1]
        x = cos(theta) * cos(phi)
        y = sin(theta) * cos(phi)
        z = sin(phi)
        return array([major * x, minor * y, major * z]).T


class Torus:
    @staticmethod
    def sample(n):
        l = ceil(sqrt(n))
        z = arange(n) / n
        phi = 2 * pi * z
        theta = (l * phi) % (2 * pi)
        return array([theta, phi]).T

    @staticmethod
    def shape(p):
        return array([cos(p[:, 0]), sin(p[:, 0]),
                      cos(p[:, 1]), sin(p[:, 1])]).T

    @staticmethod
    def triangulate(p):
        """
        Concatenate 9 shifted copies of the coordinate square and triangulate
        those points. Then we take the triangles that use points in the center
        square, taking care to make the triangulation periodic.
        """

        n = p.shape[0]
        tau = 2 * pi
        dx = ones((n, 1)) @ array([[tau, 0]])
        dy = ones((n, 1)) @ array([[0, tau]])
        q = concatenate((p-dx-dy, p   -dy, p+dx-dy,
                         p-dx   , p      , p+dx   ,
                         p-dx+dy, p   +dy, p+dx+dy), axis=0)
        tri = spst.Delaunay(q).simplices
        nout = sum((tri >= 4 * n) * (tri < 5 * n), axis=1)
        parts = {}
        for row in tri[nout >= 1, :] % n:
            std = tuple(sorted(row))
            if std not in parts:
                parts[std] = row.tolist()
        return array(list(parts.values()))

    @staticmethod
    def tangents(p):
        return array([
            [-sin(p[:, 0]), cos(p[:, 0]), 0 * p[:, 1], 0 * p[:, 1]],
            [0 * p[:, 0], 0 * p[:, 0], -sin(p[:, 1]), cos(p[:, 1])]
        ])

    class Metric:
        """
        Toroidal metric based on the 4-dimensional torus

            [cos theta, sin theta, cos phi, sin phi].

        The resulting metric is homogeneous, i.e., dist(x, y) = f(dot(x, y))
        for some f.
        """

        def _cycle(self, theta):
            s, c = sin(theta), cos(theta)
            return s, c, -s, -c

        def _count(self, d, n):
            counts = [0] * n
            for i in d:
                counts[i] += 1
            return counts

        def __call__(self, y, x, d=None):
            if d is None: d = ()

            m, k = y.shape
            n, _ = x.shape
            i, j = mgrid[0:m, 0:n]
            yr = array([y[i, l].flatten() for l in range(k)]).T
            xr = array([x[j, l].flatten() for l in range(k)]).T
            if len(d) == 0:
                t = 0
                for i in range(k):
                    t += 1 - cos(yr[:, i] - xr[:, i])
                return sqrt(2 * t / k).reshape((m, n))
            else:
                std = d[0]
                for c in d[1:]:
                    if c != std:
                        return 0
                cyc = self._cycle(yr[std] - xr[std])
                return cyc[(3 + len(d)) % 4].reshape((m, n)) / n

    metric = Metric()


class FlatTorus(Torus):
    pass


class EuclideanSpace:
    class Metric:
        def _count(self, d, n):
            counts = [0] * n
            for i in d:
                counts[i] += 1
            return counts

        def __call__(self, y, x, d=None):
            if d is None: d = ()

            m, k = y.shape
            n, _ = x.shape
            i, j = mgrid[0:m, 0:n]
            yr = array([y[i, l].flatten() for l in range(k)]).T
            xr = array([x[j, l].flatten() for l in range(k)]).T
            counts = self._count(d, k)

            if len(d) == 0:
                dist = 0
                for l in range(k):
                    dist += (yr[:, l] - xr[:, l])**2
                return sqrt(dist.reshape((m, n)))
            if len(d) == 1:
                l = d[0]
                return yr[:, l].reshape((m, n))
            if len(d) == 2:
                if d[0] == d[1]:
                    return ones((m, n))
            return zeros((m, n))

    metric = Metric()
