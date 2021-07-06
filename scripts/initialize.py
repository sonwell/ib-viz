import sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix as csr
from scipy.spatial import cKDTree as kdtree

from viz.state import InitialParser, WholeParser
from viz.rbf.shape import Platelet, EuclideanSpace, FlatTorus, Sphere, Torus, \
                          RBC
from viz.rbf.basic import PHS
from viz.rbf import RBF, Polynomials


def neighbors(n, conn, k):
    if len(n) >= k:
        return n
    new = set(n)
    for i in n:
        new |= conn[i]
    return neighbors(new, conn, k)


def knn(t, x, k):
    "Use triangulation t to construct stencils"

    d = EuclideanSpace.metric
    n, dims = x.shape
    if not k:
        return np.zeros((n, 0))
    if n <= k:
        return np.ones((n, 1)) @ np.arange(n).reshape((1, n))

    conn = {i: {i} for i in range(n)}
    for i in range(t.shape[0]):
        r = t[i, :]
        for a, b in itertools.combinations(r, 2):
            conn[a].add(b)
            conn[b].add(a)

    res = np.zeros((n, k), dtype=int)
    for i in range(n):
        nb = list(neighbors(conn[i], conn, k))
        dist = d(x[nb, :], x[[i], :]).flatten()
        short = []
        for j, r in zip(nb, dist):
            if len(short) < k:
                short += [(r, j)]
            if r < short[-1][0]:
                short[-1] = (r, j)
            short = sorted(short, key=lambda x: x[0])
        res[i, :] = sorted([j for _, j in short])
        print(f'{i+1:05d}/{n:05d}', end='\r')
    print()
    return res


def project(x, c, t):
    "Project points to the plane tangent at point c"
    n, dims = x.shape
    # Get contravariant tangents. Commented out because this seems to give
    # results that do not converge. Need to explicitly pass contravariant
    # tangents instead.
    # t = solve(t.T @ t, t)
    return (x - np.ones((n, 1)) @ c.reshape((1, dims))) @ t


def interp(x, phi, poly):
    "Construct the (n + n_p) x (n + n_p) interpolation matrix"

    a = phi(x, x)
    b = poly(x)
    n = b.shape[1]
    z = np.zeros((n, n))
    m = np.concatenate((np.concatenate((a, b), axis=1),
                        np.concatenate((b.T, z), axis=1)), axis=0)
    return m


def rhs(y, x, phi, poly, d):
    """
    Construct the n x (n + n_p) right-hand side matrix. That is, the matrix of
    L Phi and L P values for linear operator L.
    """

    a = phi(y, x, d)
    b = poly(y, d)
    return np.concatenate((a, b), axis=1)


def solve(a, b):
    """
    Solve linear system a x = b for x and attempt to correct the numerical
    solution in case a has poor conditioning (not common).
    """

    c = np.linalg.solve(a, b)
    dc = np.linalg.solve(a, b - a @ c)  # Correction
    return c + dc


def operator(t, x, nn, d=()):
    "Use RBF-FD to construct differential operators"

    n = x.shape[0]
    k = nn.shape[1]
    op = np.zeros((n, n))

    basic = PHS(7)
    poly = Polynomials(7)
    phi = RBF(basic, EuclideanSpace.metric)

    for i in range(n):
        stencil = nn[i, :].flatten()
        f = project(x[stencil, :], x[i, :], t[:, :, i].T)
        z = np.zeros((1, f.shape[1]))
        a = interp(f, phi, poly)
        b = rhs(z, f, phi, poly, d)
        l = solve(a.T, b.T).T[:, :k]
        op[i, stencil] = l  # One row of the matrix
        print(f'{i+1:05d}/{n:05d}', end='\r')
    print()
    return csr(op)  # Change to sparse representation


def columns(x):
    if len(x.shape) == 1:
        return x[0], x[1], x[2]
    return x[:, 0], x[:, 1], x[:, 2]


def dot(x, y):
    l0, l1, l2 = columns(x)
    r0, r1, r2 = columns(y)
    return l0 * r0 + l1 * r1 + l2 * r2


def cross(x, y):
    l0, l1, l2 = columns(x)
    r0, r1, r2 = columns(y)
    return np.array([l1 * r2 - l2 * r1,
                     l2 * r0 - l0 * r2,
                     l0 * r1 - l1 * r0]).T


class Operators:
    def __init__(self, cls, n, k=80):
        p = cls.sample(n)
        tri = cls.triangulate(p)
        x = cls.shape(p)
        t = cls.tangents(p)
        nn = knn(tri, x, k)
        self.params = p
        self.tri = tri
        self.nn = nn  # Stencils
        self.d = [operator(t, x, nn, (0,)),
                  operator(t, x, nn, (1,))]  # Derivative operators


class Endothelium(FlatTorus):
    @staticmethod
    def shape(p):
        tau = 2 * np.pi
        scale = 1.6 / tau
        theta, phi = p[:, 0], p[:, 1]
        z = 0.075 + 0.1 * (np.cos((theta-phi)/2)*np.sin(phi))**2
        return np.array([scale * theta, scale * phi, z]).T


def rotate(x, n, ne):
    m, dims = x.shape
    axis = cross(n, ne).reshape((1, dims))
    c = dot(n, ne)
    l2 = dot(axis, axis)[0]
    l = np.sqrt(l2)
    axis /= l
    d = dot(axis, x).reshape((m, 1))
    w = cross(axis, x)
    return c * x + l * w + (1-c) * d @ axis


def test_packing(tree, xc, nc, xp, xt, nt, r, dmin=0.04):
    n = xp.shape[0]
    ones = np.ones((n, 1))
    z = rotate(xt.reshape((1, 3)), nt, -nc).flatten()
    y = rotate(xp, nt, -nc) + ones @ (xc - z + r * nc).reshape((1, 3))
    d, i = tree.query(y % 1.6)
    return np.any(d < dmin) * 1, 10 * y


class Packer:
    """
    Attempts to pack platelets in the region between the endothelium and the
    RBCs by
      1. Finding 2 points on the endothelium far enough apart
      2. Choosing random points on 2 platelets
      3. Aligning the platelet so that the normal at the randomly-chosen point
         opposes the normal on the point on the endothelium
      4. Place the platelet a random distance away from the endothelium along
         the line through the point on the endothelium and in the direction
         of the normal
    """


    def _initialize(self, me, mp):
        eops = Operators(Torus, me)
        pops = Operators(Sphere, mp)

        tau = 2 * np.pi
        scale = 1.6 / tau
        ones = scale * np.ones((me, 1))
        t0 = ones @ np.array([[0, 0, 1]])
        t1 = ones @ np.array([[1, 0, 0]])
        p = eops.params
        xe = scale * np.array([p[:, 1], 0 * p[:, 0], p[:, 0]]).T
        e0 = t0 - eops.d[0] @ xe
        e1 = t1 - eops.d[1] @ xe

        self.me = me
        self.mp = mp
        self.endo_ops = eops
        self.plt_ops = pops
        self.endo_corr = e0, e1  # Correction for the aperiodicity of Cartesian
                                 # coordinates of points on endothelium.
        self._initialize = lambda *x: None

    def _normals(self, x, ops, corr=None):
        if corr is None:
            d0 = ops.d[0] @ x
            d1 = ops.d[1] @ x
        else:
            d0 = ops.d[0] @ x + corr[0]
            d1 = ops.d[1] @ x + corr[1]
        jn = cross(d0, d1)
        j = np.sqrt(dot(jn, jn))
        return np.diag(1/j) @ jn

    class RandomState:
        def __init__(self, n, m, min, max):
            self.e = np.random.randint(n)
            self.p = np.random.randint(m)
            self.r = 0.1 * (min + (max - min) * np.random.rand(1)[0])

    def random_state(self, min, max):
        return self.RandomState(self.me, self.mp, min, max)

    N = 100000

    def _random_init(self, xp, tree, min, max, dmin=0.04):
        results = []
        for idx in range(self.N):
            x = 1.6 * np.random.rand(3)
            x[1] = (min + 1.25) / 10 + x[1] * (2.05 + max - min) / 16
            n = np.random.rand(3)
            n /= np.sqrt(np.sum(n**2))
            y = rotate(xp, n, np.array([0, 1, 0])) + x
            d = tree.query(y % 1.6)
            results.append([1.0 * (np.any(d) < dmin), 10 * x[1]])
            perc = (idx+1)/self.N
            rnd = int(perc * 80)
            rem = 80 - rnd
            print(f"[{'='*rnd}{' '*rem}] ({perc*100:5.1f}%)", end='\r')
        print()
        return np.array(results)

    def test(self, xe, xp, *xa, min=0.3, max=0.7, continuous=False):
        xb = np.concatenate((xe,) + tuple(xa), axis=0)
        tree = kdtree(xb % 1.6)
        if continuous:
            return self._random_init(xp, tree, min=min, max=max)
        me, mp = xe.shape[0], xp.shape[0]
        self._initialize(me, mp)
        en = self._normals(xe, self.endo_ops, self.endo_corr)
        pn = self._normals(xp, self.plt_ops)

        results = []
        for idx in range(self.N):
            st = self.random_state(min, max)
            i, j, r = st.e, st.p, st.r
            x, y = xe[i, :], xp[j, :]
            n, m = en[i, :], pn[j, :]
            d, z = test_packing(tree, x, n, xp, y, m, r)
            results.append([d, np.mean(z[:, 1])])
            perc = (idx+1)/self.N
            rnd = int(perc * 80)
            rem = 80 - rnd
            print(f"[{'='*rnd}{' '*rem}] ({perc*100:5.1f}%)", end='\r')
        print()
        return np.array(results)

    def __call__(self, xe, xp, *xa, min=0.3, max=0.7, continuous=False):
        xb = np.concatenate((xe,) + tuple(xa), axis=0)
        tree = kdtree(xb % 1.6)
        me, mp = xe.shape[0], xp.shape[0]
        self._initialize(me, mp)
        en = self._normals(xe, self.endo_ops, self.endo_corr)
        pn = self._normals(xp, self.plt_ops)
        xi, xj = None, None
        for _ in range(100):
            if xi is None: si = self.random_state(min, max)
            if xj is None: sj = self.random_state(min, max)
            xei, xej = xe[si.e, :], xe[sj.e, :]
            if np.sqrt(dot(xei-xej, xei-xej)) < 0.39:
                continue
            if xi is None:
                xi = test_packing(xb, xei, en[si.e, :],
                                  xp, xp[si.p, :], pn[si.p, :], si.r)
            if xj is None:
                xj = test_packing(xb, xej, en[sj.e, :],
                                  xp, xp[sj.p, :], pn[sj.p, :], sj.r)
            if xi is not None and xj is not None:
                break
        else:
            raise RuntimeError
        return (si, xi), (sj, xj)


def plot(hits, fname, min, max):
    SUCCESS = 0
    FAILURE = 1
    outcome = hits[:, 0]
    success = hits[outcome == SUCCESS, 1]
    failure = hits[outcome == FAILURE, 1]
    print(f"failures: {sum(outcome)/len(outcome)*100:5.2f}%")
    range = 0, max + 3.0
    fh, fe = np.histogram(failure, range=range, bins=100)
    sh, se = np.histogram(success, range=range, bins=100)
    plt.plot(0.5 * (se[:-1] + se[1:]), sh)
    plt.plot(0.5 * (fe[:-1] + fe[1:]), fh)
    plt.savefig(fname)
    plt.clf()


if __name__ == '__main__':
    import sys
    PLOT = True
    WRITE = False
    pack = Packer()
    reader = InitialParser(sys.argv[1])
    xp = Platelet.shape(Platelet.sample(900))
    rtri = Sphere.triangulate(Sphere.sample(2500))
    rng = np.random.default_rng()
    ts = 17 + np.cumsum(rng.poisson(30, size=10)) / 10
    it = iter(ts)
    min_time = next(it)
    print(f"waiting until ~{min_time:.2f}ms")
    count = 0
    for time, u, p, rbc, endo in reader:
        print(f"{time:.2f}ms")
        if time < min_time:
            continue
        r0 = rbc.reshape((2500 * 8, 3), order='F')

        if PLOT:
            try:
                lhits = pack.test(endo, xp, r0, min=0.3, max=4.0)
                shits = pack.test(endo, xp, r0, min=0.3, max=1.0)
                chits = pack.test(endo, xp, r0, min=0.3, max=1.0, continuous=True)
            except RuntimeError:
                continue
            plot(lhits, f'context{count}.png', 0.3, 4.0)
            plot(shits, f'result{count}.png', 0.3, 1.0)
            plot(chits, f'continuous{count}.png', 0.3, 1.0)
            min_time = next(it)
            print(f"waiting until ~{min_time:.2f}ms")

        if WRITE:
            tri = pack.plt_ops.tri
            nc = rbc.shape[1] // 3
            for i in range(nc):
                xi = rbc[:, i::nc]
                mx = np.mean(xi, axis=0)
                dx = mx % 1.6 - mx
                y = xi + np.ones((2500, 1)) @ dx.reshape((1, 3))
            z = np.concatenate((s1[1], s2[1]), axis=0).reshape((900, 6), order='F')
            writer = WholeParser(f'{sys.argv[1]}.{count}', 'wb')
            writer.write(0, u, p, rbc, z, endo)
        count += 1
