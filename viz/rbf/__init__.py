from numpy import array


class RBF:
    def __init__(self, basic, metric):
        self.basic = basic
        self.metric = metric

    def __call__(self, y, x, d=None):
        if d is None: d = ()
        if len(d) == 0:
            r = self.metric(y, x)
            return self.basic(r)
        if len(d) == 1:
            r = self.metric(y, x)
            dr = self.metric(y, x, d)
            return self.basic(r, 1) * dr
        if len(d) == 2:
            d0 = (d[0],)
            d1 = (d[1],)
            r = self.metric(y, x)
            dr0 = self.metric(y, x, d0)
            dr1 = self.metric(y, x, d1)
            ddr = self.metric(y, x, d)
            return self.basic(r, 2) * dr0 * dr1 + self.basic(r, 1) * ddr
        raise "RBF only implements up to 2nd derivatives"


class Polynomials:
    def __init__(self, degree):
        self.degree = degree

    def _nchoosek(self, n, k):
        c = 1
        for i in range(k):
            c *= (n-i)
            c //= (i+1)
        return c

    def _degree_and_offset(self, n, k):
        i = 0
        while True:
            j = self._nchoosek(n-1+i, i)
            if k < j:
                return i, k
            k -= j
            i += 1

    def _multiindex(self, n, k):
        if n == 0:
            return ()
        if n == 1:
            return (k,)
        l, j = self._degree_and_offset(n, k)
        w = self._multiindex(n-1, j)
        r = sum(w)
        return (l-r,) + w

    def _counts(self, n, d):
        counts = [0] * n
        for i in d:
            counts[i] += 1
        return counts

    def _coeff_and_exp(self, n, k, d):
        alpha = list(self._multiindex(n, k))
        counts = self._counts(n, d)
        c = 1
        for j, diff in enumerate(counts):
            for i in range(diff):
                c *= alpha[j]
                alpha[j] -= 1
        return c, tuple(alpha)

    def __call__(self, x, d=None):
        if d is None: d = ()

        m, n = x.shape
        polys = self._nchoosek(self.degree+n, n)
        v = [None] * polys
        for i in range(polys):
            c, e = self._coeff_and_exp(n, i, d)
            for j, q in enumerate(e):
                c *= x[:, j]**max(0, q)
            v[i] = c
        return array(v).T

class NoPolynomials:
    def __call__(self, x, d=None):
        return np.array([])


class Derivative:
    "Wrapper to automatically differentiate an RBF or polynomial basis"
    def __new__(self, f, d):
        if isinstance(f, Derivative):
            return Derivative(f.fn, f.ds + (d,))
        return object.__new__(self)

    def __init__(self, f, d):
        self.fn = f
        self.ds = d if isinstance(d, tuple) else (d,)

    def __call__(self, x):
        return self.fn(x, d=self.ds)


def diff(f, d):
    "Differentiate a function"
    return Derivative(f, d)
