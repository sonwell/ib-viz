from numpy import sqrt, exp, log, finfo, isinf, inf


class GMQ:
    """
    Generalized multiquadric

        phi(r) = (r**2 + epsilon^-2)**(beta/2),

    for beta odd and

        phi(r) = (r**2 + epsilon^-2)**(beta/2) * log(r**2 + epsilon**-2) / 2,

    otherwise.

    epsilon = infinity is handled specially; this with beta > 0 corresponds
    to a polyharmonic spline.
    """

    def __init__(self, beta, epsilon):
        self.beta = beta
        self.epsilon = epsilon

    def __call__(self, r, d=0):
        if isinf(self.epsilon) and 2 * d > self.beta:
            raise f"PHS is not smooth enough for order-{d} derivative"
        eps = finfo(float).eps
        a, b = self._coefficients(d)
        ex = self.beta - 2 * d
        s = sqrt(r ** 2 + self.epsilon ** -2)
        if a:
            return (a * log(s + eps) + b) * s ** ex
        return b * s ** ex

    def _transform(self, r):
        if isinf(self.epsilon):
            return r
        return sqrt(r**2 + self.epsilon**-2)

    def _coefficients(self, d):
        lsb = self.beta & 1
        a, b = lsb ^ 1, lsb
        for i in range(d):
            m = self.beta - 2 * i
            a, b = m * a, m * b + a
        return a, b


class PHS(GMQ):
    def __init__(self, beta):
        GMQ.__init__(self, beta, inf)


class IMQ(GMQ):
    def __init__(self, epsilon):
        GMQ.__init__(self, -1, epsilon)


class Gaussian:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __call__(self, r, d=0):
        return (-2)**d * self.epsilon**(2*d) * exp(-(self.epsilon*r)**2)


class Wendland:
    "Wendland compactly supported RBF"

    def __init__(self, l, k):
        p = self._binomial(l)
        for _ in range(k):
            p = self._integrate(p)
        self.poly = p / p.coeffs[0]

    def __call__(self, r, d=0):
        q = self.poly
        for _ in range(d):
            q = self._differentiate(q)
        return (r < 1) * q(r)

    class Polynomial:
        def __init__(self, coeffs):
            self.coeffs = coeffs

        def __call__(self, x):
            v = 0
            for c in reversed(self.coeffs):
                v = v * x + c
            return v

        def _axpy(self, alpha, other):
            shorter = list(self.coeffs)
            longer = list(other.coeffs)
            if len(shorter) > len(longer):
                shorter, longer = longer, shorter
            for i, coeff in shorter:
                longer[i] += alpha * coeff
            return Polynomial(longer)

        def __add__(self, other):
            self._axpy(1.0, other)

        def __sub__(self, other):
            self._axpy(-1.0, other)

        def __truediv__(self, other):
            coeffs = list(self.coeffs)
            for i in range(len(coeffs)):
                coeffs[i] /= other
            return Wendland.Polynomial(coeffs)

    def _binomial(selfi, l):
        coeffs = [0] * (l+1)
        c = 1
        for i in range(l+1):
            coeffs[i] = c
            c *= i-l
            c //= i+1
        return Wendland.Polynomial(coeffs)

    def _integrate(self, p):
        from math import gcd
        n = len(p.coeffs)
        coeffs = [0] * (n + 2)
        a = p.coeffs[0]
        sa = abs(a)
        coeffs[2] = -a

        f = 2
        for i in range(1, n):
            b = p.coeffs[i]
            sb = abs(b)
            coeffs[i+2] = -b

            if not b: continue

            sa = gcd((i+2) * sa, f * sb)
            f *= i+2
            sc = gcd(sa, f)
            sa //= sc
            f //= sc

        t = 0
        for i in range(2, n+2):
            sc = gcd(abs(coeffs[i]), i)
            coeffs[i] //= sc
            coeffs[i] *= f
            coeffs[i] //= (sa * (i // sc))
            t += coeffs[i]
        coeffs[0] = -t
        return Wendland.Polynomial(coeffs)

    def _differentiate(self, p):
        n = len(p.coeffs)
        coeffs = [0] * (n - 2)
        for i in range(n-2):
            coeffs[i] = (i+2) * p.coeffs[i+2]
        return Wendland.Polynomial(coeffs)


class Scaled:
    "Wraps and scales a basic function"

    def __new__(cls, phi, x, y=1):
        if isinstance(phi, Scaled):
            return Scaled(phi.phi, phi.x * x, phi.y * y)
        return super(Scaled, cls).__new__(cls)

    def __init__(self, phi, x, y=1):
        self.phi = phi
        self.x = x
        self.y = y

    def __call__(self, r, d=0):
        return self.y * self.x ** (2*d) * self.phi(self.x * r, d)
