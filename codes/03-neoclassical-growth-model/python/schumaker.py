""" 
SCHUMAKER SHAPE-PRESERVING QUADRATIC SPLINE

AUTHOR: Seho Jeong, Sogang University
DATE: 2025-09-12
REFERENCES:
- Hong, Jay H. 2025. "Lectures on Topics in Macroeconomics." Seoul National University.
- Rezende, Leonardo. 2001. "Schumaker.m" https://www.econ.puc-rio.br/lrezende/
- Schumaker, Larry L. 1983. "On Shape Preserving Quadratic Spline Interpolation" SIAM Journal of Numerical Analysis 20(4): 854-864.
"""
import numpy as np

class SchumakerSpline1983:
    """
    Shape-preserving quadratic spline (Schumaker, 1983).
    Port of the MATLAB function you provided.
    - x: strictly increasing 1D array
    - y: values at x
    - slopes: optional endpoint slopes at ALL nodes; if None, estimate like the MATLAB code
    The spline is C^1 and piecewise quadratic; each interval may be split by
    one interior knot, depending on data and endpoint slopes.
    """

    def __init__(self, x, y, slopes=None, atol=1e-12):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        assert x.ndim == y.ndim == 1 and x.size == y.size and x.size >= 2
        if not np.all(np.diff(x) > 0):
            raise ValueError("x must be strictly increasing.")
        self._xmin, self._xmax = x[0], x[-1]
        self.x = x
        self.y = y
        n = x.size
        h = np.diff(x)
        dy = np.diff(y)
        d = dy / h  # secant slopes

        # --- Estimate slopes like the MATLAB code if not provided ---
        if slopes is None:
            if n == 2:
                s = np.array([d[0], d[0]], dtype=float)  # linear case
            else:
                # chord lengths
                L = np.sqrt(h**2 + dy**2)
                # sb(j) for j=1..n-2 (MATLAB 1-index); indicator(d_j * d_{j+1}>0)
                ind = (d[:-1] * d[1:]) > 0
                sb = np.zeros(n-2, dtype=float)
                sb[ind] = (L[:-1][ind] * d[:-1][ind] + L[1:][ind] * d[1:][ind]) / (L[:-1][ind] + L[1:][ind])

                s = np.empty(n, dtype=float)
                s[1:-1] = sb
                s[0]    = 0.5 * (3*d[0]    - sb[0]) if n > 2 else d[0]
                s[-1]   = 0.5 * (3*d[-1]   - sb[-1]) if n > 2 else d[-1]
        else:
            s = np.asarray(slopes, dtype=float)
            if s.shape != x.shape:
                raise ValueError("slopes must have the same shape as x and y.")

        self.slopes = s

        # --- Build pieces (possibly 2 per interval) following MATLAB "Schum" ---
        starts = []
        ends   = []
        coefs  = []  # rows store [C, B, A] for y = A + B*dx + C*dx^2 (dx = x - start)

        for i in range(n-1):
            t1, t2 = x[i], x[i+1]
            F1, F2 = y[i], y[i+1]
            f1, f2 = s[i], s[i+1]
            hi = t2 - t1
            di = (F2 - F1) / hi

            # Test if a single quad can match both end slopes on [t1,t2]
            if np.isclose((f1 + f2) * hi, 2*(F2 - F1), atol=atol):
                # tsi = t2 ; single piece, b = 0 in MATLAB
                a = hi
                # fbar from MATLAB
                fbar = (2*(F2 - F1) - (a*f1 + 0.0*f2)) / hi  # = 2*di - f1
                C = (fbar - f1) / (2*a)
                B = f1
                A = F1
                starts.append(t1); ends.append(t2); coefs.append([C, B, A])
                continue

            # Else choose interior knot tsi
            if (f1 - di) * (f2 - di) >= 0:
                tsi = 0.5*(t1 + t2)
            elif abs(f2 - di) < abs(f1 - di):
                tsi = t1 + hi * (f2 - di) / (f2 - f1)
            else:
                tsi = t2 + hi * (f1 - di) / (f2 - f1)

            a = tsi - t1
            b = t2 - tsi
            # fbar as in MATLAB
            fbar = (2*(F2 - F1) - (a*f1 + b*f2)) / hi

            # Left quad on [t1, tsi]
            C1 = (fbar - f1) / (2*a)
            B1 = f1
            A1 = F1
            starts.append(t1); ends.append(tsi); coefs.append([C1, B1, A1])

            # Right quad on [tsi, t2]
            if b > 0:
                # value at knot = S * [a^2; a; 1] (MATLAB)
                y_knot = C1*(a*a) + B1*a + A1
                C2 = (f2 - fbar) / (2*b)
                B2 = fbar
                A2 = y_knot
                starts.append(tsi); ends.append(t2); coefs.append([C2, B2, A2])
            # if b == 0, no right piece (rare; tsi==t2)

        self.starts = np.array(starts)
        self.ends   = np.array(ends)
        self.coefs  = np.array(coefs)  # each row [C, B, A]

    def _locate(self, xq):
        if np.any((xq < self._xmin) | (xq > self._xmax)):
            raise ValueError("Query point outside interpolation domain.")
        # piece index j with xq in [starts[j], ends[j]]
        j = np.searchsorted(self.ends, xq, side='right')
        return np.clip(j, 0, len(self.starts)-1)

    def __call__(self, xq):
        xq = np.asarray(xq, dtype=float)
        j = self._locate(xq)
        dx = xq - self.starts[j]
        C = self.coefs[j, 0]; B = self.coefs[j, 1]; A = self.coefs[j, 2]
        return A + B*dx + C*dx*dx

    def deriv(self, xq):
        xq = np.asarray(xq, dtype=float)
        j = self._locate(xq)
        dx = xq - self.starts[j]
        C = self.coefs[j, 0]; B = self.coefs[j, 1]
        return B + 2*C*dx

