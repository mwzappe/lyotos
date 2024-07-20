import cupy as cp

from lyotos.util import batch_dot

class Sphere:
    @classmethod
    def intersect(cls, R, p, d):
        
        dpd = batch_dot(p,d)
        dpp = batch_dot(p,p)

        a = 1
        b = 2 * dpd
        c = dpp - R**2

        dsc = cp.sqrt(b**2 - 4 * a * c)

        l = cp.array([ -b + dsc, -b - dsc ]).T / 2

        return l

