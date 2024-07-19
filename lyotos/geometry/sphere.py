import cupy as cp

class Sphere:
    @classmethod
    def intersect(cls, R, bundle):
        p = bundle[:,0:3]
        d = bundle[:,4:7]
        
        dpd = cp.einsum("ij,ij->i",p,d)
        dpp = cp.einsum("ij,ij->i",p,p)

        a = 1
        b = 2 * dpd
        c = dpp - R**2

        dsc = cp.sqrt(b**2 - 4 * a * c)

        l = cp.array([ -b + dsc, -b - dsc ]).T / 2

        return l

