import cupy as cp

from lyotos.geometry import Sphere

from .surface import Surface

class SphericalSurface(Surface):
    """
    Represents a spherical surface with radius R which passes
    through 0, 0, 0 in the coordinate system provided
    """
    def __init__(self, cs, R):
        super().__init__(cs)
        self._R = R

    @property
    def R(self):
        return self._R

    def do_intersect(self, bundle):
        offset = cp.repeat(cp.array([ [ 0, 0, -self.R, 0, 0, 0, 0, 0 ] ]), len(bundle), axis=0)

        l = Sphere.intersect(self.R, bundle.bundle + offset)

        l[l < 0] = cp.nan

        p0 = bundle.pts_at(l[:,0])
        p1 = bundle.pts_at(l[:,1])
        
        if self.R > 0:
            l[(p0[:,2] > self.R),0] = cp.nan
            l[(p1[:,2] > self.R),1] = cp.nan
        else:
            l[(p0[:,2] < -self.R),0] = cp.nan
            l[(p1[:,2] < -self.R),1] = cp.nan

        FLOATMAX=1e11
            
        l[cp.isnan(l)] = 1e11
            
        li = cp.argmin(l, 1)

        p = cp.einsum("i,ij->ij", (1 - li), p0) + cp.einsum("i,ij->ij", li, p1)
        l = cp.take_along_axis(l, li[:,cp.newaxis], 1).flatten()

        l[l == 1e11] = cp.nan

        n = p - cp.repeat(cp.array([ [ 0, 0, self.R, 1] ]), len(bundle), axis=0)

        n = cp.einsum("ij,i->ij", n, 1.0/cp.linalg.norm(n, axis=1))

        if self.R < 0:
            n = -n

        return l, p, n
        
