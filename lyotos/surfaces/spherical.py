import cupy as cp

from lyotos.util import take_lowest_l_p_2
from lyotos.rays import MISS
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
        offset = cp.repeat(cp.array([ [ 0, 0, -self.R, 0 ] ]), len(bundle), axis=0)

        l = Sphere.intersect(self.R, bundle.positions + offset, bundle.directions)

        # Avoid repeat intersection
        l[cp.isnan(l)] = MISS
        l[l < 1e-7] = MISS

        p0 = bundle.pts_at(l[:,0])
        p1 = bundle.pts_at(l[:,1])
        
        if self.R > 0:
            l[(p0[:,2] > self.R),0] = MISS
            l[(p1[:,2] > self.R),1] = MISS
        else:
            l[(p0[:,2] < -self.R),0] = MISS
            l[(p1[:,2] < -self.R),1] = MISS


        l, p = take_lowest_l_p_2(l, p0, p1)
        
        n = p - cp.repeat(cp.array([ [ 0, 0, self.R, 1] ]), len(bundle), axis=0)

        n = cp.einsum("ij,i->ij", n, 1.0/cp.linalg.norm(n, axis=1))

        if self.R < 0:
            n = -n

        return l, p, n
        
