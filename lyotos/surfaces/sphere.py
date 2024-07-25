import cupy as cp

from lyotos.util import darray, take_lowest_l_p_2
from lyotos.rays import MISS

from lyotos import geometry as geo

from .surface import Surface

class Sphere(Surface):
    """
    Represents a spherical surface with radius R which passes
    through 0, 0, 0 in the coordinate system provided
    """
    def __init__(self, cs, interaction, R):
        super().__init__(cs, interaction)
        self._R = R
        
    @property
    def R(self):
        return self._R
        
    def do_intersect(self, bundle):        
        l = geo.Sphere.intersect(self.R, bundle.positions, bundle.directions)

        # Avoid repeat intersection
        l[cp.isnan(l)] = MISS
        l[l < 1e-7] = MISS

        l = cp.min(l, axis=1)

        p = bundle.pts_at(l)
                        
        n = -p

        n = cp.einsum("ij,i->ij", n, 1.0/cp.linalg.norm(n, axis=1))

        return l, p, n
        
    def render(self, renderer):
        renderer.add_spherical_cap(self.cs, self.R, self.aperture/2)
