from lyotos.util import xp

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
        
    def do_intersect(self, bundle, l, p, n):
        ls = bundle.get_scratch(2)        
        
        geo.Sphere.intersect(bundle, ls, self.R)
        
        l[:] = xp.min(ls, axis=1).reshape(l.shape)

        bundle.put_scratch(ls)
        
        bundle.pts_at(p, l)

        n[:,:] = -p

        n[:,:] = xp.einsum("ij,i->ij", n, 1.0/xp.linalg.norm(n, axis=1))
        
    def render(self, renderer):
        renderer.add_spherical_cap(self.cs, self.R, self.aperture/2)
