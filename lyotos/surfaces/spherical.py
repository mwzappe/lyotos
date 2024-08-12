from lyotos.util import xp

from lyotos.util import darray, take_lowest_l_p_2
from lyotos.rays import MISS
from lyotos.geometry import Sphere

from .surface import Surface

class SphericalSurface(Surface):
    """
    Represents a spherical surface with radius R which passes
    through 0, 0, 0 in the coordinate system provided
    """
    def __init__(self, cs, interaction, R, aperture=None):
        super().__init__(cs, interaction)
        self._R = R

        if aperture is None:
            aperture = xp.abs(R)
        
        self._aperture = aperture

        self._apsq = aperture**2 / 4

        self._edge_z = xp.sign(self.R) * (xp.abs(self.R) - xp.sqrt(self.R**2 - (self.aperture/2)**2))

        
    @property
    def R(self):
        return self._R
    
    @property
    def aperture(self):
        return self._aperture

    @property
    def edge_z(self):
        return self._edge_z
    
    @property
    def hemisphere(self):
        return self._hemisphere
    
    def do_intersect(self, bundle, l, p, n):
        ls = bundle.get_scratch(2)
        p0 = bundle.get_scratch(4)
        p1 = bundle.get_scratch(4)

        Sphere.intersect(bundle, ls, self.R,
                         center=darray([ 0, 0, self.R, 0 ]))

        bundle.pts_at(p0, ls[:,0])
        bundle.pts_at(p1, ls[:,1])
        
        if self.R > 0:
            ls[(p0[:,2] > self.R),0] = MISS
            ls[(p1[:,2] > self.R),1] = MISS
        else:
            ls[(p0[:,2] < self.R),0] = MISS
            ls[(p1[:,2] < self.R),1] = MISS
            
        take_lowest_l_p_2(l, p, ls, p0, p1)
        
        bundle.put_scratch(ls, p0, p1)

        p[l == MISS] = darray([0, 0, 0, 0])
        n[l == MISS] = darray([0, 0, 0, 0])
        l[p[:,0]**2 + p[:,1]**2 > self._apsq] = MISS

        n[l != MISS] = -(p[l != MISS] - darray([ 0, 0, self.R, 1 ])) / self.R
            
    def render(self, renderer):
        renderer.add_spherical_cap(self.cs, self.R, self.aperture/2)
