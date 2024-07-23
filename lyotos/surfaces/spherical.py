import cupy as cp

from lyotos.util import darray, take_lowest_l_p_2
from lyotos.rays import MISS
from lyotos.geometry import Sphere

from .surface import Surface

class SphericalSurface(Surface):
    """
    Represents a spherical surface with radius R which passes
    through 0, 0, 0 in the coordinate system provided
    """
    def __init__(self, cs, R, aperture=None, hemisphere=True):
        super().__init__(cs)
        self._R = R

        if aperture is None:
            aperture = cp.abs(R)
        
        self._aperture = aperture
        self._hemisphere = hemisphere

        self._edge_z = self.R - cp.sqrt(self.R**2 - (self.aperture/2)**2)

        
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
    
    def do_intersect(self, bundle):        
        offset = cp.repeat(darray([ [ 0, 0, -self.R, 0 ] ]), len(bundle), axis=0)

        l = Sphere.intersect(self.R, bundle.positions + offset, bundle.directions)

        # Avoid repeat intersection
        l[cp.isnan(l)] = MISS
        l[l < 1e-7] = MISS

        if self.hemisphere:            
            p0 = bundle.pts_at(l[:,0])
            p1 = bundle.pts_at(l[:,1])
            
            if self.R > 0:
                l[(p0[:,2] > self.R),0] = MISS
                l[(p1[:,2] > self.R),1] = MISS
            else:
                l[(p0[:,2] < -self.R),0] = MISS
                l[(p1[:,2] < -self.R),1] = MISS

            l, p = take_lowest_l_p_2(l, p0, p1)
        else:
            l = cp.min(l, axis=1)

            p = bundle.pts_at(l)
            
        l[p[:,0]**2 + p[:,1]**2 > (self.aperture/2)**2] = MISS
            
        offset[:,3] = -1
            
        n = p + offset

        n = cp.einsum("ij,i->ij", n, 1.0/cp.linalg.norm(n, axis=1))

        if self.R < 0:
            n = -n

        return l, p, n
        
    def render(self, renderer):
        renderer.add_spherical_cap(self.cs, self.R, self.aperture/2)
