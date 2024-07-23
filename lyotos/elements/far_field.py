import cupy as cp

from lyotos.geometry import CSM, GCS, Position, Sphere
from lyotos.rays import MISS
from lyotos.physics import Absorber
from lyotos.surfaces import SphericalSurface

from .element import Element

class FarField(Element):
    def __init__(self, cs=GCS, center=Position.CENTER, radius=1e4):
        super().__init__(cs=cs.xform(CSM.translate(center)))

        self._surface = SphericalSurface(self.cs, radius, hemisphere=False)
        
        self._center = center        

    @property
    def center(self):
        return self._center
    
    @property
    def R(self):
        return self._surface.R
        
    @property
    def absorber(self):
        return True

    def do_trace(self, bundle):
        pass
        
    def do_intersect(self, bundle):
        l = Sphere.intersect(self.R, bundle.positions, bundle.directions)

        assert not cp.any(cp.isnan(l))

        # Avoid repeat intersection
        l[l < 1e-7] = MISS

        l = cp.min(l, axis=1)

        p = bundle.pts_at(l)

        n = -p 

        n = cp.einsum("ij,i->ij", n, 1.0/cp.linalg.norm(n, axis=1))

        n[:,3] = 0

        return l, p, n
        
