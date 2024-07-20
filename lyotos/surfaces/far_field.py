import cupy as cp

from lyotos.geometry import CSM, GCS, Position, Sphere
from lyotos.physics import Absorber

from .surface import Surface

class FarField(Surface):
    def __init__(self, cs=GCS, center=Position.CENTER, radius=1e4):
        self._center = center
        self._radius = radius
        
        super().__init__(cs=cs.xform(CSM.translate(center)), interaction_cls=Absorber)

    @property
    def center(self):
        return self._center
    
    @center.setter
    def center(self, new_center):
        self._center = new_center
        self._cs = self._syscs.xform(CSM.translate(new_center))

    @property
    def R(self):
        return self._radius
        
    @property
    def absorber(self):
        return True
        
    def do_intersect(self, bundle):
        l = Sphere.intersect(self.R, bundle.positions, bundle.directions)

        FLOATMAX=1e11
        
        l[l < 0] = FLOATMAX
        
        li = cp.argmin(l, 1)

        l = cp.take_along_axis(l, li[:,cp.newaxis], 1).flatten()
        
        print(l)

        p = bundle.pts_at(l)
        n = -p
        
        return l, p, n

