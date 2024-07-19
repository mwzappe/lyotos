import numpy as np

from .coordinate_system import CoordinateSystem, CSM, Vector
from .surface import Surface
from .sphere import Sphere

class FarField(Surface):
    def __init__(self, cs, center, radius):
        self._center = center
        self._radius = radius
        self._syscs = cs
        
        super().__init__(cs=cs.xform(CSM.translate(center)))

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
        
    def intersect(self, ray):
        l = Sphere.line_intersect(ray.pos, ray.d, self._radius)
        
        l = l[l > 1e-7]
        
        assert len(l) == 1
        
        l = l[0]

        p = ray.at(l)
        
        return l, p, None

