import matplotlib.pyplot as plt

import numpy as np

from .coordinate_system import CoordinateSystem, CSM, Vector
from .ray import NoHit
from .aperture import CircularAperture

surf_classes = { }

def create_surface(surf_type, cs, **kwargs):
    return surf_classes[surf_type].create(cs=cs, **kwargs)

class SurfaceMetaclass(type):
    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        surf_classes[x.surf_name] = x
        return x

class Surface(metaclass=SurfaceMetaclass):
    surf_name="base"

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)
    
    def __init__(self, cs, name="", aperture=None, color=None, far_field=False, absorber=False):
        self._name = name
        self._cs = cs
        self._aper = 100
        self._zrange = None

        if isinstance(aperture, (float, int)):
            aperture = CircularAperture(cs, aperture)
        
        self._aperture = aperture
        self._color = color
        self._far_field = far_field
        self._absorber = absorber
        
    @property
    def cs(self):
        return self._cs

    @property
    def name(self):
        return self._name

    @property
    def far_field(self):
        return self._far_field
    
    @property
    def aperture(self):
        return self._aperture

    @property
    def thickness(self):
        return self._thickness

    @property
    def absorber(self):
        return self._absorber

    
    def sag(self, x, y):
        return 0 * x + 0 * y

    def normal(self, x, y):
        delta = 0.004

        da = np.array([ -2 * delta, -delta, 0, delta, 2 * delta ])

        sagx = self.sag(x + da, y)
        sagy = self.sag(x, y + da)
        
        dzdx = (-sagx[0] + 8 * sagx[1] - 8 * sagx[3] + sagx[4]) / 12 / delta
        dzdy = (-sagy[0] + 8 * sagy[1] - 8 * sagy[3] + sagy[4]) / 12 / delta

        norm = np.array([dzdx, dzdy, 1, 0 ])
        norm /= np.linalg.norm(norm)

        return Vector(norm)
    
    
    def intersect(self, ray):
        delta = 0.004
        
        # p + l * dcos == (x, y, f(x, y))

        # Solve 
        # 0 == f(p[0] + l * dcos[0], p[1] + l * dcos[1]) - p[2] - l * dcos[2]

        l = ray.l_at_z0 # -p[2]/dcos[2]
        
        for i in range(10):
            ls = l + np.array([ -2 * delta, -delta, 0, delta, 2 * delta ])

            ps = ray.at(ls)

            v = np.array([ self.sag(p.x, p.y) - p.z for p in ps ])

            if np.abs(v[2]) < 1e-7:
                break
            
            dv = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / 12 / delta

            l += -v[2] / dv

        return l

    def __repr__(self):
        return f"Surface {self._name} {self.__class__}"
        

        

class FlatSurface(Surface):
    surf_name="flat"

    def intersect(self, ray):
        return -ray.pos[2]/ray.d[2]


        
    
