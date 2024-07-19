import matplotlib.pyplot as plt

import numpy as np

from lyotos.geometry import CoordinateSystem, CSM, Vector, Position
from .ray import Ray, NoHit
from .aperture import CircularAperture

NMANT = np.finfo(float).nmant

MMANT = 2**NMANT-1

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
    
    def __init__(self, cs, name="", aperture=None, color=None, display=True, absorber=False):
        self._name = name
        self._cs = cs
        self._aper = 100
        self._zrange = None
        self._display = display

        if isinstance(aperture, (float, int)):
            aperture = CircularAperture(cs, aperture)
        
        self._aperture = aperture
        self._color = color
        self._absorber = absorber

    @property
    def display(self):
        return self._display
        
    @property
    def cs(self):
        return self._cs

    @property
    def name(self):
        return self._name

    @property
    def far_field(self):
        return False
    
    @property
    def aperture(self):
        return self._aperture

    @property
    def thickness(self):
        return self._thickness

    @property
    def absorber(self):
        return self._absorber
   
    def intersect(self, ray):
        return self.do_intersect(ray.toCS(self.cs))

    def sag(self, x, y):
        ray = Ray(self.cs, Position.from_xyz(x, y, -MMANT), Vector.Z)

        _, p, _ = self.intersect(ray)
        
        return p.z

    
    def __repr__(self):
        return f"Surface {self._name} {self.__class__}"


        
    
