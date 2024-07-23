import cupy as cp

from lyotos.util import darray
from lyotos.geometry import Cylinder
from lyotos.rays import MISS

from .surface import Surface

class CylinderSurface(Surface):
    def __init__(self, cs, R, h, first_surface_only=True):
        super().__init__(cs)
        self._R = R
        self._h = h

    @property
    def R(self):
        return self._R

    @property
    def h(self):
        return self._h

    def do_intersect(self, bundle):
        p = bundle.positions
        d = bundle.directions
        
        a = d[:,0]**2 + d[:,1]**2
        b = 2 * (d[:,0] * p[:,0] +
                 d[:,1] * p[:,1])
        c = p[:,0]**2 + p[:,1]**2 - self.R**2
        
        dsc = cp.sqrt(b ** 2 - 4 * a * c)

        l = darray([ -b + dsc, -b - dsc ]).T

        l[cp.isnan(l)] = MISS
        l[l < 1e-7] = MISS

        l = cp.min(l, axis=1)
        
        p = bundle.pts_at(l)

        n = darray([ -p[:,0], -p[:,1], cp.zeros(len(p)), cp.zeros(len(p)) ]).T

        return l, p, n

    def render(self, renderer):
        renderer.add_cylinder(self.cs, self.R, self.h)

