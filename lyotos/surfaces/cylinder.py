from lyotos.util import xp

from lyotos.util import darray
from lyotos.geometry import Cylinder
from lyotos.rays import MISS

from .surface import Surface

class CylinderSurface(Surface):
    def __init__(self, cs, interaction, R, h):
        super().__init__(cs, interaction)
        self._R = R
        self._h = h

    @property
    def R(self):
        return self._R

    @property
    def h(self):
        return self._h

    def do_intersect(self, bundle, l, p, n):
        p = bundle.positions
        d = bundle.directions

        ls = bundle.get_scratch(2)
        a = bundle.get_scratch()
        b = bundle.get_scratch()
        c = bundle.get_scratch()
        
        a[:] = d[:,0]**2 + d[:,1]**2
        b[:] = 2 * (d[:,0] * p[:,0] +
                 d[:,1] * p[:,1])
        c[:] = p[:,0]**2 + p[:,1]**2 - self.R**2
        
        dsc = xp.sqrt(b ** 2 - 4 * a * c)

        ls[:] = darray([ -b + dsc, -b - dsc ]).T / 2 / a[:,xp.newaxis]

        ls[xp.isnan(l)] = MISS
        ls[l < 1e-7] = MISS

        l[:] = xp.min(ls, axis=1).reshape(l.shape)
        
        bundle.pts_at(p, l)

        l[p[:,2] < 0] = MISS
        l[p[:,2] > self.h] = MISS

        n[:,0] = -p[:,0]
        n[:,1] = -p[:,1]
        n[:,2] = xp.zeros(len(p))
        n[:,3] = xp.zeros(len(p))

    def render(self, renderer):
        renderer.add_cylinder(self.cs, self.R, self.h)

