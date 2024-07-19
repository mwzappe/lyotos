import cupy as cp

from lyotos.geometry import Cylinder

from .surface import Surface

class CylindricalSurface(Surface):
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
        raise RuntimeError("Define everything")
