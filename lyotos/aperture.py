import numpy as np

from lyotos.geometry import Position


class CircularAperture:
    def __init__(self, cs, diameter):
        self._D = diameter
        self._cs = cs
        
    @property
    def width(self):
        return self._D
    
    @property
    def height(self):
        return self._D

    @property
    def maxr(self):
        return self._D/2

    @property
    def D(self):
        return self._D

    @property
    def R(self):
        return self.D/2

    
    def inside(self, x, y):
        return x**2 + y**2 <= self._D**2 / 4
    
    def generate_surface_mesh(self):
        pts = [ ]

        l = self._D / 36

        for r in np.linspace(0, self._D / 2, 36):
            for theta in np.linspace(-np.pi, np.pi, int(2 * np.pi * r / l)):
                pts.append([ r * np.cos(theta), r * np.sin(theta) ])

        return pts

    def boundary(self, npts):
        return [ Position.from_xyz(self.R * np.cos(theta), self.R * np.sin(theta), 0) for theta in np.linspace(0, 2*np.pi, npts) ]
    
    def __repr__(self):
        return f"<Circular Aperture: {self._D}>"
