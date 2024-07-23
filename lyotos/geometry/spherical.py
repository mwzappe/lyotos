import cupy as cp

from lyotos.util import darray

from .quasirand import R2

class SphPos:
    def __init__(self, phi, theta):
        self._phi = phi
        self._theta = theta

    @property
    def phi(self):
        return self._phi

    @property
    def theta(self):
        return self._theta
        
    @property
    def cartesian(self):
        sphi = cp.sin(self._phi)
        
        return darray([ sphi * cp.cos(self.theta), sphi * cp.sin(self.theta), cp.cos(self._phi) ])

    @property
    def projected(self):
        if self.phi == 0:
            return darray([0, 0])
        
        return darray([ cp.cos(self.theta), cp.sin(self.theta) ]) / cp.tan((cp.pi-self.phi)/2)
    
    def angle(self, other):
        return cp.arccos(self.cartesian @ other.cartesian)

    @classmethod
    def random_points(cls, N, theta=cp.pi):
        u, v = R2(cp.arange(N))
        
        if theta == cp.pi:
            return cp.vstack((cp.arccos(2 * u - 1), 2 * cp.pi * v)).transpose()

        rmax = 2 * cp.cos((cp.pi-theta)/2)
        
        phi = cp.pi - 2 * cp.arccos(rmax * u/2)
        
        return cp.vstack((phi, 2 * cp.pi * v)).transpose()
    
