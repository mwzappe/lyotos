from lyotos.util import xp

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
        sphi = xp.sin(self._phi)
        
        return darray([ sphi * xp.cos(self.theta), sphi * xp.sin(self.theta), xp.cos(self._phi) ])

    @property
    def projected(self):
        if self.phi == 0:
            return darray([0, 0])
        
        return darray([ xp.cos(self.theta), xp.sin(self.theta) ]) / xp.tan((xp.pi-self.phi)/2)
    
    def angle(self, other):
        return xp.arccos(self.cartesian @ other.cartesian)

    @classmethod
    def random_points(cls, N, theta=xp.pi):
        u, v = R2(xp.arange(N))
        
        if theta == xp.pi:
            return xp.vstack((xp.arccos(2 * u - 1), 2 * xp.pi * v)).transpose()

        rmax = 2 * xp.cos((xp.pi-theta)/2)
        
        phi = xp.pi - 2 * xp.arccos(rmax * u/2)
        
        return xp.vstack((phi, 2 * xp.pi * v)).transpose()
    
