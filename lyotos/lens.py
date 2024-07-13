import numpy as np

from .sphere import Sphere
from .shape import thin_radii, thick_radii

def max_aperture(r1, r2, t):
    if r1 == 0:
        return np.abs(2*r2)
    if r2 == 0:
        return np.abs(2*r1)
        
    x2 = Sphere.intersect(t+r2-r1, r1, r2)**2

    if x2 > r1**2:
        return 2 * np.min(np.abs([r1, r2]))

    rp = np.sqrt(r1**2 - x2)

    return 2*rp

    
    zz = 4 * dist**2 * r1**2 - (dist**2-r2**2+r1**2)**2
    
    print(f"zz: {zz}")
        
    if zz < 0:
        return 2*np.min(np.abs([ r1, r2 ]))

    return np.abs(1/dist * np.sqrt(zz))
    

class Lens:
    def __init__(self, R1, R2, t, n, aper):
        self._R1 = R1
        self._R2 = R2
        self._t = t
        self._n = n
        self._aper = aper

    @property
    def n(self):
        return self._n

    @property
    def t(self):
        return self._t

    @property
    def R1(self):
        return self._R1

    @property
    def R2(self):
        return self._R2
    
    @property
    def P(self):
        return (self.n - 1) * (1/self.R1 - 1/self.R2 + (self.n-1)*self.t / (self.n * self.R1 * self.R2))

    @property
    def h1(self):
        return -self.f * (self.n - 1) * self.t / (self.R2 * self.n)

    @property
    def h2(self):
        return -self.f * (self.n - 1) * self.t / (self.R1 * self.n)

    
    @property
    def aper(self):
        return self._aper

    
    @property
    def f(self):
        return 1/self.P

    @property
    def q(self):
        return (self.R1 + self.R2)/(self.R1 - self.R2)

    @property
    def max_aperture(self):
        return max_aperture(self.R1, self.R2, self.t)
                    
    
    @property
    def petzval_sum(self):
        return (self.n - 1) / (self.R1 * self.n) + (1-self.n)/(self.R2*self.n)
    
    @classmethod
    def with_shape(cls, f, q, n, aper=None, t=1):
        # Start with thin lens approx, and solve for r1
        print(f"Creating lens with focal length {f}")

        r1, r2 = thick_radii(f, q, n, t)
        
        print(f"Thick lens: {r1} {r2}")
        
        return Lens(r1, r2, t, n, aper)
