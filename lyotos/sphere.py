import numpy as np

from lyotos.geometry import Vector
from .surface import Surface
from .flat_surface import FlatSurface
from .ray import NoHit

class Sphere:
    @classmethod
    def chord_at_t(cls, R, t):
        return np.sqrt(t * (2 * R - t))

    @classmethod
    def t_for_chord(cls, R, aper):
        return R - np.sqrt(R**2 - aper**2)

    @classmethod
    def intersect(cls, d, R1, R2):
        # p1 == 0, 0, R1
        # p2 == 0, 0, t + R2
        z = (d**2 - R2**2 + R1**2)/(2*d)

        return z

    @classmethod
    def line_intersect(cls, p, d, R):
        xp, yp, zp, _ = p
        dx, dy, dz, _ = d

        a = 1
        b = 2 * (d.v3 @ p.v3)
        c = p @ p - R**2

        if 4 * c > b**2:            
            return np.array([])
        elif 4 * c == b**2:
            return np.array([ -b / 2 ])
        
        dsc = np.sqrt(b**2 - 4 * c)
        
        return np.array([ -b - dsc, -b + dsc ]) / 2


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
        sphi = np.sin(self._phi)
        
        return np.array([ sphi * np.cos(self.theta), sphi * np.sin(self.theta), np.cos(self._phi) ])

    @property
    def projected(self):
        if self.phi == 0:
            return np.array([0, 0])
        
        return np.array([ np.cos(self.theta), np.sin(self.theta) ]) / np.tan((np.pi-self.phi)/2)
    
    def angle(self, other):
        return np.arccos(self.cartesian @ other.cartesian)

        
        
class SphericalSurface(Surface):
    surf_name="spherical"

    #
    # Represents a sphere with radius of curvature R and center at R along the Z axis
    #
    # x^2 + y^2 + (z-R)^2 == R^2
    #
    @classmethod
    def create(cls, R, **kwargs):
        if R == 0:
            print("Substituting flat surface")
            return FlatSurface(**kwargs)

        return super().create(R=R, **kwargs)
            
    def __init__(self, cs, R, offset_sag=False, max_sag=1e10, **kwargs):
        self._R = R
        self._max_sag = max_sag
        self._min_sag = -1e10
        
        super().__init__(cs=cs, **kwargs)

        if offset_sag:
            print(f"ap: {self.aperture}")
            ra = self.aperture.maxr
            z = self.sag(ra, 0)

            self._cs = CoordinateSystem(cs, CSM.tZ(-z))

    @property
    def R(self):
        return self._R

    @property
    def max_sag(self):
        ra = self.aperture.maxr
        return np.clip(self.sag(self.aperture.maxr, 0), self._min_sag, self._max_sag)
    
        
    def sag(self, x, y):
        r2 = np.clip(x**2 + y**2, 0, self._R**2)

        return np.clip(self._R - np.sign(self._R) * np.sqrt(self._R**2 - r2), self._min_sag, self._max_sag)

    def normal(self, x, y):
        #n = super().normal(x, y)

        r2 = x**2 + y**2 
        
        if r2 > self.R**2:
            return Vector([0, 0, 1, 0])

        z = np.sign(self.R) * np.sqrt(self.R**2 - r2)
        
        # the vector to the center of the sphere is (-x, -y, z)        
        n2 = Vector([-x, -y, z, 0])/self.R
        
        #assert n.isclose(n2), f"R: {self.R} x: {x} y: {y} z: {z} n: {n} n2: {n2}"

        return n2

    def do_intersect(self, ray):        
        xp, yp, zp, _ = ray.pos
        dx, dy, dz, _ = ray.d

        l = Sphere.line_intersect((ray.pos - Vector.from_xyz(0, 0, self.R)),
                                   ray.d,
                                   self.R)

        l = l[l >= 0]

        if len(l) == 0:
            raise NoHit()

        z = zp + l * dz

        if self.R > 0:
            l = l[(z >= 0) & (z <= self.R)]
        else:
            l = l[(z >= self.R) & (z <= 0)]

        if len(l) == 0:
            raise NoHit()
            
        assert len(l) == 1, f"Multiple intersections: {l}"

        l = l[0]

        p = ray.pos + l * ray.d

        n = self.normal(p.x, p.y)
        
        return l, p, n
        
        
    
if __name__ == "__main__":
    v = Sphere.chord_at_t(1, 0.1)

    assert v**2 + 0.9**2 == 1
    
