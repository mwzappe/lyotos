import numpy as np

from .surface import Surface, FlatSurface
from .vector import Vector
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

    def intersect(self, ray):
        
        xp, yp, zp, _ = ray.pos
        dx, dy, dz, _ = ray.d

        # |dcos| == 1, so dx**2 + dy**2 + dz**2 == 1

        def plane_hit():
            if dz == 0:
                raise NoHit()

            l = (-zp+self.R)/dz
            
            if l < 0:
                raise NoHit()
            
            if not self.aperture.inside(xp + l * dx, yp + l * dy):
                raise NoHit()
            
            
            return l
        
        b = 2 * (dx * xp + dy * yp + dz * (zp - self.R))
        c = xp**2 + yp**2 - 2 * self.R * zp + zp**2

        if 4 * c > b**2:            
            # hit outside of spherical portion -- assume flat
            return plane_hit()
        
        dsc = np.sqrt(b**2 - 4 * c)
        
        l = np.array([ -b - dsc, -b + dsc ]) / 2

        l = l[l>=0]

        if len(l) == 0:
            raise NoHit()
        
        z = zp + l * dz

        if self.R > 0:
            l = l[(z >= 0) & (z <= self.R)]
        else:
            l = l[(z >= self.R) & (z <= 0)]

        if len(l) == 0:
            raise NoHit()

        l = np.min(l)
        
        return l
        
    
if __name__ == "__main__":
    v = Sphere.chord_at_t(1, 0.1)

    assert v**2 + 0.9**2 == 1
    
