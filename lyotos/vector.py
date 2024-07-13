import numpy as np

class Vector:
    def __init__(self, v = [ 0, 0, 1, 0 ]):
        assert v[3] == 0, f"Vector initialized with 3rd component non-zero {v}"
        self._v = np.asarray(v)

    @classmethod
    def from_xyz(cls, x, y, z):        
        return cls([x, y, z, 0])

    @property
    def norm(self):
        return np.linalg.norm(self._v)
    
    @property
    def normalized(self):
        return Vector(self._v / self.norm)

    def cross(self, other):
        assert isinstance(other, Vector)

        vp = np.cross(self._v[:3], other._v[:3])

        return Vector(np.hstack((vp, [0])))
    
    def isclose(self, v2, **kwargs):
        return np.all(np.isclose(self._v, v2._v, **kwargs))
    
    def __getitem__(self, n):
        return self._v[n]

    def __add__(self, d):
        return Vector(self._v + d._v)

    def __sub__(self, d):
        return Vector(self._v - d._v)

    def __neg__(self):
        return Vector(-self._v)
    
    def __mul__(self, d):
        return Vector(self._v * d)

    def __rmul__(self, d):
        return Vector(d * self._v)

    def __truediv__(self, d):
        return Vector(self._v / d)
    
    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self._v @ other._v

        return Vector(self._v @ other._M)
    
    def __repr__(self):
        return f"<V:({self._v[0]}, {self._v[1]}, {self._v[2]})>"
