import cupy as cp

class Vector:
    def __init__(self, v = [ 0, 0, 1, 0 ]):
        assert v[3] == 0, f"Vector initialized with 3rd component non-zero {v}"
        self._v = cp.asarray(v, dtype=float)

    @property
    def x(self):
        return float(self._v[0])

    @property
    def y(self):
        return float(self._v[1])

    @property
    def z(self):
        return float(self._v[2])

    @x.setter
    def x(self, x):
        self._v[0] = float(x)
    
    @y.setter
    def y(self, z):
        self._v[1] = float(y)
    
    @z.setter
    def z(self, z):
        self._v[2] = float(z)

        
    @classmethod
    def from_xyz(cls, x, y, z):        
        return cls([x, y, z, 0])

    @property
    def norm(self):
        return cp.linalg.norm(self._v)
    
    @property
    def normalized(self):
        return Vector(self._v / self.norm)

    @property
    def v3(self):
        return self._v[:3]

    @property
    def v(self):
        return self._v

    @property
    def cross_product_matrix(self):
        return cp.array([
            [ 0, -self.v[2], self.v[1] ],
            [ self.v[2], 0, -self.v[0] ],
            [ -self.v[1], self.v[0], 0 ]
        ])

    def outer(self, other):
        return cp.outer(self.v, other.v)
    
    def cross(self, other):
        assert isinstance(other, Vector)

        vp = cp.cross(self._v[:3], other._v[:3])

        return Vector(cp.hstack((vp, [0])))
    
    def isclose(self, v2, **kwargs):
        return cp.all(cp.isclose(self._v, v2._v, **kwargs))
    
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

    def __eq__(self, other):
        return cp.all(self._v == other._v)
    
    def __matmul__(self, other):
        if isinstance(other, Vector):
            return self._v @ other._v

        raise RuntimeError("improper matrix multiply")
    
    def __repr__(self):
        return f"<V:({self._v[0]}, {self._v[1]}, {self._v[2]})>"

Vector.X = Vector.from_xyz(1,0,0)
Vector.Y = Vector.from_xyz(0,1,0)
Vector.Z = Vector.from_xyz(0,0,1)
