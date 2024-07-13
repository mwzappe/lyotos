import numpy as np

from .vector import Vector

class Position:
    def __init__(self, v=[ 0, 0, 0, 1]):
        assert v[3] == 1, f"Position v[3] != 1: {v}"
        self._v = np.asarray(v) 

    @property
    def x(self):
        return self._v[0]

    @property
    def y(self):
        return self._v[1]

    @property
    def z(self):
        return self._v[2]

    @z.setter
    def z(self, z):
        self._v[2] = z
    
    @classmethod
    def from_xyz(cls, x, y, z):
        return cls([x, y, z, 1])

    def __getitem__(self, n):
        return self._v[n]
    
    def __add__(self, other):
        if isinstance(other, Vector):
            return Position(self._v + other._v)
        if isinstance(other, Position):
            return Position.from_xyz(self.x+other.x, self.y+other.y, self.z+other.z)
        
        return Position(self._v + other)

    def __sub__(self, other):
        if isinstance(other, Position):
            return Vector(self._v - other._v)
    
    def __truediv__(self, other):
        return Position(self._v / [other, other, other, 1 ])

    def __matmul__(self, other):
        if isinstance(other, Position):
            return self._v @ other._v

        return Position(self._v @ other._M)

    
    def __repr__(self):
        return f"<P:({self._v[0]}, {self._v[1]}, {self._v[2]})>"
    
    
