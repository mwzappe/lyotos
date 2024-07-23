import cupy as cp
from scipy.spatial.transform import Rotation

from lyotos.util import darray

from .vector import Vector
from .position import Position
    
class CSM:
    def __init__(self, M=darray([
            [ 1, 0, 0, 0 ],
            [ 0, 1, 0, 0 ],
            [ 0, 0, 1, 0 ],
            [ 0, 0, 0, 1 ]
        ])):
        self._M = M

    @property
    def M(self):
        return self._M
        
    @property
    def inv(self):
        return CSM(cp.linalg.inv(self._M))
        
    def __matmul__(self, other):
        if isinstance(other, CSM):
            return CSM(self._M @ other._M)
        elif isinstance(other, Vector):
            return Vector(self._M @ other._v)
        elif isinstance(other, Position):
            return Position(self._M @ other._v)
        
        return self._M @ other

    def __repr__(self):
        return f"<|{self._M}|>"

    
    @classmethod
    def rotX(cls, theta):
        M = cp.identity(4)

        M[1][1] = cp.cos(theta)
        M[1][2] = -cp.sin(theta)
        M[2][1] = cp.sin(theta)
        M[2][2] = cp.cos(theta)

        return CSM(M)

    @classmethod
    def rotY(cls, theta):
        M = cp.identity(4)

        M[0][0] = cp.cos(theta)
        M[0][2] = cp.sin(theta)
        M[2][0] = -cp.sin(theta)
        M[2][2] = cp.cos(theta)

        return CSM(M)

    @classmethod
    def rotZ(cls, theta):
        M = cp.identity(4)

        M[0][0] = cp.cos(theta)
        M[0][1] = -cp.sin(theta)
        M[1][0] = cp.sin(theta)
        M[1][1] = cp.cos(theta)

        return CSM(M)

    @classmethod
    def rot2(cls, phi, theta):
        # Rotate Z axis by phi

        v = Vector.from_xyz(cp.sin(phi), 0, cp.cos(phi))
        
        return cls.from_axis_angle(v, theta) @ cls.rotY(phi)

    @classmethod
    def from_axis_angle(cls, axis, theta):
        return cls.from_scipy_rotation(Rotation.from_rotvec(axis._v[:3] * theta))

    @classmethod
    def align_z(cls, v):
        if v == -Vector.Z:
            return cls.rotY(cp.pi)
        
        retval = cls.from_scipy_rotation(Rotation.align_vectors(v.normalized._v[:3], [0, 0, 1])[0])

        return retval
        
    @classmethod
    def from_scipy_rotation(cls, R):
        m = cp.pad(cp.asarray(R.as_matrix()), ((0,1),(0,1))) 
        m[3][3] = 1
        return cls(m)

    
        
        
        
        
    
    # Create a new CS translated by dz
    @classmethod
    def tX(cls, dx):
        M = cp.identity(4)

        M[0][3] = -dx

        return cls(M)

    
    # Create a new CS translated by dz
    @classmethod
    def tY(cls, dy):
        M = cp.identity(4)

        M[1][3] = -dy

        return cls(M)

    
    # Create a new CS translated by dz
    @classmethod
    def tZ(cls, dz):
        M = cp.identity(4)

        M[2][3] = -dz

        return cls(M)

    # Create a new CS translated by dz
    @classmethod
    def translate(cls, pos):
        return cls(darray([
            [ 1, 0, 0, -pos.x ],
            [ 0, 1, 0, -pos.y ],
            [ 0, 0, 1, -pos.z ],
            [ 0, 0, 0, 1 ]
        ]))

        

class CoordinateSystem:
    def __init__(self, parent, M=CSM()):
        # M goes from parent to child
        
        self._parent = parent
        self._M = M

    def newCS(self, p, d = Vector.Z):
        return self.xform(CSM.align_z(d) @ CSM.translate(p))
        
    def xform(self, M):
        return CoordinateSystem(self, M)
        
    @property
    def fromGCS(self):
        if self._parent is not None:
            return self._M @ self._parent.fromGCS

        return self._M
    
    @property
    def toGCS(self):
        return self.fromGCS.inv

    @property
    def isGCS(self):
        return self._parent is None

    def __repr__(self):
        return f"<<|{self._M._M}|>>"
    
        
GCS = CoordinateSystem(None, CSM())
