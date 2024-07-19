import numpy as np
from scipy.spatial.transform import Rotation

from vtkmodules.vtkCommonMath import vtkMatrix4x4

from .vector import Vector
from .position import Position
    
class CSM:
    def __init__(self, M=np.array([
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
        return CSM(np.linalg.inv(self._M))
        
    @property
    def vtk(self):
        retval = vtkMatrix4x4()

        for i in range(4):
            for j in range(4):
                retval.SetElement(i, j, self._M[i][j])

        return retval

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
        M = np.identity(4)

        M[1][1] = np.cos(theta)
        M[1][2] = -np.sin(theta)
        M[2][1] = np.sin(theta)
        M[2][2] = np.cos(theta)

        return CSM(M)

    @classmethod
    def rotY(cls, theta):
        M = np.identity(4)

        M[0][0] = np.cos(theta)
        M[0][2] = np.sin(theta)
        M[2][0] = -np.sin(theta)
        M[2][2] = np.cos(theta)

        return CSM(M)

    @classmethod
    def rotZ(cls, theta):
        M = np.identity(4)

        M[0][0] = np.cos(theta)
        M[0][1] = -np.sin(theta)
        M[1][0] = np.sin(theta)
        M[1][1] = np.cos(theta)

        return CSM(M)

    @classmethod
    def rot2(cls, phi, theta):
        # Rotate Z axis by phi

        v = Vector.from_xyz(np.sin(phi), 0, np.cos(phi))
        
        return cls.from_axis_angle(v, theta) @ cls.rotY(phi)

    @classmethod
    def from_axis_angle(cls, axis, theta):
        return cls.from_scipy_rotation(Rotation.from_rotvec(axis._v[:3] * theta))

    @classmethod
    def align_z(cls, v):
        if v == -Vector.Z:
            return cls.rotY(np.pi)
        
        retval = cls.from_scipy_rotation(Rotation.align_vectors(v.normalized._v[:3], [0, 0, 1])[0])

        return retval
        
    @classmethod
    def from_scipy_rotation(cls, R):
        m = np.pad(R.as_matrix(), ((0,1),(0,1))) 
        m[3][3] = 1
        return cls(m)

    
        
        
        
        
    
    # Create a new CS translated by dz
    @classmethod
    def tX(cls, dx):
        return cls(np.array([
            [ 1, 0, 0, -dx ],
            [ 0, 1, 0, 0 ],
            [ 0, 0, 1, 0 ],
            [ 0, 0, 0, 1 ]
        ]))

    
    # Create a new CS translated by dz
    @classmethod
    def tY(cls, dy):
        return cls(np.array([
            [ 1, 0, 0, 0 ],
            [ 0, 1, 0, -dy ],
            [ 0, 0, 1, 0 ],
            [ 0, 0, 0, 1 ]
        ]))

    
    # Create a new CS translated by dz
    @classmethod
    def tZ(cls, dz):
        return cls(np.array([
            [ 1, 0, 0, 0 ],
            [ 0, 1, 0, 0 ],
            [ 0, 0, 1, -dz ],
            [ 0, 0, 0, 1 ]
        ]))

    # Create a new CS translated by dz
    @classmethod
    def translate(cls, pos):
        return cls(np.array([
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
