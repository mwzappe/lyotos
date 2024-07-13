import numpy as np

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
        M[1][2] = np.sin(theta)
        M[2][1] = -np.sin(theta)
        M[2][2] = np.cos(theta)

        return CSM(M)

    # Create a new CS translated by dz
    @classmethod
    def tX(cls, dx):
        return CSM(np.array([
            [ 1, 0, 0, -dx ],
            [ 0, 1, 0, 0 ],
            [ 0, 0, 1, 0 ],
            [ 0, 0, 0, 1 ]
        ]))

    
    # Create a new CS translated by dz
    @classmethod
    def tY(cls, dy):
        return CSM(np.array([
            [ 1, 0, 0, 0 ],
            [ 0, 1, 0, -dy ],
            [ 0, 0, 1, 0 ],
            [ 0, 0, 0, 1 ]
        ]))

    
    # Create a new CS translated by dz
    @classmethod
    def tZ(cls, dz):
        return CSM(np.array([
            [ 1, 0, 0, 0 ],
            [ 0, 1, 0, 0 ],
            [ 0, 0, 1, -dz ],
            [ 0, 0, 0, 1 ]
        ]))

    # Create a new CS translated by dz
    @classmethod
    def translate(cls, pos):
        return CSM(np.array([
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
