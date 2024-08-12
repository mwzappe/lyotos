from lyotos.util import xp
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
        return CSM(xp.linalg.inv(self._M))

    def batch_mult(self, other):
        assert len(other.shape) == 2, f"Only handles vectors for now (i.e. shape must be Nx4): shape {other.shape}"

        return xp.einsum("jk,ik->ij", self._M, other).reshape((len(other), 4))
    
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
        M = xp.identity(4)

        M[1][1] = xp.cos(theta)
        M[1][2] = -xp.sin(theta)
        M[2][1] = xp.sin(theta)
        M[2][2] = xp.cos(theta)

        return CSM(M)

    @classmethod
    def rotY(cls, theta):
        M = xp.identity(4)

        M[0][0] = xp.cos(theta)
        M[0][2] = xp.sin(theta)
        M[2][0] = -xp.sin(theta)
        M[2][2] = xp.cos(theta)

        return CSM(M)

    @classmethod
    def rotZ(cls, theta):
        M = xp.identity(4)

        M[0][0] = xp.cos(theta)
        M[0][1] = -xp.sin(theta)
        M[1][0] = xp.sin(theta)
        M[1][1] = xp.cos(theta)

        return CSM(M)

    @classmethod
    def rot2(cls, phi, theta):
        # Rotate Z axis by phi

        v = Vector.from_xyz(xp.sin(phi), 0, xp.cos(phi))
        
        return cls.from_axis_angle(v, theta) @ cls.rotY(phi)

    @classmethod
    def from_axis_angle(cls, axis, theta):
        #return cls.from_scipy_rotation(Rotation.from_rotvec(axis._v[:3] * theta))

        ux, uy, uz = axis.normalized.v[:3]
        ct = xp.cos(theta)
        st = xp.sin(theta)

        # ct * I + st * axis.cross_product_matrix + (1 - ct) * outer(u, u)
        
        M = xp.array([
            [ ct + ux**2 * (1 - ct), ux * uy * (1 - ct) - uz * st, ux * uz * (1 - ct) + uy * st, 0 ],
            [ uy * ux * (1 - ct) + uz * st, ct + uy**2 * (1 - ct), uy * uz * (1 - ct) - ux * st, 0 ],
            [ uz * ux * (1 - ct) - uz * st, uy * uz * (1 - ct) + ux * st, ct + uz**2 * (1 - ct), 0 ],
            [ 0, 0, 0, 1 ]
        ])

        return cls(M)
    
    @classmethod
    def align_z(cls, v):
        if v == Vector.Z:
            return cls.I
            
        if v == -Vector.Z:
            return cls.rotY(xp.pi)

        axis = v.cross(Vector.Z).normalized
        
        retval = cls.from_scipy_rotation(Rotation.align_vectors(v.normalized._v[:3], [0, 0, 1])[0])

        return retval
        
    @classmethod
    def from_scipy_rotation(cls, R):
        m = xp.pad(xp.asarray(R.as_matrix()), ((0,1),(0,1))) 
        m[3][3] = 1
        return cls(m)

    
        
        
        
        
    
    # Create a new CS translated by dz
    @classmethod
    def tX(cls, dx):
        M = xp.identity(4)

        M[0][3] = -dx

        return cls(M)

    
    # Create a new CS translated by dz
    @classmethod
    def tY(cls, dy):
        M = xp.identity(4)

        M[1][3] = -dy

        return cls(M)

    
    # Create a new CS translated by dz
    @classmethod
    def tZ(cls, dz):
        M = xp.identity(4)

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
        self._fromGCS = self._M @ parent.fromGCS if parent is not None else self._M
        self._toGCS = self._fromGCS.inv
        
    def newCS(self, p, d = Vector.Z):
        return self.xform(CSM.align_z(d) @ CSM.translate(p))
        
    def xform(self, M):
        return CoordinateSystem(self, M)
        
    @property
    def fromGCS(self):
        return self._fromGCS
    
    @property
    def toGCS(self):
        return self._toGCS

    @property
    def isGCS(self):
        return self._parent is None

    def __repr__(self):
        return f"<<|From GCS: {self.toGCS}, M: {self._M._M}|>>"
    

CSM.I = CSM()
GCS = CoordinateSystem(None, CSM())
