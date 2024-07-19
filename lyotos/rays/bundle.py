from numba import cuda
import numpy as np

from lyotos.coordinate_system import CoordinateSystem, GCS, CSM, Position, Vector
from .ray import Ray, NoHit

TPB=16

@cuda.jit
def bulk_multics_conv(p, m, r):
    i, j = cuda.grid(2)

    if i < r.shape[0] and j < r.shape[1]:
        tmp = 0.
        for k in range(p.shape[1]):
            tmp += p[i, k] * m[i, j, k]
        r[i, j] = tmp

#matmul[64, 64](np.zeros((64, 64)), np.zeros((64, 64)), np.zeros((64, 64)))

#print("Matmul worked")

class RayBundle:
    def __init__(self, cs, bundle):
        self._cs = cs
        self._bundle = bundle
        
    @classmethod
    def from_rays(cls, rays):
        p = np.array([ r.pos.v for r in rays ])
        d = np.array([ r.d.v for r in rays ])

        m = np.array([ r.cs.toGCS.M for r in rays ])

        pgcs = np.zeros(p.shape, dtype="float32")
        dgcs = np.zeros(p.shape, dtype="float32")
        
        bulk_multics_conv[16,(4, 4)](p, m, pgcs)

        print("Convert dirs")
        bulk_multics_conv[16,(4, 4)](d, m, dgcs)

        
        return cls(GCS, np.hstack((pgcs, dgcs)))
