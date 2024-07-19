import cupy as cp

from lyotos.geometry import CoordinateSystem, GCS, CSM, Position, Vector
from .ray import Ray, NoHit

class RayBundle:
    def __init__(self, cs, bundle):
        self._cs = cs
        self._bundle = bundle

    def toCS(self, newcs):
        M = newcs.fromGCS @ self.cs.toGCS

        pp = cp.einsum("jk,ik->ij", M._M, self.bundle[:,0:4])
        dp = cp.einsum("jk,ik->ij", M._M, self.bundle[:,4:8])

        return RayBundle(newcs, cp.hstack((pp, dp)))

    def pts_at(self, ls):
        return self.positions + cp.einsum("i,ij->ij", ls, self.directions)
        
        
    @property
    def cs(self):
        return self._cs

    @property
    def bundle(self):
        return self._bundle

    def __len__(self):
        return self.bundle.shape[0]
    
    @property
    def positions(self):
        return self._bundle[:,0:4]
    
    @property
    def directions(self):
        return self._bundle[:,4:8]

    
    @classmethod
    def from_rays(cls, rays):
        p = cp.array([ r.pos.v for r in rays ])
        d = cp.array([ r.d.v for r in rays ])

        m = cp.array([ r.cs.toGCS.M for r in rays ])

        pgcs = cp.einsum("ijk,ik->ij", m, p)
        dgcs = cp.einsum("ijk,ik->ij", m, d)
        
        return cls(GCS, cp.hstack((pgcs, dgcs)))
