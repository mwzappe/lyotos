import cupy as cp

from lyotos.util import MISS, matrix_mult_nvecs
from lyotos.geometry import CoordinateSystem, GCS, CSM, Position, Vector
from .ray import Ray, NoHit

class RayBundle:
    _next_id = 1
    bundles = []

    def __init__(self, cs, positions, directions, ids=None, parents=None):
        assert positions.shape == directions.shape, f"Position ({position.shape}) and direction ({directions.shape}) shapes do not match"
        assert positions.shape[1] == 4, f"Improper number of coordinates for positions and directions"

        self.bundles.append(self)
        
        self._cs = cs
        self._positions = positions
        self._directions = directions

        if ids is None:
            ids = cp.arange(self._next_id, self._next_id + positions.shape[0])
            self._next_id += positions.shape[0]

        self._ids = ids

        if parents is None:
            parents = cp.zeros(positions.shape[0], dtype=int)

        self._parents = parents
        
    def toCS(self, newcs):
        M = newcs.fromGCS @ self.cs.toGCS

        
        pp = matrix_mult_nvecs(M._M, self.positions)
        dp = matrix_mult_nvecs(M._M, self.directions)

        return RayBundle(newcs, pp, dp)

    def pts_at(self, ls):
        return self.positions + cp.einsum("i,ij->ij", ls, self.directions)

    @property
    def ids(self):
        return self._ids

    @property
    def parents(self):
        return self._parents
    
    @property
    def cs(self):
        return self._cs

    def __len__(self):
        return self.positions.shape[0]
    
    @property
    def positions(self):
        return self._positions
    
    @property
    def directions(self):
        return self._directions

    
    @classmethod
    def from_rays(cls, rays):
        p = cp.array([ r.pos.v for r in rays ])
        d = cp.array([ r.d.v for r in rays ])

        m = cp.array([ r.cs.toGCS.M for r in rays ])

        pgcs = cp.einsum("ijk,ik->ij", m, p)
        dgcs = cp.einsum("ijk,ik->ij", m, d)
        
        return cls(GCS, cp.hstack((pgcs, dgcs)))
