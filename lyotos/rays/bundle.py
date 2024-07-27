from collections import defaultdict

from lyotos.util import xp

from lyotos.geometry import GCS

from .base import MISS    
from .bundle_base import _BundleBase
from .bundle_hits import BundleHits

class Bundle(_BundleBase):
    _next_id = 1
    _next_bundle_id = 0
    bundles = []

    def __init__(self, positions, directions, cs = GCS, ids=None, parents=None):
        super().__init__(positions, directions)

        self._bundle_id = self._next_bundle_id
        self._next_bundle_id += 1

        self.bundles.append(self)
        
        self._cs = cs
        
        if ids is None:
            ids = xp.arange(self._next_id, self._next_id + positions.shape[0])
            self._next_id += positions.shape[0]

        self._ids = ids

        if parents is None:
            parents = xp.zeros(positions.shape[0], dtype=int)

        self._parents = parents

        self._hits = BundleHits(self)

        self._scratch_arrays = defaultdict(list)
        
    def get_scratch(self, N=1):
        try:
            retval = self._scratch_arrays[N-1].pop()
        except:
            if N == 1:
                retval = xp.empty(self.n_rays, dtype=xp.float64)
            else:
                retval = xp.empty((self.n_rays, N), dtype=xp.float64)
        
        return retval

    def put_scratch(self, *args):
        for s in args:
            if len(s.shape) == 1:
                self._scratch_arrays[0].append(s)
            else:
                self._scratch_arrays[s.shape[1]-1].append(s)
        
        
    def toCS(self, newcs):
        if newcs != self.cs:
            return BundleAlias(self, newcs)

        return self

    @property
    def GCS(self):
        if self.cs != GCS:
            return BundleAlias(self, GCS)

        return self

    @property
    def cs(self):
        return self._cs
    
    @property
    def hits(self):
        return self._hits
    
    @property
    def ids(self):
        return self._ids

    @property
    def parents(self):
        return self._parents

    @property
    def n_rays(self):
        return self.positions.shape[0]
    
    @property
    def positions(self):
        return self._positions
    
    @property
    def directions(self):
        return self._directions

    @property
    def bundle_id(self):
        return self._bundle_id

    @property
    def root_bundle(self):
        return self

    @property
    def ids(self):
        return self._ids
    
    def __repr__(self):
        s = f"Bundle(\n"
        s = f"CS: {self.cs}\n"
        s += f"Positions: {self.positions}\n"
        s += f"Directions: {self.directions}\n"
        s += f")\n"
        
        return s

    def add_hits(self, obj, l, p, n):
        return self.hits.add(obj, self.directions, l, p, n)
    
    
class BundleAlias(_BundleBase):
    def __init__(self, bundle, cs):
        self._bundle = bundle
        self._cs = cs
        
        M = cs.fromGCS @ bundle.cs.toGCS
        
        super().__init__(M.batch_mult(bundle.positions),
                         M.batch_mult(bundle.directions))
        

    @property
    def hits(self):
        return self._bundle.hits
        
    @property
    def root_bundle(self):
        return self._bundle
        
    @property
    def ids(self):
        return self._bundle._ids

    @property
    def parents(self):
        return self._bundle._parents

    @property
    def bundle_id(self):
        return self._bundle.bundle_id
    
    def toCS(self, newcs):
        return BundleAlias(self._bundle, newcs)

    def add_hits(self, obj, l, p, n):
        return self.hits.add(obj, self.directions, l, p, n)

    def get_scratch(self, N=1):
        return self._bundle.get_scratch(N)

    def put_scratch(self, *args):
        self._bundle.put_scratch(*args)

    
