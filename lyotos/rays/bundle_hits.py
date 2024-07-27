from lyotos.util import xp

from lyotos.util import darray, iarray
from lyotos.geometry import GCS

from .base import MISS
from .hit_set import HitSet

class BundleHits:
    def __init__(self, bundle):
        self._bundle = bundle
        self._l = xp.ones(bundle.n_rays) * MISS
        self._d = xp.zeros((bundle.n_rays, 4))
        self._p = xp.zeros((bundle.n_rays, 4))
        self._n = xp.zeros((bundle.n_rays, 4))
        self._obj_stack = [ [] for _ in range(bundle.n_rays) ]
        
    def add(self, obj, d, l, p, n):
        idx = l < self.l

        self.d[idx,:] = d[idx]
        self.l[idx] = l[idx]
        self.p[idx,:] = p[idx]
        self.n[idx,:] = n[idx]

        for i in xp.argwhere(idx == True):
            self._obj_stack[int(i)] = [ obj.id ]
        
        return HitSet(self.bundle, idx)

    def render(self, renderer):
        sp = self.bundle.positions
        ep = sp + self.l[:,xp.newaxis] * self.bundle.directions

        renderer.add_lines(self.bundle.cs, sp, ep)

    @property
    def hit_sets(self):
        retval = {}
        
        objs = iarray([ ose[-1] for ose in self._obj_stack ])

        uo = xp.unique(objs)

        for oid in uo:
            retval[int(oid)] = HitSet(self.bundle, objs == oid)

        return retval
        
    @property
    def bundle(self):
        return self._bundle

    @property
    def d(self):
        return self._d
    
    @property
    def l(self):
        return self._l
        
    @property
    def p(self):
        return self._p

    @property
    def n(self):
        return self._n
        
