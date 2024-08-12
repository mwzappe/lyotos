from lyotos.util import xp, use_gpu

from lyotos.util import darray, iarray, WavelengthToRGB
from lyotos.geometry import GCS

from .base import MISS
from .hit_set import HitSet

if use_gpu:
    from numba import cuda

    @cuda.jit(cache=True)
    def add_best_hit(idx, d, l, p, n):
        pass

    
class BundleHits:
    def __init__(self, bundle):
        self._bundle = bundle
        self._l = xp.ones(bundle.n_rays) * MISS
        self._d = xp.zeros((bundle.n_rays, 4))
        self._p = xp.zeros((bundle.n_rays, 4))
        self._n = xp.zeros((bundle.n_rays, 4))
        self._hit_set = xp.zeros(bundle.n_rays, dtype=int)

        self._hit_sets = [ HitSet(0, None, self.bundle) ]
        
        self._idx = xp.empty(bundle.n_rays, dtype=bool)
                     
    def add(self, obj, d, l, p, n):
        self._idx[:] = l < self.l

        if not xp.any(self._idx):
            return self._hit_sets[0]

        if obj is None:
            raise RuntimeError("Null object passed into add")
        
        retval = HitSet(len(self._hit_sets), obj, self.bundle)

        self._hit_sets.append(retval)
        
        self.d[self._idx,:] = d[self._idx]
        self.l[self._idx] = l[self._idx]
        self.p[self._idx,:] = p[self._idx]
        self.n[self._idx,:] = n[self._idx]

        self._hit_set[self._idx] = retval.id
                        
        return retval

    def render(self, renderer):
        if xp.any(self.l == MISS):
            print("WARNING: ESCAPED RAY")
            print(self.bundle)
            return

        sp = self.bundle.positions
        ep = sp + self.l[:,xp.newaxis] * self.bundle.directions
        
        renderer.add_lines(self.bundle.cs, sp, ep, WavelengthToRGB(self.bundle.nu), self.bundle.amplitudes[:,0])

    @property
    def hit_sets(self):
        retval = {}
        
        hit_sets = xp.unique(self._hit_set)

        for hsid in hit_sets:
            hsid = int(hsid)
            hs = self._hit_sets[hsid]
            self._idx[:] = self._hit_set == hsid 
            hs.set_idx(self._idx)
            retval[hs.obj.id] = hs

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
