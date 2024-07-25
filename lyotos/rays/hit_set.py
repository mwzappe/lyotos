import cupy as cp

from lyotos.util import iarray

from .hit_base import HitBase

class HitSet:
    def __init__(self, bundle, idx):
        self._bundle = bundle
        self._idx = idx

    def push_obj(self, obj):
        for i in cp.argwhere(self._idx == True):
            self.bundle.hits._obj_stack[int(i)].append(obj.id)

    def pop_obj(self):
        objs = [ self.bundle.hits._obj_stack[int(i)] for i in cp.argwhere(self._idx == True) ]

        for o in objs:
            o.pop()

        uo = cp.unique(objs)
            
        retval = {}
            
        objs = iarray([ ose[-1] for ose in self.bundle.hits._obj_stack ])

        for oid in uo:
            retval[int(oid)] = HitSet(self.bundle, objs == oid)

        return retval
        
        
            
    @property
    def bundle(self):
        return self._bundle

    @property
    def ids(self):
        return self._bundle.ids[self._idx]
    
    @property
    def l(self):
        return self.bundle.hits.l[self._idx]

    @property
    def p(self):
        return self.bundle.hits.p[self._idx]

    @property
    def n(self):
        return self.bundle.hits.n[self._idx]
        
    @property
    def nu(self):
        return self.bundle.nu
        
    @property
    def positions(self):
        return self.bundle.positions[self._idx]
    
    @property
    def directions(self):
        return self.bundle.directions[self._idx]

    def pts_at(self, ls):
        return self.positions + cp.einsum("i,ij->ij", ls, self.directions)

    def __len__(self):
        return self.positions.shape[0]

    def __eq__(self, other):
        return self.bundle_id == other.bundle_id
    
