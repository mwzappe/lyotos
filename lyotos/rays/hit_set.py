from lyotos.util import xp

from lyotos.util import iarray

class HitSet:
    def __init__(self, id, obj, bundle):
        self._id = id
        self._bundle = bundle

        if obj is not None:
            self._obj_stack = [ obj.id ]
        else:
            self._obj_stack = None
            
    def set_idx(self, idx):
        self._idx = idx
        
    def push_obj(self, obj):
        # Only None for the all-miss HitSet 0
        if self._obj_stack is not None:
            self._obj_stack.append(obj)
        
    def pop_obj(self):
        return self._obj_stack.pop()

    @property
    def obj(self):
        return self._obj_stack[-1]
        
    @property
    def id(self):
        return self._id
    
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
        return self.positions + xp.einsum("i,ij->ij", ls, self.directions)

    def __len__(self):
        return self.positions.shape[0]

    def __eq__(self, other):
        return self.bundle_id == other.bundle_id
    
