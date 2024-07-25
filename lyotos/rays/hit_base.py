import cupy as cp

class HitBase:
    def __init__(self, bundle, objects, l, p, n):
        self._l = l
        self._p = p
        self._n = n
        self._bundle = bundle.root_bundle
        self._obj_stack = [ objects ]

    @property
    def nu(self):
        return self.bundle.nu
    
    @property
    def l(self):
        return self._l

    @property
    def p(self):
        return self._p

    @property
    def n(self):
        return self._n

    @property
    def objects(self):
        return self._obj_stack[-1]

    @property
    def bundle(self):
        return self._bundle

    @property
    def ids(self):
        return self._bundle.ids

    def push_obj(self, obj):
        self._obj_stack.append([ obj ] * self.l.shape[0])

    def pop_obj(self):
        self._objects = self._obj_stack.pop()
