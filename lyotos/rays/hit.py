import cupy as cp

from lyotos.util import darray
from lyotos.geometry import GCS

class HitBundle:
    def __init__(self, origins, directions, objects, l, p, n):
        self._origins = origins
        self._directions = directions
        self._objects = objects
        self._l = l
        self._p = p
        self._n = n
        
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
    def origins(self):
        return self._origins

    @property
    def directions(self):
        return self._directions
    
    @property
    def objects(self):
        return self._objects
        
    def merge(self, other):
        l = darray([ l1 if l1 < l2 or cp.isnan(l2) else l2 for l1, l2 in zip(self.l, other.l) ])
        p = darray([ p1 if l1 < l2 or cp.isnan(l2) else p2 for l1, l2, p1, p2 in zip(self.l, other.l, self.p, other.p) ])
        n = darray([ n1 if l1 < l2 or cp.isnan(l2) else n2 for l1, l2, n1, n2 in zip(self.l, other.l, self.n, other.n) ])
        objects = [ s1 if l1 < l2 or cp.isnan(l2) else s2 for l1, l2, s1, s2 in zip(self.l, other.l, self.objects, other.objects) ]
        origins = darray([ r1 if l1 < l2 or cp.isnan(l2) else r2 for l1, l2, r1, r2 in zip(self.l, other.l, self.origins, other.origins) ])
        directions = darray([ r1 if l1 < l2 or cp.isnan(l2) else r2 for l1, l2, r1, r2 in zip(self.l, other.l, self.directions, other.directions) ])

        return HitBundle(origins, directions, objects, l, p, n)

    def render(self, renderer):
        m = darray([ o.cs.toGCS._M for o in self.objects])

        sp = cp.einsum("ijk,ik->ij", m, self.origins)
        ep = cp.einsum("ijk,ik->ij", m, self.p)

        renderer.add_lines(GCS, sp, ep)

    def __repr__(self):
        s = "HitBundle(\n"
        s += f"Objects: {self.objects}\n"
        s += f"l: {self.l}\n"
        s += f"p: {self.p}\n"
        s += f"n: {self.n}\n"
        s += f"origins: {self.origins}\n"
        s += f"directions: {self.directions}\n"
        s += ")\n"
        
        return s
