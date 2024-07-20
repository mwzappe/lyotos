import cupy as cp

class HitBundle:
    def __init__(self, origins, directions, surfaces, l, p, n):
        self._origins = origins
        self._directions = directions
        self._surfaces = surfaces
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
    def surfaces(self):
        return self._surfaces
    
    @property
    def cs(self):
        return self.surface.cs
    
    def merge(self, other):
        l = cp.array([ l1 if l1 < l2 or cp.isnan(l2) else l2 for l1, l2 in zip(self.l, other.l) ])
        p = cp.array([ p1 if l1 < l2 or cp.isnan(l2) else p2 for l1, l2, p1, p2 in zip(self.l, other.l, self.p, other.p) ])
        n = cp.array([ n1 if l1 < l2 or cp.isnan(l2) else n2 for l1, l2, n1, n2 in zip(self.l, other.l, self.n, other.n) ])
        surfaces = [ s1 if l1 < l2 or cp.isnan(l2) else s2 for l1, l2, s1, s2 in zip(self.l, other.l, self.surfaces, other.surfaces) ]
        origins = cp.array([ r1 if l1 < l2 or cp.isnan(l2) else r2 for l1, l2, r1, r2 in zip(self.l, other.l, self.origins, other.origins) ])
        directions = cp.array([ r1 if l1 < l2 or cp.isnan(l2) else r2 for l1, l2, r1, r2 in zip(self.l, other.l, self.directions, other.directions) ])

        return HitBundle(origins, directions, surfaces, l, p, n)
        
        
