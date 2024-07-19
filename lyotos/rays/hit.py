import cupy as cp

class HitBundle:
    def __init__(self, rays, surfaces, l, p, n):
        self._rays = rays
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
    def rays(self):
        return self._rays
    
    @property
    def surfaces(self):
        return self._surfaces
    
    @property
    def cs(self):
        return self.surface.cs
    
    def merge(self, other):
        l = [ l1 if l1 < l2 or cp.isnan(l2) else l2 for l1, l2 in zip(self.l, other.l) ]
        p = [ p1 if l1 < l2 or cp.isnan(l2) else p2 for l1, l2, p1, p2 in zip(self.l, other.l, self.p, other.p) ]
        n = [ n1 if l1 < l2 or cp.isnan(l2) else n2 for l1, l2, n1, n2 in zip(self.l, other.l, self.n, other.n) ]
        surfaces = [ s1 if l1 < l2 or cp.isnan(l2) else s2 for l1, l2, s1, s2 in zip(self.l, other.l, self.surfaces, other.surfaces) ]
        rays = [ r1 if l1 < l2 or cp.isnan(l2) else r2 for l1, l2, r1, r2 in zip(self.l, other.l, self.rays, other.rays) ]

        return HitBundle(rays, surfaces, l, p, n)
        
        
