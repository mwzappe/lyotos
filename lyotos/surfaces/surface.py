from lyotos.rays import HitBundle

class Surface:
    def __init__(self, cs):
        self._cs = cs

    @property
    def cs(self):
        return self._cs

    def intersect(self, bundle):
        bundle = bundle.toCS(self.cs)
        
        l, p, n = self.do_intersect(bundle)

        return HitBundle(bundle.bundle, [ self for i in range(len(l)) ], l, p, n)
        
        
        
