from lyotos.rays import HitBundle

class Surface:
    def __init__(self, cs):
        self._cs = cs

    @property
    def cs(self):
        return self._cs

    def intersect(self, bundle):
        l, p, n = self.do_intersect(bundle.toCS(self.cs))

        return HitBundle(bundle, self, l, p, n)
        
        
        
