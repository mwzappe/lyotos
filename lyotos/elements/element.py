from lyotos.rays import HitBundle

class Element:
    def __init__(self, cs):
        self._cs = cs

    @property
    def cs(self):
        return self._cs
        
    def trace(self, bundle):
        pass
        #return self.do_trace(ray.toCS(self.cs))
        
    def do_trace(self, bundle):
        raise RuntimeError("Intersection is not implemented for class {self.__class__}")
        
    def intersect(self, bundle):
        bundle = bundle.toCS(self.cs)
        
        l, p, n = self.do_intersect(bundle)

        return HitBundle(bundle.positions, bundle.directions, [ self for i in range(len(l)) ], l, p, n)

    def do_intersect(self, bundle):
        raise RuntimeError("Intersection is not implemented for class {self.__class__}")

    def render(self, renderer):
        raise RuntimeError("Render is not implemented for class {self.__class__}")
