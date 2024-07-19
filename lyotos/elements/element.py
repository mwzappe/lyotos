class Element:
    def __init__(self, cs):
        self._cs = cs

    @property
    def cs(self):
        return self._cs
        
    def intersect(self, ray):
        return self.do_intersect(ray.toCS(self.cs))
        
    def do_intersect(self, bundle):
        raise RuntimeError("Intersection is not implemented for class {self.__class__}")
        
