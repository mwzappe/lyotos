from lyotos.geometry import GeometryObj
from .compound_obj import CompoundObj

class Element(CompoundObj):
    def __init__(self, cs):
        super().__init__(cs)
                        
    def do_trace(self, hit_set):
        raise RuntimeError(f"Intersection is not implemented for class {self.__class__.__name__}")
            
    def render(self, renderer):
        raise RuntimeError(f"Render is not implemented for class {self.__class__.__name__}")

    def test_hit(self, bundle):
        bundle = bundle.toCS(self.cs)

        for b in self.boundary:
            b.intersect(bundle).push_obj(self)
