from lyotos.geometry import GeometryObj
from lyotos.physics import Interface

class Surface(GeometryObj):
    def __init__(self, cs, interaction):
        super().__init__(cs)
        self._interaction = interaction

    @property
    def interaction(self):
        return self._interaction

    def interact(self, hit_set, m1, m2):
        
        new_rays = self.interaction.interact(self, hit_set)

        return new_rays
    
    def intersect(self, bundle):
        l = bundle.get_scratch()
        p = bundle.get_scratch(4)
        n = bundle.get_scratch(4)
        
        bundle = bundle.toCS(self.cs)
        
        self.do_intersect(bundle, l, p, n)

        retval = bundle.add_hits(self, l, p, n)

        bundle.put_scratch(l)
        bundle.put_scratch(p)
        bundle.put_scratch(n)

        return retval

    def render(self, renderer):
        raise RuntimeError("Render is not implemented for class {self.__class__.__name__}")
        
        
