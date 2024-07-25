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
        bundle = bundle.toCS(self.cs)
        
        l, p, n = self.do_intersect(bundle)

        return bundle.add_hits(self, l, p, n)

    def render(self, renderer):
        raise RuntimeError("Render is not implemented for class {self.__class__.__name__}")
        
        
