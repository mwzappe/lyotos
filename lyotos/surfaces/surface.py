from lyotos.rays import HitBundle
from lyotos.physics import Interface

class Surface:
    def __init__(self, cs, interaction_cls=Interface):
        self._cs = cs
        self._interaction = interaction_cls(self)

    @property
    def interaction(self):
        return self._interaction
        
    @property
    def cs(self):
        return self._cs

    def intersect(self, bundle):
        l, p, n = self.do_intersect(bundle.toCS(self.cs))

        return HitBundle(bundle.positions, bundle.directions, [ self for i in range(len(l)) ], l, p, n)

    def render(self, renderer):
        raise RuntimeError("Render is not implemented for class {self.__class__}")
        
        
