from lyotos.util import xp

from lyotos.geometry import CSM, GCS, Position, Sphere
from lyotos.rays import MISS
from lyotos.physics import Absorber

from lyotos import surfaces

from lyotos.physics import Interaction
from .element import Element

class FarFieldInteraction(Interaction):
    pass


class FarField(Element):
    def __init__(self, cs=GCS, center=Position.CENTER, radius=1e4):
        super().__init__(cs=cs.xform(CSM.translate(center)))

        self._surface = surfaces.Sphere(self.cs, FarFieldInteraction(), radius)
        
        self._center = center        

        self._hits = []
        
    @property
    def boundary(self):
        return [ self._surface ]
        
    @property
    def center(self):
        return self._center
    
    @property
    def R(self):
        return self._surface.R
        
    @property
    def absorber(self):
        return True

    def propagate(self, hit_set):
        # Keep an array of rays by direction
        self._hits.append(hit_set)
        
        return []

    
    def render(self, renderer):
        pass
