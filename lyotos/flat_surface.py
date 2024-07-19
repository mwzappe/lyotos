from lyotos.geometry import Position, Vector
from .ray import NoHit
from .surface import Surface

class FlatSurface(Surface):
    surf_name="flat"

    
    
    def do_intersect(self, ray):
        l = -ray.pos[2]/ray.d[2]

        return l, ray.at(l), Vector.Z
