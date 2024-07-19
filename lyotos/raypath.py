import numpy as np

from lyotos.geometry import GCS, CoordinateSystem, Position, Vector
from .ray import NoHit

class RaySegment:
    def __init__(self, ray, l, s1, s2):
        self._rp = ray.toGCS

        self._l = l

        self._s1 = s1
        self._s2 = s2

    @property
    def l(self):
        return self._l
        
    @property
    def p1(self):
        return self._rp.pos

    @property
    def p2(self):
        return self._rp.pos + self.l * self._rp.d

    @property
    def s1(self):
        return self._s1

    @property
    def s2(self):
        return self._s2

    
    def __repr__(self):
        return f"<{self.p1}->{self.p2}>"

class RayPath:
    def __init__(self, color=None):
        self._segments = []
        self._color = color

    def add_segment(self, ray, l, s1, s2):
        self._segments += [ RaySegment(ray, l, s1, s2) ]

    @property
    def final_point(self):
        return self._segments[-1].p2
        
    @property
    def empty(self):
        return len(self._segments) == 0

    @property
    def color(self):
        return self._color

    @property
    def cur_ray(self):
        return self._cur_ray

    @property
    def segments(self):
        return self._segments
    
    @property
    def polyline(self):
        if len(self._segments) == 0:
            return []
        
        return [ x.p1._v[:3] for x in self._segments ] + [ self._segments[-1].p2._v[:3] ]

    def find_hit(self, surface):
        for seg in self._segments:
            if seg.s2 == surface:
                return seg.p2

        raise NoHit()

    def hits_surface(self, surface):
        try:
            self.find_hit(surface)
            return True
        except NoHit:
            return False
