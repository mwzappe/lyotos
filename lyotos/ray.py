import numpy as np

from .coordinate_system import GCS, CoordinateSystem, Position, Vector

class Ray:
    def __init__(self, cs, pos=Position(), d=Vector()):
        assert isinstance(cs, CoordinateSystem)
        assert isinstance(pos, Position)
        assert isinstance(d, Vector)
        
        self._cs = cs
        self._pos = pos
        self._d = d

        if not cs.isGCS:
            M = self._cs.toGCS
            
            self._ingcs = Ray(GCS, M @ self._pos, M @ self._d)
        else:
            self._ingcs = self
            
    def toCS(self, newcs):
        M = newcs.fromGCS 
        rp = self.toGCS
        return Ray(newcs, M @ rp._pos, M @ rp._d)

    @property
    def toGCS(self):
        return self._ingcs
        
    @property
    def pos(self):
        return self._pos

    @property
    def d(self):
        return self._d

    def at(self, l):
        try:
            return np.array([ self._pos + l_ * self._d for l_ in l ])
        except TypeError:
            return self._pos + l * self._d
        
    @property
    def l_at_z0(self):
        return -self._pos[2] / self._d[2]

    
    def __repr__(self):
        return f"<RAY: {self._cs} {self._pos} {self._d}>"

class RaySegment:
    def __init__(self, ray, l, s1, s2):
        rp = ray.toGCS
        
        self._p1 = rp.pos
        self._p2 = rp.pos + l * rp.d

        self._s1 = s1
        self._s2 = s2
        
    @property
    def p1(self):
        return self._p1

    @property
    def p2(self):
        return self._p2

    @property
    def s1(self):
        return self._s1

    @property
    def s2(self):
        return self._s2

    
    def __repr__(self):
        return f"<{self.p1}->{self.p2}>"

class NoHit(Exception):
    pass
    
class RayPath:
    def __init__(self, color=None):
        self._segments = []
        self._color = color

    def add_segment(self, ray, l, s1, s2):
        self._segments += [ RaySegment(ray, l, s1, s2) ]

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
