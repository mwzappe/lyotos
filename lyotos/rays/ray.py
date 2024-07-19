import numpy as np

from lyotos.geometry import GCS, CoordinateSystem, Position, Vector

MAXFLOAT = np.finfo(float).max
hit_tolerance = 1e-9

class NoHit(Exception):
    pass
    
class Ray:
    def __init__(self, cs, pos=Position(), d=Vector()):
        assert isinstance(cs, CoordinateSystem)
        assert isinstance(pos, Position)
        assert isinstance(d, Vector)
        assert np.isclose(d.norm, 1)
        
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
    def cs(self):
        return self._cs
    
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


    def find_surface(self, surfaces):
        hit = MAXFLOAT, None, None, None, None

        for i, s in enumerate(surfaces):
            try:
                l, p, n = s.intersect(self)
            except NoHit:
                continue

            if l > hit_tolerance and l < hit[0]:
                hit = l, p, n, i, s

        if hit[1] is None:
            raise NoHit()

        print(f"Find Surface: {hit}")
        
        return hit
    
    
    def __repr__(self):
        assert np.isclose(self._d.norm, 1)
        return f"<RAY: {self._cs} {self._pos} {self._d}>"

