import numpy as np

from lyotos.geometry import CoordinateSystem, GCS, CSM, Position, Vector
from .ray import Ray, NoHit
from .raypath import RayPath
from .caster import Caster

class InfiniteFan(Caster):
    def __init__(self, trace, theta=0, D=None, color=None, target=None):
        self._trace = trace
        self._theta = 0
        self._color = color
        self._paths = []
        self._target = target
        
        system = trace.system
        
        if D is None:
            D = system.surfaces[0].aperture.height

        self._D = D

        Npoints = 10
        
        dy = D/2 / Npoints
        
        M = system.surfaces[0].cs.toGCS

        # Cheap, dirty, and wrong, but not too wrong
        dist = 400

        base_cs = system.surfaces[0].cs.xform(CSM.tZ(-dist) @ CSM.rotX(theta))
        
        self._maybe_append_path(trace.trace_path(Ray(base_cs), color=color))

        for x in ((np.arange(Npoints)+1) * dy):
            self._maybe_append_path(trace.trace_path(Ray(base_cs.xform(CSM.tX(x))), color=color))
            self._maybe_append_path(trace.trace_path(Ray(base_cs.xform(CSM.tX(-x))), color=color))

        
        for y in ((np.arange(Npoints)+1) * dy):
            pycs = base_cs.xform(CSM.tY(y))
            nycs = base_cs.xform(CSM.tY(-y))
            self._maybe_append_path(trace.trace_path(Ray(pycs), color=color))
            self._maybe_append_path(trace.trace_path(Ray(nycs), color=color))

            for x in ((np.arange(Npoints)+1) * dy):
                self._maybe_append_path(trace.trace_path(Ray(pycs.xform(CSM.tX(x))), color=color))
                self._maybe_append_path(trace.trace_path(Ray(pycs.xform(CSM.tX(-x))), color=color))
                self._maybe_append_path(trace.trace_path(Ray(nycs.xform(CSM.tX(x))), color=color))
                self._maybe_append_path(trace.trace_path(Ray(nycs.xform(CSM.tX(-x))), color=color))

    def _maybe_append_path(self, path):
        if (path is not None) and ((self._target is None) or path.hits_surface(self._target)):
            self._paths.append(path)
        
            
    def spot_size(self, surface):
        pts = []
        for p in self._paths:
            try:
                pts.append(p.find_hit(surface))
            except NoHit:
                continue
        
        mean = sum(pts, start=Position.from_xyz(0,0,0)) / len(pts)

        s = 0
        
        for p in pts:
            s += (p - mean).norm**2

        s = np.sqrt(s / len(pts))

        return s
