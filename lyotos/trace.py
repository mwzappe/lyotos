import numpy as np

from .coordinate_system import CoordinateSystem, GCS, CSM, Position, Vector
from .ray import Ray, RayPath, NoHit


class Trace:
    def __init__(self, system, target=None):
        self._system = system
        self._paths = [ ]
        self._target = None

    @property
    def system(self):
        return self._system
        
    def trace_path(self, ray, color=None, ignore_surf0=False):
        path = RayPath(color)

        last_hit = None
        last_surf = None
        
        while True:
            hit = None
            best_l = 1e10

            def test_surface(s):
                rp = ray.toCS(s.cs)

                l = s.intersect(rp)

                return l, rp

            for s, n1, n2 in zip(self.system.surfaces[1:], self.system._indexes[:-1], self.system._indexes[1:]):
                try:
                    l, rp = test_surface(s)
                except NoHit:
                    continue

                if l > 1e-7 and l < best_l:
                    best_l = l
                    hit = (s, n1, n2, rp, l)                    
                    
            if not ignore_surf0:
                l, rp = test_surface(self.system.surfaces[0])

                if l > 1e-7 and l < best_l:
                    path.add_segment(ray, l, last_surf, self.system.surfaces[0])                
                    break
                    
            if hit is None:
                break
            
            s, n1, n2, rp, l = hit
            
            ip = rp.at(l)

            if not s.aperture.inside(ip.x, ip.y):
                break
            
            normal = s.normal(ip[0], ip[1])

            ndi = normal @ rp.d

            # Going in the opposite direction of the conventional sense
            if ndi < 0:
                print("Going backwards")
                n1, n2 = n2, n1
                normal = -normal
                ndi = normal @ rp.d

            if False:
                print(f"Processing hit:")
                print(f" Surface: {s}")
                print(f" l: {l} hit pos: {ip}")
                print(f" incident: {rp.d}")
                print(f" normal: {normal}")

            path.add_segment(ray, l, last_surf, s)

            if s.absorber:
                break
            
            last_surf = s
                
            mu = n1/n2

            # v2 = normal.cross(rp.d)
            # q2 = 1 - mu**2 * v2 @ v2

            
            q = 1 - mu**2 * (1 - ndi**2)

            if q < 0:
                # TIR, actually
                print(f"TIR: mu: {mu} ndi: {ndi} {normal} {rp.d} {q} {n1} {n2}")

                r = rp.d - 2 * (rp.d @ normal) * normal

                ray = Ray(s._cs, ip, r)

                continue
        
            t = np.sqrt(q) * normal + mu * (rp.d - ndi * normal)

            if False:
                print(f" transmitted: {t}")

            ray = Ray(s._cs, ip, t)

        if path.empty:
            return None

        if self._target is None or path.hits_surface(self._target):
            self._paths += [ path ]

        return path


class InfiniteFan:
    def __init__(self, trace, theta=0, D=None, color=None, target=None):
        self._trace = trace
        self._theta = 0
        self._color = color
        self._paths = []
        self._target = target
        
        system = trace.system
        
        if D is None:
            D = system.surfaces[1].aperture.height

        self._D = D

        Npoints = 10
        
        dy = D/2 / Npoints
        
        M = system.surfaces[1].cs.toGCS

        # Cheap, dirty, and wrong, but not too wrong
        dist = M._M[2][3]

        base_cs = system.surfaces[1].cs.xform(CSM.tZ(-dist) @ CSM.rotX(theta))
        
        self._maybe_append_path(trace.trace_path(Ray(base_cs), color=color, ignore_surf0=True))
                           
        for y in ((np.arange(Npoints)+1) * dy):
            self._maybe_append_path(trace.trace_path(Ray(base_cs.xform(CSM.tY(y))), color=color, ignore_surf0=True))
            self._maybe_append_path(trace.trace_path(Ray(base_cs.xform(CSM.tY(-y))), color=color, ignore_surf0=True))
            self._maybe_append_path(trace.trace_path(Ray(base_cs.xform(CSM.tX(y))), color=color, ignore_surf0=True))
            self._maybe_append_path(trace.trace_path(Ray(base_cs.xform(CSM.tX(-y))), color=color, ignore_surf0=True))

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

        

def illuminate(trace, position):
    cs = GCS.xform(CSM.rotX(np.pi) @ CSM.translate(position))

    #print(cs)

    for theta in np.linspace(-10, 10, 11):
        cs2 = cs.xform(CSM.rotX(theta * np.pi / 180))
        trace.trace_path(Ray(cs2))
    
