import numpy as np

from lyotos.geometry import CoordinateSystem, GCS, CSM, Position, Vector
from .ray import Ray, NoHit
from .raypath import RayPath

class Tracer:
    def __init__(self, system, debug=False):
        self._system = system
        self.debug = debug
        self._trace_count = 0
        
    def debug_null(self, msg):
        pass

    def debug_print(self, msg):
        print(msg)

    @property
    def debug(self):
        return False if self._debug == self.debug_null else True

    @debug.setter
    def debug(self, v):
        if v == True:
            self._debug = self.debug_print
        else:
            self._debug = self.debug_null

    @property
    def trace_count(self):
        return self._trace_count
            
    @property
    def system(self):
        return self._system

    def trace_path(self, ray, color=None):
        debug = False
        self._trace_count += 1
        
        path = RayPath(color)

        last_hit = None
        last_surf = None

        if debug:
            print(f"***Trace {_trace_count}***")
        
        while True:
            l = None

            if debug:
                print(f"{_trace_count}: Testing ray: {ray}")

            try:
                l, p, normal, i, s = ray.find_surface(self.system.surfaces)
            except NoHit:
                if self.system.far_field is not None:
                    l, p, _ = self.system.far_field.intersect(ray)
                    
                    path.add_segment(ray, l, last_surf, self.system.far_field)
                break
                    
            n1 = self.system._indexes[i]
            n2 = self.system._indexes[i+1]

            rp = ray.toCS(s.cs)
                        
            ip = rp.at(l)

            if not s.aperture.inside(ip.x, ip.y):
                if debug:
                    print(f"{_trace_count}: Outside aperture: {ip.x} {ip.y} {s}")
                break

            ndi = normal @ rp.d

            # Going in the opposite direction of the conventional sense
            if ndi < 0:
                n1, n2 = n2, n1
                normal = -normal
                ndi = normal @ rp.d

            if debug:
                print(f"{_trace_count}: Processing hit:")
                print(f"{_trace_count}:  Surface: {s}")
                print(f"{_trace_count}:  l: {l} hit pos: {ip}")
                print(f"{_trace_count}:  incident: {rp.d}")
                print(f"{_trace_count}:  normal: {normal}")

            path.add_segment(ray, l, last_surf, s)

            if s.absorber:
                if debug:
                    print(f"{_trace_count}: Ray {ray} hit absorber")
                break
            
            last_surf = s
                
            mu = n1/n2

            # v2 = normal.cross(rp.d)
            # q2 = 1 - mu**2 * v2 @ v2

            
            q = 1 - mu**2 * (1 - ndi**2)

            if q < 0:
                # TIR, actually
                #print(f"TIR: mu: {mu} ndi: {ndi} {normal} {rp.d} {q} {n1} {n2}")

                r = rp.d - 2 * (rp.d @ normal) * normal

                ray = Ray(s._cs, ip, r)

                continue
        
            t = np.sqrt(q) * normal + mu * (rp.d - ndi * normal)

            if debug:
                print(f"{_trace_count}:  transmitted: {t}")

            ray = Ray(s._cs, ip, t)

        if path.empty:
            return None

        if self._target is None or path.hits_surface(self._target):
            self._paths += [ path ]

        return path
