import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from lyotos.geometry import GCS, CSM, Vector
from .ray import Ray
from .caster import Caster
from .sphere import SphPos





    
class Radiator(Caster):
    def __init__(self, trace, position, direction, theta, Npoints=100, color=None, target=None):
        self._trace = trace
        self._theta = theta
        self._target = target
        self._paths = []

        system = trace.system

        self.pos = np.append([ SphPos(0, 0) ],  saR2(np.arange(Npoints), theta))
        
        self.triangles = Delaunay([ p.projected for p in self.pos  ],
                                  incremental=True).simplices

        cs = GCS.xform(CSM.align_z(direction) @ CSM.translate(position))
        
        self.rays = [ Ray(cs.xform(CSM.rot2(p.phi, p.theta))) for p in self.pos ]

        self.all_paths = []
        
        for ray in self.rays:
            path = trace.trace_path(ray, color="purple")

            self._maybe_append_path(path)

            self.all_paths += [ path ]
        
        self.all_paths = np.array(self.all_paths)

        # Fix me -- quick hack

        R = trace.system.far_field.R

        def compute_gain(tri):
            paths = self.all_paths[tri]

            if np.any(paths == None):
                return -1e10
            
            spos = self.pos[tri]

            a = spos[0].cartesian
            b = spos[1].cartesian
            c = spos[2].cartesian

            area1 = 2 * np.arctan(np.abs(a @ np.cross(b, c)) / (1 + a @ b + b @ c + a @ c))

            ap = paths[0].final_point
            bp = paths[1].final_point - ap
            xp.= paths[2].final_point - ap
            
            area2 = bp.cross(xp..norm / 2
            
            # If we take the mean of the radiance function at a, b, c
            # then we can multiply by the area to get the total power in a triangle
            #
            # This power is then distributed over area2.
            #
            # If the radiator was isotropic, it would be over area * R**2

            P = area1
            iso_area = area1 * R**2
            
            PDreal = P / area2
            PDiso = P / (area1 * R**2)

            return PDreal/PDiso

        
        self.gains = [ compute_gain(tri) for tri in self.triangles ]

            
            
            
        
