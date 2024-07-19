import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

from lyotos.geometry import GCS, CSM, Vector
from .ray import Ray
from .caster import Caster
from .sphere import SphPos



_g = 1.32471795724474602596
a1 = 1.0/_g
a2 = 1.0/(_g*_g)

@np.vectorize
def R2(n):
    return (0.5+a1*n) %1, (0.5+a2*n) %1

def sphR2(n):
    # Lambert project
    u, v = R2(n)

    return np.arccos(2 * u - 1), 2 * np.pi * v

def discR2(n):
    u, v = R2(n)

    return np.sqrt(u), 2 * np.pi * v
    
@np.vectorize
def saR2(n, theta):
    u, v = R2(n)

    r = np.sqrt(u * theta / np.pi)
    theta = 2 * np.pi * v

    return SphPos(np.pi - 2 * np.arccos(r), theta)


    
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
            cp = paths[2].final_point - ap
            
            area2 = bp.cross(cp).norm / 2
            
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

            
            
            
        
