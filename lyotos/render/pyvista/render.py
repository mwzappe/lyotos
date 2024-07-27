from lyotos.util import xp

import pyvista as pv

from lyotos.util import iarray
from lyotos.rays import Bundle

from .trace import render_trace

class PVRenderer:
    def __init__(self, system):
        self._plotter = pv.Plotter()
        self._system = system
        
    def add_cylinder(self, cs, R, h, capping=False):
        M = cs.toGCS

        mesh = pv.Cylinder(center=(0,0,h/2), direction=(0,0,1), radius=R, height=h, capping=capping)

        mesh.transform(xp.get(M._M))

        self._add_mesh(mesh)

    def add_spherical_cap(self, cs, R, r):
        M = cs.toGCS

        phi = xp.arcsin(r/R) * 180/xp.pi
        
        if R > 0:
            mesh = pv.Sphere(radius=R,
                             center=(0.0, 0.0, R),
                             direction=(0.0, 0.0, -1.0),
                             end_phi=phi)
        else:
            mesh = pv.Sphere(radius=-R,
                             center=(0.0, 0.0, R),
                             direction=(0.0, 0.0, 1.0),
                             end_phi=-phi)

            
        mesh.transform(xp.get(M._M))

        self._add_mesh(mesh)

    def add_lines(self, cs, start_points, end_points):
        M = cs.toGCS

        pts = xp.get(xp.concatenate((start_points, end_points))[:,:3])

        lines = xp.get(iarray([ [ 2, i, i + start_points.shape[0] ] for i in range(start_points.shape[0]) ]).flatten())
        
        mesh = pv.PolyData(pts, lines=lines)
        
        mesh.transform(xp.get(M._M))

        self._add_mesh(mesh)
        
    def _add_mesh(self, mesh, **kwargs):
        if "opacity" not in kwargs:
            kwargs["opacity"] = 0.5

        #if "line_width" not in kwargs:
        #    kwargs["line_width"] = 3
            
        return self._plotter.add_mesh(mesh, **kwargs)

    def show(self):
        self._system.render(self)
        
        for b in Bundle.bundles:
            b.hits.render(self)
        
        s = pv.Sphere(radius=1, center=(0, 0, 000))

        self._plotter.add_mesh(s, color="red")
        
        s = pv.Sphere(radius=1, center=(0, 0, 100))        

        self._plotter.add_mesh(s, color="green")

        s = pv.Sphere(radius=1, center=(0, 0, 200))        

        self._plotter.add_mesh(s, color="blue")

        self._plotter.view_zx()
        #self._plotter.set_viewup([1, 0, 0])
        self._plotter.show_axes()
        self._plotter.show()
