
from .surface import render_surface

import numpy as np
import pyvista as pv


def render_system(renderer, system):
    Nboundpts = 180
    
    render_surface(renderer, system.surfaces[0])

    last_s = None
    
    for s, n1, n2 in zip(system.surfaces, system._indexes[:-1], system._indexes[1:]):
        render_surface(renderer, s)

        aper_bound = s.aperture.boundary(Nboundpts)
        
        if n1 > 1:
            pts = [ ]
            for p1, p2 in zip(last_s.aperture.boundary(Nboundpts),
                              s.aperture.boundary(Nboundpts)):
                p1.z = last_s.sag(p1.x, p1.y)
                p2.z = s.sag(p2.x, p2.y)

                p1 = last_s.cs.toGCS @ p1
                p2 = s.cs.toGCS @ p2
                
                pts += [ p1, p2 ]

            pts = np.array([ [ p.x, p.y, p.z ] for p in pts ])
                
            strips = [ len(pts) ] + [ i for i in range(len(pts)) ]

            mesh = pv.PolyData(pts, strips=strips)

            renderer.add_mesh(mesh, color="#C0C0FF")
            print(pts)

        last_s = s
                
    
    for surf in system.surfaces:
        print(f"Rendering surface {surf}")
        
