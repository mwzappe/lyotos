import numpy as np

import pyvista as pv

from .colors import surface_color

def render_surface(renderer, surface):
    if not surface.display:
        return
    
    pts = surface.aperture.generate_surface_mesh()

    z = np.array([ (x, y, surface.sag(x,y)) for x, y in pts ])
    
    D = 200

    #x = np.linspace(-D/2, D, 100)
    #y = np.linspace(-D/2, D, 100)

    #z = np.array([ (x, y, surface.sag(x,y)) for x in np.linspace(-D/2, D/2, 100) for y in np.linspace(-D/2, D/2, 100) ])

    mesh = pv.PolyData(z).delaunay_2d()

    mesh.transform(surface.cs.toGCS._M)

    renderer.add_mesh(mesh, color=surface_color if surface._color is None else surface._color)
