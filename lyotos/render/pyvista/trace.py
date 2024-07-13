import pyvista as pv

from .system import render_system

from .colors import ray_color

arrows = False

def render_trace(renderer, trace):
    render_system(renderer, trace.system)
    
    for path in trace._paths:
        color = ray_color if path.color is None else path.color
        
        if arrows:
            for p1, p2 in zip(path.polyline[:-1], path.polyline[1:]):
                print(f"Arrow {p1}->{p2}")
                renderer.add_mesh(pv.Arrow(p1, p2-p1, scale='auto'), color=color)
        else:
            mesh = pv.lines_from_points(path.polyline)
            renderer.add_mesh(mesh, color=color)

