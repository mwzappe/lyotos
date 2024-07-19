import pyvista as pv

from .trace import render_trace

class PVRenderer:
    def __init__(self):
        self._plotter = pv.Plotter()
        self._plotter.add_mesh(pv.Plane())
        
    def add_mesh(self, mesh, **kwargs):
        if "opacity" not in kwargs:
            kwargs["opacity"] = 0.5

        if "line_width" not in kwargs:
            kwargs["line_width"] = 3
            
        self._plotter.add_mesh(mesh, **kwargs)
    
    def render_trace(self, trace):
        render_trace(self, trace)

    def show(self):
        self._plotter.show()
