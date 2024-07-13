import pyvista as pv

from .trace import render_trace

class PVRenderer:
    def __init__(self):
        self._plotter = pv.Plotter()
        self._plotter.add_mesh(pv.Plane())
        
    def add_mesh(self, mesh, color="green"):
        self._plotter.add_mesh(mesh, color=color, opacity=0.5, line_width=3)
    
    def render_trace(self, trace):
        render_trace(self, trace)

    def show(self):
        self._plotter.show()
