from .coordinate_system import GCS, CSM, CoordinateSystem
from .surface import create_surface
from .aperture import CircularAperture


class SequentialSystem:
    def __init__(self, cs = GCS, aperture=200):
        self._surfaces = []
        self._indexes = []
        self._cs = cs
        self._curcs = cs

        if isinstance(aperture, (float, int)):
            aperture = CircularAperture(cs, aperture)
            
        self._aperture = aperture
        
    def add_surface(self, thickness, n, surf_type, tilt=None, **kwargs):
        if False:
            print(f"Adding surface: {thickness} {n} {surf_type} {kwargs}")
            
        if 'aperture' not in kwargs:
            kwargs['aperture'] = self._aperture            

        lens_cs = self._curcs
            
        if tilt is not None:
            lens_cs = lens_cs.xform(CSM.rotX(tilt))
            
        self._surfaces += [ create_surface(surf_type, lens_cs, **kwargs) ]

        if thickness == "auto":
            thickness = self._surfaces[-1].max_sag
        
        self._indexes += [ n ]
        self._curcs = CoordinateSystem(self._curcs, CSM.tZ(thickness))

    @property
    def surfaces(self):
        return self._surfaces
    
    
    
