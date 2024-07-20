from lyotos.geometry import GCS, CSM, CoordinateSystem, Position

class System:
    def __init__(self, cs=GCS, far_field=None):
        self._cs = GCS
        self._far_field = far_field

        self._elements = []

    @property
    def far_field(self):
        return self._far_field
        
    @property
    def elements(self):
        return self._elements
        
    def add_element(self, element):
        self._elements.append(element)

