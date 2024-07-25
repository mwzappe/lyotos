from lyotos.geometry import GCS, CSM, CoordinateSystem, Position
from lyotos.elements import CompoundObj

class System(CompoundObj):
    def __init__(self, cs=GCS, far_field=None):
        super().__init__(cs)
        self._cs = GCS
        self._far_field = far_field

        self._elements = [ ]

        if far_field is not None:
            self._elements.append(far_field)
        
                
    @property
    def far_field(self):
        return self._far_field

    @property
    def children(self):
        return self.elements
    
    @property
    def elements(self):
        return self._elements
        
    def add_element(self, element):
        self._elements.append(element)

