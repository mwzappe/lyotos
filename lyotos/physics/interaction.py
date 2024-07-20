class Interaction:
    def __init__(self, surface):
        self._surface = surface

    @property
    def surface(self):
        return self._surface
        
    def interact(self, positions, directions, normals):
        raise RuntimeError("Interact is not defined for class {self.__class__}")
    
