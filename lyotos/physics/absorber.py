from lyotos.util import xp

from .interaction import Interaction

class Absorber(Interaction):
    def __init__(self, surface):
        super().__init__(surface)

    def interact(self, positions, directions, normals):
        return xp.array([]), xp.array([])
