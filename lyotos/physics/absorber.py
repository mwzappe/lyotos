import cupy as cp

from .interaction import Interaction

class Absorber(Interaction):
    def __init__(self, surface):
        super().__init__(surface)

    def interact(self, positions, directions, normals):
        return cp.array([]), cp.array([])
