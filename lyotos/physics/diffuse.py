from .interaction import Interaction

class Diffuse(Interaction):
    def __init__(self, surface):
        super().__init__(surface)

    def interact(self, positions, directions, normals):
        pass
