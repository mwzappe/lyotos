from .interaction import Interaction

from lyotos.util import batch_dot

class Mirror(Interaction):
    def __init__(self, surface):
        super().__init__(surface)

    def interact(self, positions, directions, normals):
        dp = directions - 2 * batch_dot(directions, normals) * normals

        return positions, dp
