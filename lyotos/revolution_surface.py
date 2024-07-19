
from .surface import Surface

class RevolutionSurface(Surface):
    def __init__(self, cs):
        super().__init__(cs)


    def sag(self, r):
        raise RuntimeException("Saggita function not defined for class {self.__class__}")

    #
    # Ray R with p + l * d
    #
    # r = sqrt((p.x + l * d.x)**2 + (p.y + l * d.y)**2)
    #
    # sag(sqrt((p.x + l * d.x)**2 + (p.y + l * d.y)**2)) == p.z + l * d.z
    #
    # 
