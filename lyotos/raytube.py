import numpy as np

from .raypath import RayPath

class RayTube:
    def __init__(self, p1, p2, p3):
        paths = [ p1, p2, p3 ]
        
        segments = [ p.segments[0] for p in paths ]
        
        lengths = [ s.l for s in segments ]

        m = np.argmin(lengths)
        
        self.G = paths[m]

        paths = np.delete(paths, m)

        if lengths[0] < lengths[1]:
            self.E, self.D = paths
        else:
            self.D, self.E = paths

        print(f"{self.G} {self.E} {self.D}")

        

    
