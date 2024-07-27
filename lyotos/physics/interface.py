from lyotos.util import xp

from lyotos.util import batch_dot
from lyotos.rays import Bundle

from .interaction import Interaction

class Interface(Interaction):
    def __init__(self, m1, m2):
        super().__init__()
        self._m1 = m1
        self._m2 = m2

    @property
    def m1(self):
        return self._m1

    @property
    def m2(self):
        return self._m2
        
    def interact(self, obj, hit_set):
        p = hit_set.p
        d = hit_set.directions
        n = hit_set.n

        ct = xp.empty(p.shape[0])

        batch_dot(ct, d, n)

        #print(f"Directions: {d}")
        #print(f"Normals: {n}")
        #print(f"Dot products: {ct}")

        m1 = xp.ones(len(ct)) * self.m1.n(hit_set.nu)
        m2 = xp.ones(len(ct)) * self.m2.n(hit_set.nu)

        m1[ct < 0] = self.m2.n(hit_set.nu)
        m2[ct < 0] = self.m1.n(hit_set.nu)

        mu = m1/m2

        nc = xp.sqrt(1 - mu**2 * (1 - ct**2)) - mu * ct
        
        #print(m1)
        #print(m2)
        #print(nc)

        t = mu[:,xp.newaxis] * d + nc[:,xp.newaxis] * n

        #print(t)
        
        return [ Bundle(p, t, cs = obj.cs, parents=hit_set.ids) ]
