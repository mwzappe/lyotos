from lyotos.util import xp, use_gpu

from lyotos.util import batch_dot
from lyotos.rays import Bundle

from .interaction import Interaction

if use_gpu:
    from numba import cuda
    import math
    
    @cuda.jit(cache=True)
    def _refract(t, d, n, mu1, mu2):
        i = cuda.grid(1)

        if i < t.shape[0]:
            ct = d[i,0] * n[i, 0] + d[i, 1] * n[i, 1] + d[i, 2] * n[i, 2]

            if ct > 0:
                mu = mu1
            else:
                mu = mu2

            nc = math.sqrt(1 - mu**2 * (1 - ct**2)) - mu * ct

            t[i,0] = mu * d[i, 0] + nc * n[i, 0] 
            t[i,1] = mu * d[i, 1] + nc * n[i, 1] 
            t[i,2] = mu * d[i, 2] + nc * n[i, 2] 
            t[i,3] = 0
            
    def refract(t, d, n, m1, m2):
        tpb = 128
        bpg = -(-t.shape[0] // tpb)

        _refract[bpg, tpb](t, d, n, m1, m2)
        
                           
                
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

    if use_gpu:
        def interact(self, obj, hit_set):
            t = xp.empty((hit_set.p.shape))

            m1 = self.m1.n(hit_set.nu)
            m2 = self.m2.n(hit_set.nu)
            
            mu1 = m1 / m2
            mu2 = m2 / m1

            refract(t, hit_set.directions, hit_set.n, mu1, mu2)
            
            return [ Bundle(hit_set.p, t, cs = obj.cs, parents=hit_set.ids) ]
    else:
        def interact(self, obj, hit_set):
            p = hit_set.p
            d = hit_set.directions
            n = hit_set.n

            ct = xp.empty(p.shape[0])

            batch_dot(ct, d, n)

            m1 = xp.ones(len(ct)) * self.m1.n(hit_set.nu)
            m2 = xp.ones(len(ct)) * self.m2.n(hit_set.nu)

            m1[ct < 0] = self.m2.n(hit_set.nu)
            m2[ct < 0] = self.m1.n(hit_set.nu)

            mu = m1/m2

            nc = xp.sqrt(1 - mu**2 * (1 - ct**2)) - mu * ct
            
            t = mu[:,xp.newaxis] * d + nc[:,xp.newaxis] * n
            
            return [ Bundle(p, t, cs = obj.cs, parents=hit_set.ids) ]
