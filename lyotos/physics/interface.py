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

            v = mu**2 * (1 - ct**2)

            if v <= 1:
                nc = math.sqrt(1 - v) - mu * ct

                t[i,0] = mu * d[i, 0] + nc * n[i, 0] 
                t[i,1] = mu * d[i, 1] + nc * n[i, 1] 
                t[i,2] = mu * d[i, 2] + nc * n[i, 2] 
            else:
                t[i, 0] = d[i, 0] + 2 * (d[i, 0] - ct * n[i, 0])
                t[i, 1] = d[i, 1] + 2 * (d[i, 1] - ct * n[i, 1])
                t[i, 2] = d[i, 2] + 2 * (d[i, 2] - ct * n[i, 2])

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

    if False:
        def interact(self, obj, hit_set):
            t = xp.empty((hit_set.p.shape))

            m1 = self.m1.n(hit_set.nu)
            m2 = self.m2.n(hit_set.nu)
            
            mu1 = m1 / m2
            mu2 = m2 / m1

            refract(t, hit_set.directions, hit_set.n, mu1, mu2)
            
            return [ Bundle(hit_set.p, t, cs = obj.cs, parents=hit_set.ids) ]
    else:
        def reflect(self, cs, p, d, n, ndi, ids):
            r = d - 2 * ndi * n
            
            return [ Bundle(p, t, cs=cs, parents=ids) ]

        def compute_boundary(self,  rs, rp, ndi, n1, n2):
            # sti = xp.sqrt(1 - ndi**2)
            #
            # np.sqrt(1 - (mu sti)**2)
            # np.sqrt(1 - mu**2 (1 - ndi**2))
            # np.sqrt(1 - mu**2 + mu**2 * ndi**2)

            mu2 = (n1/n2)**2
            
            cos2_theta_t = 1 - mu2 + mu2 * ndi**2 

            refract = cos2_theta_t > 0

            cos_theta_t = xp.sqrt(cos2_theta_t)

            rs[refract] = ((n1 * ndi[refract] - n2 * cos_theta_t[refract])/(n1 * ndi[refract] + n2 * cos_theta_t[refract]))**2
            rp[refract] = ((n1 * cos_theta_t[refract] - n2 * ndi[refract])/(n1 * cos_theta_t[refract] + n2 * ndi[refract]))**2

            refract[rs > 1] = False
            refract[rp > 1] = False

            rs = xp.clip(rs, 0, 1)
            rp = xp.clip(rp, 0, 1)
            
            return refract

            
        
        def do_interact(self, hit_set, cs, ndi, n1, n2):
            retval = []
            
            p = hit_set.p
            d = hit_set.directions
            n = hit_set.n
            
            r = xp.empty(p.shape)

            rs = xp.ones(p.shape[0])
            rp = xp.ones(p.shape[0])
            
            mu = n1/n2


            # Compute TIR 
            refract = self.compute_boundary(rs, rp, ndi, n1, n2)

            
            if xp.any(refract):
                hsr = hit_set.subset(refract)
                
                t = xp.empty((xp.count_nonzero(refract), 4))
                
                nc = xp.sqrt(1 - mu**2 * (1 - ndi[refract]**2)) - mu * ndi[refract]
                
                t[:, :] = mu * d[refract,:] + nc[:, xp.newaxis] * n[refract,:]

                assert not xp.any(xp.isnan(t))

                # Work with ray coordinate system, this is all wrong
                amps = hsr.amplitudes[:,0] * (1 - rs)
                ampp = hsr.amplitudes[:,1] * (1 - rp)
                
                retval += [ hsr.create_bundle(cs, t, amplitudes=xp.vstack((amps, ampp)).T) ]

            r[:,:] = d - 2 * ndi[:, xp.newaxis] * n

            amps = hit_set.amplitudes[:,0] * rs
            ampp = hit_set.amplitudes[:,1] * rp
            
            retval += [ hit_set.create_bundle(cs, r, amplitudes=xp.vstack((amps, amps)).T) ]

            return retval
        
        def interact(self, obj, hit_set):
            assert obj.id == hit_set.obj

            p = hit_set.p
            d = hit_set.directions
            n = hit_set.n

            ndi = xp.empty(p.shape[0])

            batch_dot(ndi, d, n)

            forward = ndi > 0
            reverse = xp.logical_not(forward)

            retval = []

            n1 = self.m1.n(hit_set.nu)
            n2 = self.m2.n(hit_set.nu)
            
            if xp.any(forward):
                hsf = hit_set.subset(forward)
                
                retval += self.do_interact(hsf, obj.cs, ndi, n1, n2)

            if xp.any(reverse): 
                hsr = hit_set.subset(reverse)
                retval += self.do_interact(hsr, obj.cs, ndi, n2, n1)

            return retval

            
