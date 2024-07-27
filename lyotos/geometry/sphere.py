from lyotos.util import xp, use_gpu, MISS

from lyotos.util import use_gpu, batch_dot, darray

from .position import Position

if use_gpu:
    from numba import cuda
    import math
    
    @cuda.jit(cache=True)
    def _intersect_kernel(l, p, d, R2):
        i = cuda.grid(1)

        if i < l.shape[0]:
            b = 2 * (p[i, 0] * d[i, 0] + p[i, 1] * d[i, 1] + p[i, 2] * d[i, 2])
            c = (p[i, 0] * p[i, 0] + p[i, 1] * p[i, 1] + p[i, 2] * p[i, 2]) - R2

            dsc = math.sqrt(b**2 - 4 * c)

            l[i,0] = (-b + dsc) / 2 if not dsc == math.nan else MISS
            l[i,1] = (-b - dsc) / 2 if not dsc == math.nan else MISS
            
            l[i,0] = l[i, 0] if l[i, 0] > 1e-7 else MISS
            l[i,1] = l[i, 1] if l[i, 1] > 1e-7 else MISS
            
class Sphere:
    if use_gpu:        
        @classmethod
        def intersect(cls, bundle, l, R, center=Position.CENTER.v):
            p = bundle.get_scratch(4)

            xp.subtract(bundle.positions, center, out=p)

            tpb = 128
            bpg = -(-l.shape[0] // tpb)

            _intersect_kernel[bpg, tpb](l, p, bundle.directions, R**2)
            
    else:
        @classmethod
        def intersect(cls, bundle, l, R, center=Position.CENTER.v):
            p = bundle.get_scratch(4)

            xp.subtract(bundle.positions, center, out=p)
            
            b = bundle.get_scratch()
            c = bundle.get_scratch()
            dsc = bundle.get_scratch()
        
            batch_dot(b, p, bundle.directions)
            batch_dot(c, p, p)

            xp.multiply(b, 2, out=b)
            xp.subtract(c, R**2, out=c)

            #a = 1
            # dsc = sqrt(b**2 - 4 * a (=1) * c)
            xp.power(b, 2, out=dsc)
            xp.multiply(c, 4, out=c)
            xp.subtract(dsc, c, out=dsc)
            xp.sqrt(dsc, out=dsc)
            
            bundle.put_scratch(p, b, c, dsc)
        
            l[:,0] = (-b + dsc) / 2
            l[:,1] = (-b - dsc) / 2
            
            l[xp.isnan(l)] = MISS
            # Avoid repeat intersection
            l[l < 1e-7] = MISS

        
