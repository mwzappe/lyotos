from lyotos.util import xp, use_gpu, MISS

from lyotos.util import batch_dot, darray

from .position import Position

class Sphere:
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

        
