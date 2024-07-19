import cupy as cp

from lyotos.geometry import CSM, Vector, Sphere
from lyotos.surfaces import SphericalSurface
from lyotos.materials import Vacuum

from .shape import thin_radii, thick_radii
from .element import Element

def max_aperture(r1, r2, t):
    if r1 == 0:
        return np.abs(2*r2)
    if r2 == 0:
        return np.abs(2*r1)
        
    x2 = Sphere.intersect(t+r2-r1, r1, r2)**2

    if x2 > r1**2:
        return 2 * np.min(np.abs([r1, r2]))

    rp = np.sqrt(r1**2 - x2)

    return 2*rp

    
    zz = 4 * dist**2 * r1**2 - (dist**2-r2**2+r1**2)**2
    
    print(f"zz: {zz}")
        
    if zz < 0:
        return 2*np.min(np.abs([ r1, r2 ]))

    return np.abs(1/dist * np.sqrt(zz))
    

class Lens(Element):
    def __init__(self, R1, R2, t, n, aper):
        self._R1 = R1
        self._R2 = R2
        self._t = t
        self._n = n
        self._aper = aper

    @property
    def n(self):
        return self._n

    @property
    def t(self):
        return self._t

    @property
    def R1(self):
        return self._R1

    @property
    def R2(self):
        return self._R2
    
    @property
    def P(self):
        return (self.n - 1) * (1/self.R1 - 1/self.R2 + (self.n-1)*self.t / (self.n * self.R1 * self.R2))

    @property
    def h1(self):
        return -self.f * (self.n - 1) * self.t / (self.R2 * self.n)

    @property
    def h2(self):
        return -self.f * (self.n - 1) * self.t / (self.R1 * self.n)

    
    @property
    def aper(self):
        return self._aper

    
    @property
    def f(self):
        return 1/self.P

    @property
    def q(self):
        return (self.R1 + self.R2)/(self.R1 - self.R2)

    @property
    def max_aperture(self):
        return max_aperture(self.R1, self.R2, self.t)
                    
    
    
    @classmethod
    def with_shape(cls, f, q, n, aper=None, t=1):
        # Start with thin lens approx, and solve for r1
        print(f"Creating lens with focal length {f}")

        r1, r2 = thick_radii(f, q, n, t)
        
        print(f"Thick lens: {r1} {r2}")
        
        return Lens(r1, r2, t, n, aper)

class MultipletLens(Element):
    def __init__(self, cs, *, materials, Rs, ts, aperture, surroundings=Vacuum):
        super().__init__(cs)

        assert len(materials) + 1 == len(Rs)
        assert len(materials) == len(ts)
        
        cur_cs = cs
        self._surfaces = []
        self._materials = materials
        self._surroundings = Vacuum

        self._surfaces.append(SphericalSurface(cs=cur_cs, R=Rs[0]))
        
        for R, t in zip(Rs[1:], ts):
            cur_cs = cur_cs.xform(CSM.translate(t * Vector.Z))
            self._surfaces.append(SphericalSurface(cs=cur_cs, R=R))
            
            
    @property
    def surfaces(self):
        return self._surfaces    

    @property
    def materials(self):
        return self._materials    

    @property
    def surroundings(self):
        return self._surroundings
    
    def do_intersect(self, bundle):
        hit = None
        best_l = None

        p, d = bundle.positions, bundle.directions

        hits = self.surfaces[0].intersect(bundle)

        for s in self.surfaces[1:]:
            hits = hits.merge(s.intersect(bundle))

        print(hits.surfaces)


    @property
    def petzval_sum(self, nu=None):
        outside_n = self.surroundings.n(nu) 
        last_n = outside_n
        retval = 0
        
        for s, m in zip(self.surfaces[:-1], self.materials):
            n = m.n(nu)
            print(f"last n: {last_n} n: {n}")
            retval += (n - last_n) / (s.R * n * last_n)
            last_n = n

        retval += (outside_n - last_n)/(self.surfaces[-1].R * outside_n * last_n)
        
        return retval


class SingletLens(MultipletLens):
    def __init__(self, cs, *, material, R1, R2, t, aperture):
        super().__init__(cs, materials = [ material ], Rs = [ R1, R2 ], ts = [ t ], aperture=aperture)

