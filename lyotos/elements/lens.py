import cupy as cp

from lyotos.geometry import GeometryObj, CSM, Vector, Sphere
from lyotos.surfaces import SphericalSurface, CylinderSurface
from lyotos.physics import Interface
from lyotos.materials import Vacuum

from .shape import thin_radii, thick_radii
from .element import Element

class MultipletLens(Element):
    def __init__(self, cs, *, materials, Rs, ts, aperture, surroundings=Vacuum):
        super().__init__(cs)

        assert len(materials) + 1 == len(Rs)
        assert len(materials) == len(ts)
        
        cur_cs = cs
        self._surfaces = []
        self._materials = materials
        self._surroundings = Vacuum
        self._thicknesses = ts

        self._surfaces.append(SphericalSurface(cs=cur_cs,
                                               interaction=Interface(self.surroundings, self.materials[0]),
                                               R=Rs[0],
                                               aperture=aperture))

        last_m = self.materials[0]
        
        for R, t, m in zip(Rs[1:-1], ts[:-1], self.materials[1:-1]):
            cur_cs = cur_cs.xform(CSM.translate(t * Vector.Z))
            self._surfaces.append(SphericalSurface(cs=cur_cs,
                                                   interaction=Interface(last_m, m),
                                                   R=R,
                                                   aperture=aperture))
            last_m = m

        cur_cs = cur_cs.xform(CSM.translate(ts[-1] * Vector.Z))
        self._surfaces.append(SphericalSurface(cs=cur_cs,
                                               interaction=Interface(self.materials[-1], self.surroundings),
                                               R=Rs[-1],
                                               aperture=aperture))
            
        self._total_thickness = sum(ts)

        z1 = self.surfaces[0].edge_z
        z2 = self.surfaces[-1].edge_z
        edge_h = (self._total_thickness + z2 - z1).get()
        
        self._edge = CylinderSurface(cs=cs.xform(CSM.tZ(z1)),
                                     interaction=Interface(last_m, self.surroundings),
                                     R=aperture/2,
                                     h = edge_h)

        self._surface_idx = cp.array([ o.id for o in self._surfaces ])
        
    @property
    def surfaces(self):
        return self._surfaces    

    @property
    def materials(self):
        return self._materials

    @property
    def surroundings(self):
        return self._surroundings

    def P(self, nu=500):
        n1 = self.surroundings.n(nu)
        n2 = self.materials[0].n(nu)

        Pcur = (n2 - n1) / (n1 * self.surfaces[0].R)

        for s, t, m in zip(self.surfaces[:-1], self._thicknesses[:-1], self.materials):
            n1 = n2
            
            n2 = m.n(nu)
            P2 = (n2 - n1) / (n1 * s.R)
            
            Pcur = Pcur + P2 - t * Pcur * P2

        n1 = n2
        n2 = self.surroundings.n(nu)
        
        P2 = (n2 - n1) / (n1 * self.surfaces[-1].R)

        return Pcur + P2 - self._thicknesses[-1] * Pcur * P2
        
    def f(self, nu=500):
        return 1/self.P(nu)

    @property
    def q(self):
        for s1, s2 in zip(self.surfaces[:-1], self.surfaces[1:]):
            return (s1.R + s2.R)/(s1.R - s2.R)

    @property
    def boundary(self):
        return [ self.surfaces[0], self.surfaces[-1], self._edge ]
        
        
    def propagate(self, hit_set):
        bundles = []
        
        hit_sets = hit_set.pop_obj()

        for oid, hs in hit_sets.items():
            obj = GeometryObj.get(oid)
            print(f"Tracing rays for {obj}")

            
            if obj == self.surfaces[0]:
                m1 = self.surroundings
                m2 = self.materials[0]
            elif obj == self.surfaces[-1]:
                m2 = self.materials[-1]
                m1 = self.surroundings
            else:
                si = int(cp.argwhere(self._surface_idx == oid)[0])
                m1 = self.materials[si-1]
                m2 = self.materials[si]

            bundles += obj.interact(hs, m1, m2)

        print(f"Propagated bundles: {len(bundles)}")
            
        return bundles

    @property
    def petzval_sum(self, nu=None):
        outside_n = self.surroundings.n(nu) 
        last_n = outside_n
        retval = 0
        
        for s, m in zip(self.surfaces[:-1], self.materials):
            n = m.n(nu)
            retval += (n - last_n) / (s.R * n * last_n)
            last_n = n

        retval += (outside_n - last_n)/(self.surfaces[-1].R * outside_n * last_n)
        
        return retval

    def render(self, renderer):
        self._edge.render(renderer)

        for s in self.surfaces:
            s.render(renderer)

class SingletLens(MultipletLens):
    def __init__(self, cs, *, material, R1, R2, t, aperture):
        super().__init__(cs, materials = [ material ], Rs = [ R1, R2 ], ts = [ t ], aperture=aperture)


    @classmethod
    def max_aperture(cls, r1, r2, t):
        if r1 == 0:
            return np.abs(2*r2)
        if r2 == 0:
            return np.abs(2*r1)
        
        x2 = Sphere.intersect(t+r2-r1, r1, r2)**2
        
        if x2 > r1**2:
            return 2 * np.min(np.abs([r1, r2]))

        rp = np.sqrt(r1**2 - x2)
        
        return 2*rp    
    

        
    @classmethod
    def with_shape(cls, f, q, n, aper=None, t=1):
        # Start with thin lens approx, and solve for r1
        print(f"Creating lens with focal length {f}")

        r1, r2 = thick_radii(f, q, n, t)
        
        print(f"Thick lens: {r1} {r2}")
        
        return Lens(r1, r2, t, n, aper)
