from .surface import Surface

class SagSurface(Surface):
    surf_name="base"

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)
    
    def __init__(self, cs, **kwargs):
        super().__init__(cs, **kwargs)
    
    def sag(self, x, y):
        raise RuntimeException(f"Sag function not implemented for class {self.__class__}")
        return 0 * x + 0 * y

    def normal(self, x, y):
        delta = 0.004

        da = np.array([ -2 * delta, -delta, 0, delta, 2 * delta ])

        sagx = self.sag(x + da, y)
        sagy = self.sag(x, y + da)
        
        dzdx = (-sagx[0] + 8 * sagx[1] - 8 * sagx[3] + sagx[4]) / 12 / delta
        dzdy = (-sagy[0] + 8 * sagy[1] - 8 * sagy[3] + sagy[4]) / 12 / delta

        norm = np.array([dzdx, dzdy, 1, 0 ])
        norm /= np.linalg.norm(norm)

        return Vector(norm)
    
    
    def do_intersect(self, ray):
        delta = 0.004
        
        # p + l * dcos == (x, y, f(x, y))

        # Solve 
        # 0 == f(p[0] + l * dcos[0], p[1] + l * dcos[1]) - p[2] - l * dcos[2]

        l = ray.l_at_z0 # -p[2]/dcos[2]
        
        for i in range(10):
            ls = l + np.array([ -2 * delta, -delta, 0, delta, 2 * delta ])

            ps = ray.at(ls)

            v = np.array([ self.sag(p.x, p.y) - p.z for p in ps ])

            if np.abs(v[2]) < 1e-7:
                break
            
            dv = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / 12 / delta

            l += -v[2] / dv

        p = ray.at(l)

        n = self.normal(p.x, p.y)
            
        return l, p, n
    

    def __repr__(self):
        return f"Surface {self._name} {self.__class__}"
