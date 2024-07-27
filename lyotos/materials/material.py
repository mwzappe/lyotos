from lyotos.util import xp

class Material:
    def __init__(self, n=None, er=None):
        if n is None:
            if er is None:
                raise RuntimeException("n or er must be defined")

            if not callable(er):                
                n = lambda _: float(xp.sqrt(er))
            else:
                n = lambda nu: xp.sqrt(er(nu))

            self._n = n
        else:
            if er is not None:
                raise RuntimeException("Only one of er or n can be defined")

            if not callable(n):
                n = float(n)
                self._n = lambda nu: n
            else:
                self._n = n
            
    def n(self, nu=500):
        return self._n(nu)

Vacuum = Material(n=1)
