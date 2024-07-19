import cupy as cp

class Material:
    def __init__(self, n=None, er=None):
        if n is None:
            if er is None:
                raise RuntimeException("n or er must be defined")

            if not callable(er):                
                n = lambda _: cp.sqrt(er)
            else:
                n = lambda nu: cp.sqrt(er(nu))

            self._n = n
        else:
            if er is not None:
                raise RuntimeException("Only one of er or n can be defined")

            if not callable(n):
                self._n = lambda nu: n
            else:
                self._n = n
            
    def n(self, nu):
        return self._n(nu)

Vacuum = Material(n=1)
