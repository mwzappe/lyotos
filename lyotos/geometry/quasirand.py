from lyotos.util import xp

_g = 1.32471795724474602596
a1 = 1.0/_g
a2 = 1.0/(_g*_g)

@xp.vectorize
def R2(n):    
    return (0.5+a1*n) %1, (0.5+a2*n) %1
