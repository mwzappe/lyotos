from lyotos.util import xp

from numba import cuda

from lyotos.util import darray
from lyotos.geometry import SphPos, CSM
from .bundle import Bundle

def sph_to_cart(sph, cart):
    i = cuda.grid(1)

    if i < sph.shape[0]:
        s = xp.sin(sph[i,0])
        cart[i, 0] = s * xp.cos(sph[i, 1])
        cart[i, 1] = s * xp.sin(sph[i, 1])
        cart[i, 2] = xp.cos(sph[i, 0])
        
def create_fan(cs, pos, dir, theta, N=10):
    new_cs = cs.xform(CSM.align_z(dir) @ CSM.translate(pos))

    pos = xp.repeat(darray([ pos.v ]), N, axis=0)

    pts = SphPos.random_points(N - 1, theta)

    s = xp.sin(pts[:,0])

    dcos = darray([ s * xp.cos(pts[:,1]),
                    s * xp.sin(pts[:,1]),
                    xp.cos(pts[:,0]),
                    xp.zeros(N-1) ]).transpose()
    
    dcos = xp.vstack((darray([ [ 0, 0, 1, 0] ]),
                      dcos))

    return Bundle(pos, dcos, cs=cs)


    
