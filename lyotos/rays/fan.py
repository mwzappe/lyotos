import cupy as cp

from numba import cuda

from lyotos.util import darray
from lyotos.geometry import SphPos, CSM
from .bundle import RayBundle

def sph_to_cart(sph, cart):
    i = cuda.grid(1)

    if i < sph.shape[0]:
        s = cp.sin(sph[i,0])
        cart[i, 0] = s * cp.cos(sph[i, 1])
        cart[i, 1] = s * cp.sin(sph[i, 1])
        cart[i, 2] = cp.cos(sph[i, 0])
        
def create_fan(cs, pos, dir, theta, N=10):
    new_cs = cs.xform(CSM.align_z(dir) @ CSM.translate(pos))

    pos = cp.repeat(darray([ pos.v ]), N, axis=0)

    pts = SphPos.random_points(N - 1, theta)

    s = cp.sin(pts[:,0])

    dcos = darray([ s * cp.cos(pts[:,1]),
                    s * cp.sin(pts[:,1]),
                    cp.cos(pts[:,0]),
                    cp.zeros(N-1) ]).transpose()
    
    dcos = cp.vstack((darray([ [ 0, 0, 1, 0] ]),
                      dcos))

    return RayBundle(cs, pos, dcos)


    
