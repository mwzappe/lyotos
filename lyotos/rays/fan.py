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

    return Bundle(pos, dcos, cs=new_cs)


def create_column(cs, pos, dir, R, N=10, nu=535):
    new_cs = cs.xform(CSM.align_z(dir) @ CSM.translate(pos))

    r = R * xp.sqrt(xp.random.uniform(0, 1, size=N))
    theta = xp.random.uniform(0, 2*xp.pi, size=N)

    pos = xp.vstack(( (r * xp.cos(theta)),
                      (r * xp.sin(theta)),
                      xp.zeros(N),
                      xp.ones(N))).T

    dcos1 = darray([ [ 0.0, 0.0, 1.0, 0.0 ] ] )

    dcos = xp.repeat(dcos1, N).reshape(4, N).T

    return Bundle(pos, dcos, cs=new_cs, nu=nu)

def create_comb(cs, pos, dir, R, N=10, half=False):
    new_cs = cs.xform(CSM.align_z(dir) @ CSM.translate(pos))

    if half:
        l = xp.linspace(0, R, N)
    else:
        l = xp.linspace(-R, R, N)


    pos = xp.vstack((l,
                     xp.zeros(N),
                     xp.zeros(N),
                     xp.ones(N))).T
    
    dcos1 = darray([ [ 0.0, 0.0, 1.0, 0.0 ] ] )

    dcos = xp.repeat(dcos1, N).reshape(4, N).T

    return Bundle(pos, dcos, cs=new_cs)


def create_cross(cs, pos, dir, R, N=10):
    new_cs = cs.xform(CSM.align_z(dir) @ CSM.translate(pos))

    l = xp.linspace(-R, R, N)


    pos = xp.vstack((xp.hstack((l, xp.zeros(N))),
                     xp.hstack((xp.zeros(N), l)),
                     xp.zeros(2 * N),
                     xp.ones(2 * N))).T
    
    dcos1 = darray([ [ 0.0, 0.0, 1.0, 0.0 ] ] )

    dcos = xp.repeat(dcos1, 2 * N).reshape(4, 2 * N).T

    return Bundle(pos, dcos, cs=new_cs)
