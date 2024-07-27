import os

if (('LYOTOS_USE_GPU' in os.environ) and 
    (os.environ['LYOTOS_USE_GPU'] != '1')):
    import numpy as xp

    xp.get = lambda x: x
    use_gpu = False
else:
    import cupy as xp 
    xp.get = lambda x: x.get()
    use_gpu = True

    #from cupyx import jit

    from numba import cuda

    print(f"Using CUDA: Compute Capability {xp.cuda.Device(0).compute_capability}")
    
_xp = xp
    
float_type = xp.float64

    
def darray(o):
    try:
        return xp.array(o, float_type)
    except Exception as e:
        print(f"Failed to create array from {o}")
        raise e
    
def iarray(o):
    return xp.array(o, int)


MISS = xp.finfo(float_type).max

if use_gpu:
    @cuda.jit(cache=True)
    def _take_lowest_l_p_2(l, p, ls, p0, p1):
        i = cuda.grid(1)

        if i < l.shape[0]:
            if ls[i, 0] < ls[i, 1]:
                l[i] = ls[i, 0]
                p[i,0] = p0[i,0]
                p[i,1] = p0[i,1]
                p[i,2] = p0[i,2]
                p[i,3] = p0[i,3]
            else:
                l[i] = ls[i, 1]
                p[i,0] = p1[i,0]
                p[i,1] = p1[i,1]
                p[i,2] = p1[i,2]
                p[i,3] = p1[i,3]

    def take_lowest_l_p_2(l, p, ls, p0, p1):
        tpb = 128
        bpg = -(-l.shape[0] // tpb)
        _take_lowest_l_p_2[tpb, bpg](l, p, ls, p0, p1)
else:
    def take_lowest_l_p_2(l, p, ls, p0, p1):
        li = xp.argmin(ls, 1)

        p[li == 0] = p0[li == 0]
        p[li == 1] = p1[li == 1]
        l[:] = xp.take_along_axis(ls, li[:,xp.newaxis], 1).flatten()

if use_gpu:
    @cuda.jit(cache=True)
    def _batch_dot(out, vs1, vs2):
        i = cuda.grid(1)
        
        if i < out.shape[0]:
            out[i] = (vs1[i, 0] * vs2[i, 0] +
                      vs1[i, 1] * vs2[i, 1] +
                      vs1[i, 2] * vs2[i, 2])

    def batch_dot(out, vs1, vs2):
        tpb = 128
        bpg = -(-out.shape[0] // tpb)
        _batch_dot[bpg, tpb](out, vs1, vs2)
else:
    def batch_dot(out, vs1, vs2):
        return xp.einsum("ij,ij->i",vs1,vs2, out=out)
