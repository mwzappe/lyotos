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

def take_lowest_l_p_2(l, p0, p1):
    l[xp.isnan(l)] = 1e11
            
    li = xp.argmin(l, 1)

    p = xp.einsum("i,ij->ij", (1 - li), p0) + xp.einsum("i,ij->ij", li, p1)
    l = xp.take_along_axis(l, li[:,xp.newaxis], 1).flatten()
    
    l[l == 1e11] = xp.nan

    return l, p

if use_gpu:
    @cuda.jit
    def _batch_dot(out, vs1, vs2):
        i = cuda.grid(1)
        
        if i < out.shape[0]:
            out[i] = (vs1[i, 0] * vs2[i, 0] +
                      vs1[i, 1] * vs2[i, 1] +
                      vs1[i, 2] * vs2[i, 2])

    def batch_dot(out, vs1, vs2):
        tpb = 16
        bpg = -(-out.shape[0] // tpb)
        _batch_dot[bpg, tpb](out, vs1, vs2)
else:
    def batch_dot(out, vs1, vs2):
        return xp.einsum("ij,ij->i",vs1,vs2, out=out)
