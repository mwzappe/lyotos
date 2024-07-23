import cupy as cp


def darray(o):
    return cp.array(o, float)

def iarray(o):
    return cp.array(o, int)


MISS = cp.finfo(cp.float32).max

def take_lowest_l_p_2(l, p0, p1):
    l[cp.isnan(l)] = 1e11
            
    li = cp.argmin(l, 1)

    p = cp.einsum("i,ij->ij", (1 - li), p0) + cp.einsum("i,ij->ij", li, p1)
    l = cp.take_along_axis(l, li[:,cp.newaxis], 1).flatten()
    
    l[l == 1e11] = cp.nan

    return l, p

def batch_dot(vs1, vs2):
    return cp.einsum("ij,ij->i",vs1,vs2)

def matrix_mult_nvecs(M, vs):
    return cp.einsum("jk,ik->ij", M, vs)
