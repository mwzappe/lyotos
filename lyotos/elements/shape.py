# q = (r1 + r2)/(r1 - r2)
# q == 1 -> r2 == inf
# q == -1 -> r1 == inf
#
# r1 = (q+1)*r2/(q-1)
# r2 = (q-1)*r1/(q+1)
#
# 1/f = (n-1)*(1/r1-1/r2)
# r1 -> inf  r2 = -(n-1) f
# r2 -> inf  r1 = (n-1) f

def thin_radii(f, q, n):
    if q == -1:
        return 0,f * (1-n)
    elif q==1:
        return - f * (1-n), 0
        
    r1 = 2*f*(1-n)/(q-1)

    return r1, (q-1)*r1/(q+1)

# 1/f = (n-1)*(1/r1-1/r2+(n-1)*d/(n r1 r2))

def thick_radii(f, q, n, t):
    if q == 1:
        return 0,f * (1-n)
    elif q == -1:
        return -f * (1-n), 0

    d = -f * t * (n-1)**2 * (q + 1)
    b = 2 * f * n * (1 - n)
    
    dsc = (b**2  + -4 * (q-1) * n * d )

    v = np.sqrt(dsc)

    s1 = (b + v) / 2 / (n*q-n)
    s2 = (b - v) / 2 / (n*q-n)

    if np.abs(s1) > np.abs(s2):
        r1 = s1
    else:
        r1 = s2
    
    return r1, (q-1)*r1/(q+1)
