from .quasirandom import R2

def discR2(n):
    u, v = R2(n)

    return np.sqrt(u), 2 * np.pi * v
