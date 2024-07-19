
#
# r = e p / (1 + e cos(theta))
#
# Focus zero, dirextrix on x == p
#
# sqrt(x^2 + y^2) == e * (p - x)
# x^2 + y^2 = e^2 (p - x)^2
#
# y^2 + (1-e^2)x^2 + 2 e^2 p x - e^2 p^2 == 0
#
# A = 1-e^2
# B = 0
# C = 1
# D = 2 e^2 p
# E = 0
# F = -e^2 p^2
#
# 
#

def ConicSection:
    def __init__(self, e, p):
        self._p = p
        self._e = e

        self._a = 1 - self.e**2
        self._b = -2 * self.e**2 * self.p
        self._c = self.e**2 * self.p**2

    def y(self, x):
        v =  self._a * x**2 + self._b * x + self._c

        if v < 0:
            return []        
        
        if dsc < 0:
            return np.array()

        if dsc == 0:
            return np.array([ 0 ])

        v = np.sqrt(v)
        
        return [ v, -v ]
        
    def x(self, y):
        c = self._c - y**2
        
        dsc = self._b**2 - 4 * self._a * c

        if dsc < 0:
            return np.array()

        if dsc == 0:
            return np.array([-self._b / self._a / 2])
        
        dsc = np.sqrt(dsc)
        
        return np.array([-self._b + dsc, -self._b - dsc]) / self._a / 2
        
        
    @property
    def p(self):
        return self._p 

    @property
    def e(self):
        return self._e
        
    @property
    def l(self):
        return self.p * self.e

    @property
    def c(self):
        return self.a * self.e

    @property
    def a(self):
        return self.e * self.p / (1 - self.e**2)

    @property
    def A(self):
        return 1-self.e**2

    @property
    def B(self):
        return 
    
    @property
    def C(self):
        return 1

    @property
    def D(self):
        return 2 * self.e**2 * self.p

    @property
    def E(self):
        return 0
    
    @property
    def F(self):
        return -self.e**2 * self.p**2
    
    @property
    def homoM(self):
        return np.array([
            [ self.A, self.B/2, self.D/2 ],
            [ self.B/2, self.C, self.E/2 ],
            [ self.D/2, self.E/2, self.F ]
        ])
    
    
