from .coordinate_system import Position

def CartesianOval:
    """
    The Cartesian oval is represented with points P at (0, 0) and Q at (0, qz)

    The implicit equation (S - P).norm + m * (S - Q).norm == a

    (S - Q).norm == (a - S.norm) / m

    Which is then the locus of intersection of circles
    
    """
    
    def __init__(self, a, m, qz):
        self._a = a
        self._m = m
        self._qz = qz

    @property
    def a(self):
        return self._a

    @property
    def m(self):
        return self._m

    @property
    def qz(self):
        return self._qz

    @property
    def P(self):
        return Position.CENTER

    @property
    def Q(self):
        return Position.from_xyz(0, 0, self.qz)

    def intersect(self, ray):
        
