import cupy as cp

class _BundleBase:
    def __init__(self, positions, directions):
        assert len(positions.shape) == 2, "Must be Nx4 array"
        assert len(directions.shape) == 2, "Must be Nx4 array"
        assert positions.shape == directions.shape, f"Position ({position.shape}) and direction ({directions.shape}) shapes do not match"
        assert positions.shape[1] == 4, f"Improper number of coordinates for positions and directions"

        self._positions = positions
        self._directions = directions


    @property
    def l(self):
        return self._l

    @property
    def p(self):
        return self._p

    @property
    def n(self):
        return self._n
        
    @property
    def nu(self):
        return 500
        
    @property
    def positions(self):
        return self._positions
    
    @property
    def directions(self):
        return self._directions

    def pts_at(self, ls):
        return self.positions + cp.einsum("i,ij->ij", ls, self.directions)

    def __len__(self):
        return self.positions.shape[0]

    def __eq__(self, other):
        return self.bundle_id == other.bundle_id
