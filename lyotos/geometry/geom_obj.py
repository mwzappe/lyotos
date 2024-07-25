from lyotos.util import classproperty

class GeometryObj:
    _objects = []

    def __init__(self, cs):
        self._cs = cs
        
        self._N = len(self._objects)
        self._objects.append(self)

    @property
    def cs(self):
        return self._cs

    @property
    def id(self):
        return self._N
    
    def __hash__(self):
        return self._N

    def __eq__(self, other):
        return self._N == other._N

    def __repr__(self):
        return f"GO({self._N}): {self.__class__}"
    
    @classmethod
    def get(cls, N):
        return cls._objects[N]
    
    @classproperty
    def objects(self):
        return self._objects
