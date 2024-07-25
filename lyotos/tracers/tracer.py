from lyotos.geometry import GeometryObj
from lyotos.rays import Bundle

class Tracer:
    def __init__(self, system, debug=False):
        self._system = system
        self.debug = debug
        self._trace_count = 0
        
    def debug_null(self, msg):
        pass

    def debug_print(self, msg):
        print(msg)

    @property
    def debug(self):
        return False if self._debug == self.debug_null else True

    @debug.setter
    def debug(self, v):
        if v == True:
            self._debug = self.debug_print
        else:
            self._debug = self.debug_null

    @property
    def trace_count(self):
        return self._trace_count

    @property
    def system(self):
        return self._system

    @property
    def elements(self):
        return self.system.elements

    def step_bundle(self, bundle):
        self.system.far_field.intersect(bundle)
                
        for e in self.elements:
            e.intersect(bundle)

        for oid, hs in bundle.hits.hit_sets.items():
            GeometryObj.get(oid).trace(hs)

        
    
    def trace(self):
        pass
