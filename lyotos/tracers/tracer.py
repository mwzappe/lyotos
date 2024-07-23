

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
        h = self.system.far_field.intersect(bundle)
                
        for e in self.elements:
            hp = e.intersect(bundle)
            h = h.merge(hp)

        return h
            
    
    def trace_bundle(self, bundle):
        pass
