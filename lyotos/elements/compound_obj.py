from lyotos.geometry import GeometryObj

hit_count_limit = 15

class CompoundObj(GeometryObj):
    def __init__(self, cs):
        super().__init__(cs)

        self._bundle_queue = []

    def push_bundle(self, bundle):
        self._bundle_queue.append(bundle)

    def push_bundles(self, bundles):
        self._bundle_queue += bundles
        
    def pop_bundle(self):
        return self._bundle_queue.pop(0)

    @property
    def pending_bundles(self):
        return len(self._bundle_queue) > 0
    
    def trace_loop(self):
        loop_count = 0
        
        while self.pending_bundles:
            bundle = self.pop_bundle()

            if bundle.hit_count > hit_count_limit:
                continue
            
            loop_count += 1

            # Test for hitting boundary
            for c in self.children:
                c.test_hit(bundle)

            # extract escaping bundles
                
            for oid, hs in bundle.hits.hit_sets.items():
                obj = GeometryObj.get(oid)
                bundles = obj.propagate(hs)
                self.push_bundles(bundles)

            if loop_count > 400:
                print("Loop break")
                break
                            

    def render(self, renderer):
        for c in self.children:
            c.render(renderer)
                
    def propagate(self, hit_set):
        raise RuntimeError(f"Propagate not defined for class {self.__class__.__name__}")
