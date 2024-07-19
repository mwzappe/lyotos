
class Caster:
    def _maybe_append_path(self, path):
        if (path is not None) and ((self._target is None) or path.hits_surface(self._target)):
            self._paths.append(path)

    
