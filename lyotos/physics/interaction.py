class Interaction:
    def __init__(self):
        pass
        
    def interact(self, obj, hit_set):
        raise RuntimeError("Interact is not defined for class {self.__class__}")
    
