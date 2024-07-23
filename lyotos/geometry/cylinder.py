

class Cylinder:
    @classmethod
    def intersect(cls, R, bundle):
        # solve (p.x + l * d.x)**2 + (p.y + l * d.y)**2 == R**2

        px, py = bundle[:,0:2]
        dx, dy = bundle[:,4:6]

        a = dx**2 + dy**2
        b = 2 * (px * dx + py * dy)
        c = px**2 + py**2 - R**2

        print(a, b, c)
