#!/usr/bin/env python3
import numpy as np

from lyotos import *

n2 = np.sqrt(2.53)

s = SequentialSystem(aperture=CircularAperture(diameter=800))


        

class ReverseTelephoto:
    def __init__(self, FL, BFL):
        self._FL = FL
        self._BFL = BFL

        self._D = (FL - BFL)**2/FL

    @property
    def BFL(self):
        return self._BFL

    @property
    def FL(self):
        return self._FL

    @property
    def D(self):
        return self._D
    
    @property
    def FA(self):
        return self.D * self.FL / (self.FL - self.BFL)

    @property
    def FB(self):
        return -self.D * self.BFL / (self.FL - self.BFL - self.D)


    

rt = ReverseTelephoto(200, 300)

print(f"BFL: {rt.BFL} FL: {rt.FL} D: {rt.D} FA: {rt.FA} FB: {rt.FB}")


l = Lens(800, -800, 100, n2, CircularAperture(diameter=800))

l1 = Lens.with_shape(rt.FA, 2.0, n2, t=10)
l2 = Lens.with_shape(rt.FB, 0.0, n2, t=30)

a = np.min([ l1.max_aperture, l2.max_aperture ])

print(f"Aperture: {a}")

aper = CircularAperture(diameter=2*a)

print(f"Petzval sum: {l.petzval_sum}")


        
print(f"{l.P} {l.f}")

s.add_surface(200, 1, "flat", name="1", far_field=True)

s.add_surface(l1.t, l1.n, "spherical", R=l1.R1, name="2", aperture=aper)

aper = CircularAperture(diameter=2*l1.R2)

s.add_surface(rt.D - l1.h2 + l2.h1, 1, "spherical", R=l1.R2, name="3", aperture=aper)


aper = CircularAperture(diameter=l2.max_aperture)

s.add_surface(l2.t, l2.n, "spherical", R=l2.R1, name="4", aperture=aper)
s.add_surface(rt.BFL, 1, "spherical", R=l2.R2, name="5", aperture=aper)

s.add_surface(200, 1, "flat", name="6", color="green")
s.add_surface(0, 1, "flat", name="7", color="red", aperture=CircularAperture(diameter=1200))

t = Trace(s)

if False:
    infinite_fan(t, color="red")
    infinite_fan(t, theta=22.5 * np.pi/180, color="green")
    infinite_fan(t, theta=45 * np.pi/180, color="blue")


pos = s.surfaces[-2].cs.toGCS @ Position.from_xyz(0, 0, 0)

print(f"pos: {pos}")

illuminate(t, pos)

renderer = PVRenderer()

renderer.render_trace(t)

renderer.show()



