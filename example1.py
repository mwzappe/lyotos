#!/usr/bin/env python3
import numpy as np

from lyotos import *

n2 = np.sqrt(2.53)

ff = FarField(GCS, Position.from_xyz(0,0,0), 1000)

s = System(far_field=ff)

Rexolite = Material(er=2.53)

l = SingletLens(GCS.newCS(Position.from_xyz(0,0,100)),
                material=Rexolite,
                R1=100,
                R2=100,
                t=50,
                aperture=20)

print(f"Lens Petzval Sum: {l.petzval_sum}")

s.add_element(l)

t = Tracer(s)

bundle = rays.create_fan(GCS, Position.CENTER, Vector.Z, 30 * np.pi / 180)

t.step_bundle(bundle)

h = l.intersect(bundle)

print(h.l)
print(h.p)
print(h.n)

hff = ff.intersect(bundle)

print(hff.l)
print(hff.p)
print(hff.n)

h = h.merge(hff)

print(h.l)
print(h.p)
print(h.n)
