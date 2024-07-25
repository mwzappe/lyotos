#!/usr/bin/env python3
import cupy as cp
import cupyx

import time

from lyotos import *

n2 = cp.sqrt(2.53)

ff = FarField(GCS, Position.from_xyz(0,0,0), 200)

s = System(far_field=ff)

Rexolite = Material(er=2.53)

l = SingletLens(GCS.newCS(Position.from_xyz(0,0,100)),
                material=Rexolite,
                R1=25,
                R2=-25,
                t=5,
                aperture=20)

print(f"Lens focal length: {l.f()}")

print(f"Lens Petzval Sum: {l.petzval_sum}")

s.add_element(l)

bundle = rays.create_fan(GCS, Position.CENTER, Vector.Z, 2 * cp.pi / 180, N=20)

s.push_bundle(bundle)

s.trace_loop()

r = PVRenderer(s)

r.show()
