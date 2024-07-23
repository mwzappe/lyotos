#!/usr/bin/env python3
import cupy as cp
import time

from lyotos import *

n2 = cp.sqrt(2.53)

ff = FarField(GCS, Position.from_xyz(0,0,0), 200)

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

start = time.perf_counter_ns()
bundle = rays.create_fan(GCS, Position.CENTER, Vector.Z, 2 * cp.pi / 180, N=10000)
end = time.perf_counter_ns()

print(f"{end-start}ns")

h = t.step_bundle(bundle)

r = PVRenderer()

h.render(r)

l.render(r)

r.show()
