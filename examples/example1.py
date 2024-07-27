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

bundle = rays.create_fan(GCS, Position.CENTER, Vector.Z, 2 * cp.pi / 180, N=10)

print(bundle)

s.push_bundle(bundle)

import cupyx
from cupyx.profiler import benchmark

import time
start = time.perf_counter_ns()

with cupyx.profiler.profile():
    s.trace_loop()

end = time.perf_counter_ns()

print(f"Trace time: {(end-start)/1000000}ms")

if False:
    exit()

r = PVRenderer(s)

r.show()
