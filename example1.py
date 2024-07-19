#!/usr/bin/env python3
import numpy as np

from lyotos import *

n2 = np.sqrt(2.53)

ff = FarField(GCS, Position.from_xyz(0,0,0), 1000)

s = System(far_field=ff)

l = SingletLens(GCS.newCS(Position.from_xyz(0,0,100)),
                n=n2,
                R1=100,
                R2=100,
                t=50,
                aperture=20)

s.add_element(l)

t = Tracer(s)


bundle = RayBundle.from_rays([ Ray(l.cs, Position.from_xyz(x, x, x), Vector.Z) for x in range(20) ])
