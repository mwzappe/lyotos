#!/usr/bin/env python3
import numpy as np

from lyotos import *

n2 = np.sqrt(2.53)

s = SequentialSystem(aperture=CircularAperture(diameter=800))

aper = CircularAperture(diameter=100)

s.add_surface(200, 1, "flat", name="1", far_field=True)

s.add_surface(100, n2, "spherical", R=100, name="2", aperture=aper)
s.add_surface(100, 1, "spherical", R=-100, name="3", aperture=aper)

s.add_surface(0, 1, "flat", name="5", color="red", aperture=CircularAperture(diameter=1200))

t = Trace(s)

infinite_fan(t)
infinite_fan(t, theta=22.5 * np.pi/180)
infinite_fan(t, theta=45 * np.pi/180)

renderer = PVRenderer()

renderer.render_trace(t)

renderer.show()



