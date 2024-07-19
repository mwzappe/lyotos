#!/usr/bin/env python3
import numpy as np
from scipy.optimize import minimize_scalar

from lyotos import *

n2 = np.sqrt(2.53)

def build(bfl):
    s = SequentialSystem(aperture=800, far_field_radius=1000)

    #s.add_surface(300, 1, "flat", name="1", far_field=True, aperture=400)

    if False:
        s.add_surface(400, n2, "spherical", R=200, name="hemisphere1", aperture=400)
        s.add_surface(0, n2, "spherical", R=-200, name="hemisphere1", aperture=400)

        t = Trace(s)
        return t
    
    
    f1 = -400
    f2 = 500

    t1 = 10

    d = f1 + f2
    
    l1 = Lens.with_shape(f1, 1.5, n2, t=t1)
    l2 = Lens.with_shape(f2, -1.5, n2, t=100)

    print(f"Lens1: R1: {l1.R1} R2: {l1.R2} t: {l1.t}")
    print(f"Lens2: R1: {l2.R1} R2: {l2.R2} t: {l2.t}")

    print(f"Lens distance: {d}")

    if False:
        s.add_surface(l1.t, l1.n, "spherical", R=l1.R1, name="l1-1", aperture=l1.max_aperture)
        s.add_surface(d, 1, "spherical", R=l1.R2, name="l1-2", aperture=l1.max_aperture)
    
        s.add_surface(l2.t, l2.n, "spherical", R=l2.R1, name="l2-1", aperture=l2.max_aperture)
        s.add_surface(0, 1, "spherical", R=l2.R2, name="l2-2", aperture=l2.max_aperture)

    s.add_surface(bfl, n2, "spherical", R=119, name="2", aperture=119*2)
    
    s.add_surface(0, 1, "flat", name="image", aperture=119*2, absorber=True)
    
    t = Trace(s)

    return t

def trial_for_angle(theta):
    @np.vectorize
    def trial(bfl):
        t = build(bfl)
    
        if True:
            f = InfiniteFan(t, theta=theta, color="red", D=380)

            return f.spot_size(t.system.surfaces[2])

    return trial
        #infinite_fan(t, theta=22.5 * np.pi/180, color="green")
        #infinite_fan(t, theta=45 * np.pi/180, color="blue")


        
def solve_minima(f, initial):
    l = initial

    delta = 0.004

    for i in range(10):
        ls = l + np.array([ -2 * delta, -delta, 0, delta, 2 * delta ])

        v = f(ls)

        print(f"{ls}->{v}")

        if np.abs(v[2]) < 1e-7:
            break

        if v[2] == min(v):
            break
            
        dv = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / 12 / delta

        print(f"dv: {dv} -v[2]/dv: {-v[2]/dv}")
        
        l += 0.8 * (-v[2] / dv)

    return l
         
cur_bfl = 200

if False:
    for theta in np.linspace(0, 35, 8) * np.pi / 180:
        f = trial_for_angle(theta)
        print(f"xa: {f(200)} xb: {f(400)} xc: {f(600)}")
    
        l = minimize_scalar(f, bracket=(200, 400, 600)) #solve_minima(trial, 200)
        
        print(l.x)

def hsv2rgb(h, s, v):
    c = v*s
    x = c * (1 - np.abs((h/60)%2 - 1))
    m = v - c

    if h < 60:
        r, g, b = (c, x, 0)
    elif h < 120:
        r, g, b = (x, c, 0)
    elif h < 180:
        r, g, b = (0, c, x)
    elif h < 240:
        r, g, b = (0, x, c)
    elif h < 300:
        r, g, b = (x, 0, c)
    else:
        r, g, b = (c, 0, x)

    r += m
    g += m
    b += m

    s = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

    print(f"{s}: {r} {g} {b}")
    
    return s
        
t = build(245)

#t._target = t.system.surfaces[-1]

total_thickness = t.system.surfaces[-1].cs.toGCS._M[2][3]

print(f"total_thickness: {total_thickness}")

p = Position.from_xyz(0, 0, total_thickness)

t.system.far_field.center = p


r = Radiator(t,
             p,
             Vector.from_xyz(0, 0, -1),
             10 * np.pi/180,
             target=t.system.surfaces[0])
             
        
renderer = PVRenderer()

renderer.render_trace(t)

import pyvista as pv

ap = np.array(r.all_paths)

test_tube = RayTube(*ap[r.triangles[0]])

allpts = np.array([ p.final_point.v3 if p is not None else None for p in r.all_paths ])

faces = np.hstack((np.array([ 3 for _ in r.triangles ]).reshape(r.triangles.shape[0], 1), r.triangles))
        
pd = pv.PolyData(allpts, faces=faces)

pd.cell_data["gains"] = 10 * np.log10(r.gains)

renderer.add_mesh(pd, scalars="gains", opacity=1)

print(r.gains)

renderer.show()



