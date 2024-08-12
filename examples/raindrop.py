#!/usr/bin/env python3
import time

from lyotos import *
from lyotos.util import xp, darray, iarray


ff = FarField(GCS, Position.from_xyz(0,0,0), 20)

s = System(far_field=ff)

droplet_R = 0.5

l = SingletLens(GCS.newCS(Position.CENTER),
                material=Water,
                R1=droplet_R,
                R2=-droplet_R,
                t=2*droplet_R,
                aperture=2*droplet_R)

s.add_element(l)

wavelengths = xp.linspace(405, 640, 11)

bundles = [ rays.create_column(GCS, Position.CENTER, Vector.Z, droplet_R * 0.99999, N=100000, nu=nu) for nu in wavelengths ]

for bundle in bundles:
    s.push_bundle(bundle)

#bundle = rays.create_comb(GCS, Position.from_xyz(0,0,-10), Vector.Z, droplet_R * 0.99999, N=101)

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

#r = PVRenderer(s)

#r.show()

import healpy as hp
import pyvista as pv


nside = 256
npix = hp.nside2npix(nside)
print(f"npix: {npix}")

colors = xp.zeros((hp.nside2npix(nside), 4))

for hs in ff._hits:
    pos = hs.bundle.GCS.directions[:,:3]
    amp = hs.bundle.amplitudes[:,0]
    
    pix = hp.vec2pix(nside, pos[:,0], pos[:,1], pos[:,2])

    rgb = WavelengthToRGB.as_array(hs.nu)

    rs = rgb[0] * amp
    gs = rgb[1] * amp
    bs = rgb[2] * amp
    
    #print(f"pix: {pix}")
    
    for p, r, g, b, a in zip(pix, rs, gs, bs, amp):
        colors[p, 0] += r
        colors[p, 1] += g
        colors[p, 2] += b
        colors[p, 3] += a

max_intensity = xp.log10(xp.max(colors[:,3]))
min_intensity = xp.log10(xp.min(colors[:,3][colors[:,3]>0]))

print(f"Intensity: max: {max_intensity} min: {min_intensity}")
colors[:,3][colors[:,3] == 0] = 1e-6

colors[:,3] = (xp.log10(colors[:,3]) - min_intensity)/(max_intensity - min_intensity)
colors[:,:3] /= colors[:,:3].max(axis=1)[:,xp.newaxis]

#print(colors)

plotter = pv.Plotter()
plotter.background_color="#606060"

boundaries = hp.boundaries(nside, xp.arange(npix))

boundaries = darray([
    boundaries[:,0,:].flatten(),
    boundaries[:,1,:].flatten(),
    boundaries[:,2,:].flatten()
]).T

print(boundaries)

#neighbors = hp.get_all_neighbours(nside, [ i for i in range(npix) ])

#faces = iarray([ xp.ones(npix) * 3, xp.arange(npix), neighbors[3,:], neighbors[5,:] ]).T

#print(faces)

faces = [ [ 4, 4*i, 4*i+1, 4*i+2, 4*i+3 ] for i in range(npix) ]

cloud = pv.PolyData(boundaries, faces=faces)

cloud["color_rgba"] = colors[:,:3]

plotter.add_mesh(cloud)

plotter.show()
