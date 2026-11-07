#!/usr/bin/env python3
"""Add the missing slim full-width rear light bar seen in the reference.

The source tailgate has a vague horizontal ridge but no defined light bar.
We sample the existing rear surface, smooth it, and union a shallow rounded
strip that follows the tailgate curvature.  The strip stops either side of
the VW emblem so the badge remains the protruding central element.
"""
import argparse

import numpy as np
import trimesh
from scipy.signal import savgol_filter
from shapely.geometry import box


def rounded_rect(w, h, r=0.6):
    return box(-w / 2 + r, -h / 2 + r, w / 2 - r, h / 2 - r).buffer(r, resolution=4)


def rear_surface_x(mesh, z0=47.0):
    v = mesh.vertices
    ys = np.arange(-150, 151, 10)
    xs = []
    for y in ys:
        s = v[(np.abs(v[:, 1] - y) < 6) & (v[:, 2] > z0 - 3) &
              (v[:, 2] < z0 + 3) & (v[:, 0] < -400)]
        xs.append(np.percentile(s[:, 0], 15) if len(s) > 10 else np.nan)
    xs = np.array(xs, dtype=float)
    ok = ~np.isnan(xs)
    xs = np.interp(ys, ys[ok], xs[ok])
    return ys, savgol_filter(xs, 9, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    mesh = trimesh.load(args.src, process=True)
    comps = sorted(mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True)
    body, wheels = comps[0], comps[1:]
    ys, xs = rear_surface_x(body)
    parts = []
    for lo, hi in ((-150, -16), (16, 150)):
        sel = (ys >= lo) & (ys <= hi)
        yy = ys[sel]
        # Interpolate a denser path for a smooth sweep.
        ypath = np.linspace(lo, hi, 61)
        xpath = np.interp(ypath, yy, xs[sel]) - 0.7
        path = np.column_stack([xpath, ypath, np.full_like(ypath, 47.0)])
        # Section lies in the x-z plane; depth 2.2, height 3.2.
        bar = trimesh.creation.sweep_polygon(rounded_rect(2.2, 3.2, 0.7), path)
        parts.append(bar)
    body = trimesh.boolean.union([body] + parts, engine='manifold')
    out = trimesh.util.concatenate([body] + wheels)
    out.export(args.dst)
    print('watertight', out.is_watertight, 'components', len(out.split(only_watertight=False)),
          'faces', len(out.faces))


if __name__ == '__main__':
    main()
