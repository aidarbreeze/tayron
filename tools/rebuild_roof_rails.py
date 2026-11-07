#!/usr/bin/env python3
"""Replace the thick integrated roof-rail pads with slim photo-matched arches.

The source roof rails are broad, flat pads.  The reference car has narrow
raised arches with visible feet and a gap under the centre.  This script
flattens the old pads back toward the local roof surface, sweeps two new
rounded rectangular rails along the roof line, adds end feet, and unions
them into the body.  The four wheel shells are preserved.
"""
import argparse
import numpy as np
import trimesh
from shapely.geometry import Polygon
from scipy.interpolate import interp1d


def roof_base_curve(body):
    v = body.vertices
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ay = np.abs(y)
    xs, zs = [], []
    for x0 in range(-280, 80, 10):
        sel = (x > x0) & (x < x0 + 10) & (ay > 82) & (ay < 98) & (z > 138)
        if sel.sum() > 5:
            xs.append(x0 + 5)
            zs.append(np.median(z[sel]))
    return interp1d(xs, zs, kind='cubic', fill_value='extrapolate')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    mesh = trimesh.load(args.src, process=True)
    comps = sorted(mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True)
    body, wheels = comps[0], comps[1:]
    assert len(wheels) == 4
    f = roof_base_curve(body)
    V = body.vertices.copy()
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ay = np.abs(y)
    base = f(x)
    excess = z - base
    rail = (x > -268) & (x < 68) & (ay > 98) & (ay < 140) & (z > 140)
    w = np.clip((excess - 1.5) / 6.0, 0, 1) * rail
    V[:, 2] = z - w * (excess - 1.0)
    body.vertices = V

    new_parts = []
    for side in (-1, 1):
        path = []
        for xx in np.linspace(-250, 50, 61):
            zz = f(xx) + 6.0 + 4.0 * np.sin(np.pi * (xx + 250) / 300) ** 0.7
            path.append([xx, side * 112.0, zz])
        path = np.array(path)
        poly = Polygon([(-4, -2.5), (4, -2.5), (4, 2.5), (-4, 2.5)])
        rail_solid = trimesh.creation.sweep_polygon(poly, path)
        for xx in (-245, 45):
            foot = trimesh.creation.box(extents=[18, 10, 6])
            foot.apply_translation([xx, side * 112, f(xx) + 2])
            rail_solid = trimesh.boolean.union([rail_solid, foot], engine='manifold')
        new_parts.append(rail_solid)
    body = trimesh.boolean.union([body] + new_parts, engine='manifold')
    out = trimesh.util.concatenate([body] + wheels)
    out.export(args.dst)
    print('watertight', out.is_watertight, 'components', len(out.split(only_watertight=False)), 'faces', len(out.faces))


if __name__ == '__main__':
    main()
