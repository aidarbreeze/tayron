#!/usr/bin/env python3
"""Replace the simplified 6-spoke wheels with photo-matched 10-spoke wheels.

The reference photos show a 10-blade alloy wheel (≈72 % rim/tire diameter
ratio).  The source STL has generic 6-spoke wheels.  This script rebuilds the
four wheel shells at the exact source centres / diameters / widths, unions
them into closed solids, and swaps them into the corrected body mesh.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon


def make_wheel(radius=72.99, width=50.47, rim_radius=52.0, sections=128):
    hw = width / 2
    # Tire: closed revolved profile, rounded shoulders.
    profile = np.array([
        [rim_radius - 4, -hw], [rim_radius, -hw],
        [radius * 0.90, -hw * 0.90], [radius * 0.975, -hw * 0.62],
        [radius, -hw * 0.30], [radius, hw * 0.30],
        [radius * 0.975, hw * 0.62], [radius * 0.90, hw * 0.90],
        [rim_radius, hw], [rim_radius - 4, hw],
        [rim_radius - 4, hw * 0.62], [rim_radius - 2, 0],
        [rim_radius - 4, -hw * 0.62], [rim_radius - 4, -hw],
    ], dtype=float)
    tire = trimesh.creation.revolve(profile, sections=sections)

    # Rim barrel + outer lip.
    barrel = trimesh.creation.annulus(r_min=rim_radius - 5, r_max=rim_radius,
                                      height=width * 0.72, sections=sections)
    lip = trimesh.creation.annulus(r_min=rim_radius - 2.5, r_max=rim_radius + 0.6,
                                   height=3.0, sections=sections)
    lip.apply_translation([0, 0, hw * 0.72])

    hub = trimesh.creation.cylinder(radius=13.0, height=10.0, sections=64)
    hub.apply_translation([0, 0, hw * 0.62])
    cap = trimesh.creation.cylinder(radius=8.0, height=2.5, sections=64)
    cap.apply_translation([0, 0, hw * 0.62 + 5.0])

    # Recessed brake disc behind the spokes so the wheel does not read as a
    # set of white cut-outs in clay previews.
    disc = trimesh.creation.cylinder(radius=rim_radius - 4.0, height=3.0, sections=96)
    disc.apply_translation([0, 0, hw * 0.25])
    caliper = trimesh.creation.box(extents=[16, 26, 8])
    caliper.apply_translation([rim_radius * 0.45, 0, hw * 0.25])
    parts = [tire, barrel, lip, hub, cap, disc, caliper]
    # Reference wheel: five paired / split spokes (10 blades total).
    for i in range(5):
        base = np.radians(i * 72.0 + 18.0)
        for side in (-1, 1):
            root = base + side * np.radians(4.0)
            tip = base + side * np.radians(17.0)
            r0, r1 = 11.0, rim_radius - 1.0
            w0, w1 = np.radians(4.5), np.radians(7.5)
            pts = []
            for r, ang, hw_ang in [(r0, root, w0), (r1, tip, w1)]:
                for s in (-1, 1):
                    a = ang + s * hw_ang
                    pts.append((r * np.cos(a), r * np.sin(a)))
            poly = Polygon([pts[0], pts[1], pts[3], pts[2]])
            spoke = trimesh.creation.extrude_polygon(poly, height=8.0)
            spoke.apply_translation([0, 0, hw * 0.62 - 4.0])
            parts.append(spoke)

    wheel = trimesh.boolean.union(parts, engine='manifold')
    wheel.fix_normals()
    return wheel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    mesh = trimesh.load(args.src, process=True)
    comps = sorted(mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True)
    body = comps[0]
    wheels = comps[1:]
    assert len(wheels) == 4, 'expected four wheel shells'
    new_wheels = []
    for w in wheels:
        lo, hi = w.bounds
        center = (lo + hi) / 2
        radius = (hi[2] - lo[2]) / 2
        width = hi[1] - lo[1]
        wheel = make_wheel(radius=radius, width=width)
        # Local wheel axis is +Z with outer face at +Z.  Map to world ±Y.
        if center[1] < 0:
            T = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        else:
            T = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        wheel.apply_transform(T)
        wheel.apply_translation(center)
        new_wheels.append(wheel)
        print('wheel', center, 'r', radius, 'w', width, 'watertight', wheel.is_watertight)
    out = trimesh.util.concatenate([body] + new_wheels)
    out.export(args.dst)
    print('saved', args.dst, 'watertight', out.is_watertight, 'components', len(out.split(only_watertight=False)))


if __name__ == '__main__':
    main()
