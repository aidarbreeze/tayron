#!/usr/bin/env python3
"""Overlay cross-sections of two models at given X stations.

IoU only judges the side outline.  Whether a flank is a slab or a sculpted
surface shows up in the cross-section, so this slices both meshes with a plane
normal to X and draws them on top of each other in model coordinates (ground
Z=0, centre Y=0).  The reference STL is centred on Z, so it is stood on the
ground first.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path, stand=True):
    m = trimesh.load(path)
    if stand:
        m.apply_translation([0.0, 0.0, -float(m.bounds[0][2])])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a', help='reference model (red)')
    ap.add_argument('b', help='model under test (blue)')
    ap.add_argument('--stations', type=float, nargs='+', default=[0.0, 283.4])
    ap.add_argument('--out', default='renders/sections.png')
    ap.add_argument('--label-a', default='reference')
    ap.add_argument('--label-b', default='from scratch')
    args = ap.parse_args()

    meshes = [(load(os.path.join(ROOT, args.a)), (210, 40, 40), args.label_a),
              (load(os.path.join(ROOT, args.b)), (40, 90, 220), args.label_b)]
    W, H = 900, 760
    im = Image.new('RGB', (W, H * len(args.stations)), (255, 255, 255))
    d = ImageDraw.Draw(im)
    sx = lambda y: int(W / 2 + y * 2.0)
    sy = lambda z: int(H - 40 - z * 1.9)
    for k, X in enumerate(args.stations):
        oy = k * H
        for z in range(0, 341, 50):
            d.line([0, oy + sy(z), W, oy + sy(z)], fill=(235, 235, 235))
            d.text((4, oy + sy(z) - 12), str(z), fill=(120, 120, 120))
        for mesh, col, _ in meshes:
            sec = mesh.section(plane_origin=[X, 0, 0], plane_normal=[1, 0, 0])
            if sec is None:
                continue
            for y, z in sec.vertices[:, 1:]:
                if z < -5 or z > 350:
                    continue
                d.ellipse([sx(y) - 1, oy + sy(z) - 1, sx(y) + 1, oy + sy(z) + 1], fill=col)
        d.text((10, oy + 10),
               f'x={X}   red={args.label_a}   blue={args.label_b}', fill=(0, 0, 0))
    out = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    im.save(out)
    print('saved', out)


if __name__ == '__main__':
    main()
