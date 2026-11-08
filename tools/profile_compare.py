#!/usr/bin/env python3
"""Top/bottom outline of the model against the side photo, station by station.

IoU says how far off the silhouette is overall; this says WHERE.  Both masks
are scaled to the same length and stood on the same ground line, then the
highest and lowest solid pixel is read at each X station and reported in model
units (length 900, ground Z=0).  Use it to retune the loft stations.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from check_similarity import ROOT, align, photo_mask, render_mask


def outline(mask):
    """Return (top, bottom) row index per column, -1 where the column is empty."""
    h = mask.shape[0]
    top = np.full(mask.shape[1], -1, int)
    bot = np.full(mask.shape[1], -1, int)
    for x in range(mask.shape[1]):
        ys = np.where(mask[:, x])[0]
        if len(ys):
            top[x], bot[x] = ys.min(), ys.max()
    return top, bot


def profiles(stl, photo, length=900.0, step=25.0):
    ref = photo_mask(os.path.join(ROOT, photo))
    mod = render_mask(os.path.join(ROOT, stl) if not os.path.isabs(stl) else stl)
    A, B = align(mod, ref)
    H, W = A.shape
    ta, ba = outline(A)
    tb, bb = outline(B)
    # column 0 = tail (-L/2), column W-1 = nose (+L/2); row H-1 = ground
    rows = []
    for X in np.arange(-length / 2, length / 2 + 1e-6, step):
        c = int(round((X / length + 0.5) * (W - 1)))
        c = min(max(c, 0), W - 1)
        z = lambda r: (H - 1 - r) * length / W if r >= 0 else None
        pt, pb = z(ta[c]), z(ba[c])
        mt, mb = z(tb[c]), z(bb[c])
        rows.append({
            'x': round(float(X), 1),
            'photo_top': None if pt is None else round(pt, 1),
            'model_top': None if mt is None else round(mt, 1),
            'd_top': None if (pt is None or mt is None) else round(mt - pt, 1),
            'photo_bottom': None if pb is None else round(pb, 1),
            'model_bottom': None if mb is None else round(mb, 1),
            'd_bottom': None if (pb is None or mb is None) else round(mb - pb, 1),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stl')
    ap.add_argument('--photo', default='1793820501.png')
    ap.add_argument('--step', type=float, default=25.0)
    ap.add_argument('--out')
    args = ap.parse_args()
    rows = profiles(args.stl, args.photo, step=args.step)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=2)
    print(f"{'X':>7} {'photoTop':>9} {'modelTop':>9} {'dTop':>7} "
          f"{'photoBot':>9} {'modelBot':>9} {'dBot':>7}")
    for r in rows:
        f = lambda v: '     -  ' if v is None else f'{v:8.1f}'
        print(f"{r['x']:7.1f} {f(r['photo_top'])} {f(r['model_top'])} {f(r['d_top'])} "
              f"{f(r['photo_bottom'])} {f(r['model_bottom'])} {f(r['d_bottom'])}")


if __name__ == '__main__':
    main()
