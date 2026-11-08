#!/usr/bin/env python3
"""Where the side silhouette disagrees with the photo, in model units.

`check_similarity.py` answers *how close*; this answers *what to change*.  It
splits the IoU error into zones and prints the top/bottom outline residual on
a 25-unit X grid, so a roof that is 6 units too tall is visible as a number
rather than a blue smudge on an overlay.

    python tools/silhouette_report.py 'tayron from scratch.stl' \
        --json reports/silhouette.json --overlay renders/silhouette.png
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_similarity as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def outline(mask, which):
    w = mask.shape[1]
    out = np.full(w, np.nan)
    for c in range(w):
        rows = np.where(mask[:, c])[0]
        if len(rows):
            out[c] = rows.min() if which == 'top' else rows.max()
    return out


def report(stl, photo):
    ref = C.photo_mask(photo)
    A, B = C.align(C.render_mask(stl), ref)
    h, w = ref.shape
    S = 900.0 / w                       # model units per pixel
    yy, xx = np.mgrid[0:A.shape[0], 0:w]
    X = (xx / (w - 1) - 0.5) * 900.0
    Z = (h - 1 - yy) * S
    miss, ext = A & ~B, B & ~A
    zones = [
        ('rear wheel', (Z < 75) & (X < -180)),
        ('front wheel', (Z < 75) & (X > 180)),
        ('sill', (Z < 75) & (np.abs(X) <= 180)),
        ('tail', (X < -390) & (Z >= 75)),
        ('nose', (X > 390) & (Z >= 75)),
        ('roof/rail', (Z > 300) & (np.abs(X) <= 390)),
        ('greenhouse', (Z > 218) & (Z <= 300) & (np.abs(X) <= 390)),
        ('body', (Z >= 75) & (Z <= 218) & (np.abs(X) <= 390)),
    ]
    zres = {n: {'missing': int((miss & z).sum()), 'extra': int((ext & z).sum())}
            for n, z in zones}
    tp, tr = outline(A, 'top'), outline(B, 'top')
    bp, br = outline(A, 'bottom'), outline(B, 'bottom')
    grid = []
    for xt in range(-450, 451, 25):
        c = int((xt / 900.0 + 0.5) * (w - 1))
        row = {'x': xt}
        for key, arr_p, arr_m, sign in (('top', tp, tr, 1), ('bottom', bp, br, -1)):
            if np.isfinite(arr_p[c]) and np.isfinite(arr_m[c]):
                row[key + '_photo'] = round(float((h - 1 - arr_p[c]) * S), 1)
                row[key + '_model'] = round(float((h - 1 - arr_m[c]) * S), 1)
                row[key + '_dz'] = round(float(sign * (arr_p[c] - arr_m[c]) * S), 1)
        grid.append(row)
    inter, union = int((A & B).sum()), int((A | B).sum())
    return {
        'iou': round(inter / union, 4),
        'union_px': union,
        'error_px': int(miss.sum() + ext.sum()),
        'error_px_for_99': int(union * 0.01),
        'missing_px': int(miss.sum()),
        'extra_px': int(ext.sum()),
        'zones': zres,
        'outline': grid,
    }, C.overlay(A, B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stl')
    ap.add_argument('--photo', default='1793820501.png')
    ap.add_argument('--json')
    ap.add_argument('--overlay')
    args = ap.parse_args()
    res, img = report(os.path.join(ROOT, args.stl), os.path.join(ROOT, args.photo))
    if args.overlay:
        os.makedirs(os.path.dirname(os.path.abspath(args.overlay)), exist_ok=True)
        Image.fromarray(img).save(args.overlay)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump(res, f, indent=2)
    print(f"IoU {res['iou']}   error {res['error_px']} px   "
          f"budget for 0.99: {res['error_px_for_99']} px   "
          f"(missing {res['missing_px']} / extra {res['extra_px']})")
    for n, d in res['zones'].items():
        print(f"  {n:12s} missing {d['missing']:6d}  extra {d['extra']:6d}")
    print('     X   top_photo top_model   dz |  bot_photo bot_model   dz')
    for r in res['outline']:
        if 'top_dz' in r:
            print(f"{r['x']:6d} {r['top_photo']:10.1f} {r['top_model']:9.1f} "
                  f"{r['top_dz']:+5.1f} | {r.get('bottom_photo', 0):9.1f} "
                  f"{r.get('bottom_model', 0):9.1f} {r.get('bottom_dz', 0):+5.1f}")


if __name__ == '__main__':
    main()
