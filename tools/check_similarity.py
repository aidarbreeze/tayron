#!/usr/bin/env python3
"""Measure how close a model's side silhouette is to the reference photo.

The photo and an orthographic silhouette render are both reduced to a binary
mask, scaled to the same overall length and stood on the same ground line.
IoU then says, in one number, how far the outline is from the car in the
photo; the overlay shows where.  Nothing here is aligned by eye.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def bbox(mask):
    ys, xs = np.where(mask)
    return xs.min(), xs.max(), ys.min(), ys.max()


def photo_mask(path, thresh=235):
    """Car silhouette from a studio photo with a near-white background.

    The car is white too, so a dark-pixel threshold is wrong: it keeps only
    glass and tyres.  Instead take everything that is not background, then
    cut the floor shadow away at the lowest genuinely dark pixel (the tyres).
    """
    a = np.asarray(Image.open(path).convert('RGB')).astype(int)
    m = largest_component(a.min(2) < thresh)
    x0, x1, y0, y1 = bbox(m)
    # Between the wheels the background is still visible under the car.  The
    # floor shadow starts on the first row where that gap closes; below it
    # only genuinely dark pixels (the tyres) belong to the car.  Cutting the
    # shadow BEFORE filling holes keeps the underbody gap open, otherwise
    # fill_holes bridges it and invents a solid skirt.
    mid = slice(int(x0 + 0.35 * (x1 - x0)), int(x0 + 0.65 * (x1 - x0)))
    frac = (a[:, mid, :].min(2) >= thresh).mean(1)
    y = int(y0 + 0.5 * (y1 - y0))
    while y < y1 and frac[y] <= 0.5:      # find the open gap under the car
        y += 1
    gap_bottom = y
    while gap_bottom < y1 and frac[gap_bottom] > 0.5:
        gap_bottom += 1
    gap_bottom -= 1
    dark = a.max(2) < 110
    m[gap_bottom + 1:, :] &= dark[gap_bottom + 1:, :]
    m = largest_component(m)
    m = ndi.binary_fill_holes(m)
    m = largest_component(m)
    x0, x1, y0, y1 = bbox(m)
    return m[y0:y1 + 1, x0:x1 + 1]


def render_mask(stl, view='side_left', size=(1800, 900)):
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
    import trimesh
    from render_mesh import VIEWS, render
    mesh = trimesh.load(stl)
    _, az, el = VIEWS[view]
    img = render(mesh, az, el, size=size, margin=0.04, silhouette=True,
                 ground_shadow=False, ao=False, supersample=1)
    m = largest_component(img.min(2) < 128)
    m = ndi.binary_fill_holes(m)
    x0, x1, y0, y1 = bbox(m)
    return m[y0:y1 + 1, x0:x1 + 1]


def align(model, target):
    """Scale the model mask to the target width, stand both on the ground."""
    h, w = target.shape
    scale = w / model.shape[1]
    mh = max(int(round(model.shape[0] * scale)), 1)
    mi = Image.fromarray((model * 255).astype(np.uint8)).resize((w, mh), Image.BILINEAR)
    mm = np.asarray(mi) > 127
    H = max(h, mh) + 8
    A = np.zeros((H, w), bool)
    B = np.zeros((H, w), bool)
    A[H - h:, :] = target
    B[H - mh:, :] = mm
    return A, B


def compare(stl, photo, out_png):
    ref = photo_mask(photo)
    mod = render_mask(stl)
    A, B = align(mod, ref)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    iou = float(inter) / float(union)
    H, W = A.shape
    img = np.full((H, W, 3), 255, np.uint8)
    img[np.logical_and(A, B)] = (150, 150, 150)
    img[np.logical_and(A, ~B)] = (220, 40, 40)     # photo only: model is missing material
    img[np.logical_and(~A, B)] = (40, 90, 220)     # model only: model has extra material
    Image.fromarray(img).save(out_png)
    return {
        'iou': round(iou, 4),
        'photo_px': int(A.sum()),
        'model_px': int(B.sum()),
        'height_ratio_model_over_photo': round(B.sum() and (np.where(B.any(1))[0].size /
                                                            np.where(A.any(1))[0].size), 4),
        'overlay': out_png,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stl')
    ap.add_argument('--photo', default='1793820501.png')
    ap.add_argument('--out', default='renders/similarity_side.png')
    args = ap.parse_args()
    import json
    res = compare(os.path.join(ROOT, args.stl) if not os.path.isabs(args.stl) else args.stl,
                  os.path.join(ROOT, args.photo),
                  os.path.join(ROOT, args.out))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
