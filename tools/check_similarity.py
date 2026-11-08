#!/usr/bin/env python3
"""Measure how close a model's side silhouette is to the reference photo.

The photo and an orthographic silhouette render are both reduced to a binary
mask, scaled to the same overall length and stood on the same ground line.
IoU then says, in one number, how far the outline is from the car in the
photo; the overlay shows where.  Nothing here is aligned by eye.

Getting the reference mask right matters more than it looks.  The car is white
on a white background, so a dark-pixel threshold keeps only glass and tyres;
and the floor shadow under the car is as black as the rubber, so brightness
alone cannot separate them.  Below the sill the car is exactly two circular
tyres, so the two tyre discs are fitted to the outboard arcs of the mask and
everything else below the sill is discarded as shadow.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.optimize import least_squares

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


def _outline_bottom(mask):
    h, w = mask.shape
    bot = np.full(w, -1, int)
    for x in range(w):
        ys = np.where(mask[:, x])[0]
        if len(ys):
            bot[x] = ys.max()
    return bot


def _fit_tyre(mask, xs_range, side, ground, sill):
    """Fit a tyre disc to the outboard part of the mask's bottom outline.

    The disc is tangent to the ground, so only its centre X and radius are
    free.  Only the outboard half of the arc is used: the floor shadow runs
    inboard, between the wheels, and would flatten the fit.  Columns whose
    bottom is barely below the sill are dropped so the bumper valance does
    not get mistaken for tyre.
    """
    bot = _outline_bottom(mask)
    xs = np.arange(xs_range[0], xs_range[1])
    ys = bot[xs]
    if not len(ys) or ys.max() < 0:
        return None
    ground = int(ys.max())          # per wheel: the camera is not perfectly level
    ok = ys > sill + 0.40 * (ground - sill)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 20:
        return None
    contact = xs[ys >= ground - 2]
    if not len(contact):
        return None
    r0 = 1.40 * (ground - sill)
    x_edge = contact.min() if side < 0 else contact.max()
    cx0 = x_edge - side * 0.15 * r0
    # The contact shadow flattens the outline for ~15 px above the floor, so
    # fit the arc above that band and let the centre float: forcing tangency
    # to the shadow line pulls the radius down by a quarter.
    keep = ((xs - cx0) * (-side) <= 0.10 * r0) & (ys < ground - 14)
    px, py = xs[keep].astype(float), ys[keep].astype(float)
    if len(px) < 12:
        return None

    def resid(p):
        cx, cy, r = p
        dy = np.clip(r * r - (px - cx) ** 2, 1e-6, None)
        return cy + np.sqrt(dy) - py

    fit = least_squares(resid, [cx0, ground - r0, r0],
                        bounds=([0, ground - 2.4 * r0, 0.7 * r0],
                                [mask.shape[1], ground, 2.4 * r0]))
    cx, cy, r = (float(v) for v in fit.x)
    return cx, cy, r * 1.02


def photo_mask(path, thresh=246, cut_shadow=True):
    m, a = raw_mask(path, thresh)
    if cut_shadow:
        m = _level_and_clean(m, a)
    return m


def raw_mask(path, thresh=246):
    """Car-vs-background mask, and the photo cropped to the very same box.

    Returning both from one place matters: the crop is taken *after* the
    underbody cut, so re-deriving it from the thresholded image alone lands a
    dozen pixels off and silently misaligns every measurement against it.

    thresh=246 rather than 235 because the studio background measures 247-250
    while specular highlights on the glass and bonnet reach 241: a lower
    threshold punches holes in the car.
    """
    a = np.asarray(Image.open(path).convert('RGB')).astype(int)
    m = largest_component(a.min(2) < thresh)
    x0, x1, y0, y1 = bbox(m)
    # Between the wheels the background is still visible under the car.  The
    # floor shadow starts on the first row where that gap closes; cutting it
    # BEFORE filling holes keeps the underbody gap open, otherwise fill_holes
    # bridges it and invents a solid skirt.
    mid = slice(int(x0 + 0.35 * (x1 - x0)), int(x0 + 0.65 * (x1 - x0)))
    frac = (a[:, mid, :].min(2) >= thresh).mean(1)
    y = int(y0 + 0.5 * (y1 - y0))
    while y < y1 and frac[y] <= 0.5:
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
    m = m[y0:y1 + 1, x0:x1 + 1]
    a = a[y0:y1 + 1, x0:x1 + 1]
    return m, a


def _tyres(m, a):
    """Locate both tyre discs from the wheel hubs, then fit the outer radius.

    The dark ring behind the alloy spokes is a compact, near-circular blob
    whose bounding box is symmetric about the axle, so its extent midpoint
    gives the wheel centre to about a pixel.  A brightness centroid does not:
    spoke lighting is asymmetric and drags it 10-25 px inboard.

    With the centre pinned, the radius is the only unknown and the outboard
    bottom arc determines it to sub-pixel residual.  Only the outboard half
    is used because the floor shadow runs inboard and is exactly as black as
    rubber - no brightness test can separate them there.
    """
    h, w = m.shape
    bot = _outline_bottom(m)
    centre = bot[int(0.44 * w):int(0.56 * w)]
    centre = centre[centre > 0]
    if not len(centre):
        return None
    sill = int(np.median(centre))
    lab, n = ndi.label((a.max(2) < 100) & m)
    if n == 0:
        return None
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    hubs = []
    for k in np.argsort(sizes)[::-1]:
        if sizes[k] < 8000:
            break
        ys, xs = np.where(lab == k + 1)
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        if not (0.06 * w < bw < 0.20 * w and 0.8 < bw / max(bh, 1) < 1.25
                and ys.mean() > 0.55 * h):
            continue
        hubs.append((float(xs.min() + xs.max()) * 0.5,
                     float(ys.min() + ys.max()) * 0.5))
    if len(hubs) != 2:
        return None
    hubs.sort()
    out = []
    for cx, cy in hubs:
        side = -1 if cx < w * 0.5 else 1
        r0 = 1.4 * (h - 1 - sill)
        lo, hi = max(int(cx - r0), 0), min(int(cx + r0), w)
        ground = int(bot[lo:hi].max())
        cols = np.arange(lo, hi)
        keep = ((cols - cx) * (-side) <= 0)
        cols = cols[keep]
        rows = bot[cols]
        keep = (rows > sill + 0.30 * (ground - sill)) & (rows < ground - 8)
        px, py = cols[keep].astype(float), rows[keep].astype(float)
        if len(px) < 10:
            return None

        def resid(r):
            return np.hypot(px - cx, py - cy) - r

        fit = least_squares(resid, [r0], loss='soft_l1', f_scale=3.0,
                            bounds=([0.7 * r0], [1.8 * r0]))
        out.append((cx, cy, float(fit.x[0]), cy, ground))
    return sill, out[0], out[1]


def _level_and_clean(m, a):
    """Undo camera roll, then drop everything below the sill that is not tyre.

    The two wheel centres in this photo sit ~9 px apart in height over 1375 px
    of wheelbase - a 0.37 deg camera roll, not a shape difference.  Levelling
    on the wheel centres removes a capture artefact instead of charging it to
    the model.
    """
    got = _tyres(m, a)
    if got is None:
        return m
    _, rear, front = got
    roll = np.degrees(np.arctan2(front[3] - rear[3], front[0] - rear[0]))
    if abs(roll) > 0.02:
        mf = ndi.rotate(m.astype(float), -roll, order=1, mode='constant', cval=0.0) > 0.5
        af = np.stack([ndi.rotate(a[:, :, k].astype(float), -roll, order=1,
                                  mode='nearest') for k in range(3)], axis=2)
        m, a = largest_component(mf), af
        x0, x1, y0, y1 = bbox(m)
        m, a = m[y0:y1 + 1, x0:x1 + 1], a[y0:y1 + 1, x0:x1 + 1]
    got = _tyres(m, a)
    if got is None:
        return m
    sill, rear, front = got
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    discs = np.zeros_like(m)
    for cx, cy, r, _, ground in (rear, front):
        # second copy nudged down to the floor keeps the contact patch that
        # the shadow-dilated fit would otherwise shave off
        drop = max(0.0, ground - (cy + r))
        for dy in (0.0, drop):
            discs |= (xx - cx) ** 2 + (yy - cy - dy) ** 2 <= (r + 1.0) ** 2
    out = m.copy()
    out[sill + 1:, :] &= discs[sill + 1:, :]
    out = largest_component(out)
    x0, x1, y0, y1 = bbox(out)
    return out[y0:y1 + 1, x0:x1 + 1]


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


def overlay(A, B):
    H, W = A.shape
    img = np.full((H, W, 3), 255, np.uint8)
    img[np.logical_and(A, B)] = (150, 150, 150)
    img[np.logical_and(A, ~B)] = (220, 40, 40)     # photo only: model misses material
    img[np.logical_and(~A, B)] = (40, 90, 220)     # model only: model has extra
    return img


def compare(stl, photo, out_png):
    ref = photo_mask(photo)
    mod = render_mask(stl)
    A, B = align(mod, ref)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    iou = float(inter) / float(union)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    Image.fromarray(overlay(A, B)).save(out_png)
    return {
        'iou': round(iou, 4),
        'photo_px': int(A.sum()),
        'model_px': int(B.sum()),
        'missing_px': int(np.logical_and(A, ~B).sum()),
        'extra_px': int(np.logical_and(~A, B).sum()),
        'height_ratio_model_over_photo': round(float(np.where(B.any(1))[0].size /
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
