#!/usr/bin/env python3
"""CPU STL inspection renderer: orthographic clay views with crease normals
and depth-based cavity shading. No OpenGL or external models are used.

The cavity term makes recessed wheel spokes and panel seams readable in a
single-material STL preview. It is an inspection aid, not a physical render.
"""
import argparse
import os

import numpy as np
import trimesh
from numba import njit
from PIL import Image


@njit(cache=True)
def rasterize(tris, cols, normals, W, H, zbuf, img, nbuf, bg):
    img[:] = bg
    zbuf[:] = -1e30
    nbuf[:] = 0.0
    for t in range(tris.shape[0]):
        x0, y0, z0 = tris[t, 0]
        x1, y1, z1 = tris[t, 1]
        x2, y2, z2 = tris[t, 2]
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-12 or minx > maxx or miny > maxy:
            continue
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                fx, fy = px + 0.5, py + 0.5
                w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) / d
                w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) / d
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z > zbuf[py, px]:
                    zbuf[py, px] = z
                    for c in range(3):
                        if cols.ndim == 3:
                            img[py, px, c] = (w0 * cols[t, 0, c] +
                                              w1 * cols[t, 1, c] +
                                              w2 * cols[t, 2, c])
                        else:
                            img[py, px, c] = cols[t, c]
                        nbuf[py, px, c] = (w0 * normals[t, 0, c] +
                                           w1 * normals[t, 1, c] +
                                           w2 * normals[t, 2, c])


@njit(cache=True)
def cavity_ao(zbuf, nbuf, scale, radii):
    H, W = zbuf.shape
    ao = np.ones((H, W), np.float32)
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1))
    for y in range(H):
        for x in range(W):
            z0 = zbuf[y, x]
            if z0 < -1e29:
                continue
            nx, ny, nz = nbuf[y, x]
            ln = np.sqrt(nx * nx + ny * ny + nz * nz)
            if ln > 1e-6:
                nx /= ln
                ny /= ln
                nz /= ln
            if nz < 0:
                nx, ny, nz = -nx, -ny, -nz
            nz = max(nz, 0.25)
            occ = 0.0
            count = 0
            for r in radii:
                rw = r / scale
                bias = 0.12 + 0.025 * rw
                for dx, dy in dirs:
                    xx, yy = x + dx * r, y + dy * r
                    if xx < 0 or xx >= W or yy < 0 or yy >= H:
                        continue
                    z1 = zbuf[yy, xx]
                    if z1 < -1e29:
                        continue
                    # Expected neighbour depth on the centre pixel's plane.
                    plane = z0 + (-nx * dx * r + ny * dy * r) / (scale * nz)
                    h = (z1 - plane - bias) / rw
                    occ += min(max(h, 0.0), 0.65)
                    count += 1
            if count:
                ao[y, x] = 1.0 - min(0.42, 1.10 * occ / count)
    return ao


def camera_basis(az_deg, el_deg):
    """Returns screen right, screen up, and object-to-camera direction."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def crease_normals(mesh, angle_deg=35):
    """Angle-weighted corner normals split at sharp edges."""
    smooth = trimesh.graph.smooth_shade(mesh, angle=np.radians(angle_deg),
                                        facet_minarea=None)
    order = np.concatenate(smooth.metadata['original_components'])
    cn = np.empty((len(mesh.faces), 3, 3), dtype=np.float64)
    cn[order] = smooth.vertex_normals[smooth.faces]
    return cn


def projection(mesh, az, el, size, margin):
    right, up, fwd = camera_basis(az, el)
    R = np.stack([right, up, fwd])
    P = mesh.vertices @ R.T
    lo, hi = P[:, :2].min(axis=0), P[:, :2].max(axis=0)
    W, H = size
    s = min(W * (1 - 2 * margin) / (hi[0] - lo[0]),
            H * (1 - 2 * margin) / (hi[1] - lo[1]))
    c = (lo + hi) / 2
    return R, s, c


def render(mesh, az, el, size=(1400, 1000), margin=0.06,
           base_color=(225, 228, 232), bg=(255, 255, 255),
           silhouette=False, ground_shadow=False, flat=False,
           frame_mesh=None, supersample=2, ao=True):
    W, H = size[0] * supersample, size[1] * supersample
    right, up, fwd = camera_basis(az, el)
    R, s, c = projection(frame_mesh if frame_mesh is not None else mesh,
                         az, el, (W, H), margin)
    P = mesh.vertices @ R.T
    F = mesh.faces
    px = (P[:, 0] - c[0]) * s + W / 2
    py = H / 2 - (P[:, 1] - c[1]) * s
    tris = np.ascontiguousarray(np.stack([px[F], py[F], P[F, 2]], axis=2))
    key = np.array([-0.42, 0.72, 0.80]); key /= np.linalg.norm(key)
    fill = np.array([0.70, 0.15, 0.65]); fill /= np.linalg.norm(fill)
    N = mesh.face_normals if flat else crease_normals(mesh)
    Ncam = N @ R.T
    lam = (0.42 + 0.50 * np.clip(Ncam @ key, 0, 1)[..., None]
           + 0.16 * np.clip(Ncam @ fill, 0, 1)[..., None])
    lam *= 0.92 + 0.08 * np.clip(Ncam[..., 2], 0, 1)[..., None]
    cols = np.clip(lam * np.asarray(base_color), 0, 255)
    if silhouette:
        cols = np.full((len(F), 3), 40.0)
        Ncam = np.repeat(mesh.face_normals[:, None, :] @ R.T, 3, axis=1)
    zbuf = np.empty((H, W), np.float32)
    img = np.empty((H, W, 3), np.float32)
    nbuf = np.empty((H, W, 3), np.float32)
    rasterize(tris, np.ascontiguousarray(cols), np.ascontiguousarray(Ncam),
              W, H, zbuf, img, nbuf, np.asarray(bg, np.float32))
    if ao and not silhouette:
        shade = cavity_ao(zbuf, nbuf, s, np.array([3, 7, 14, 28], np.int64) * supersample)
        img *= shade[..., None]
    if supersample != 1:
        img = img.reshape(size[1], supersample, size[0], supersample, 3).mean(axis=(1, 3))
        zbuf = zbuf.reshape(size[1], supersample, size[0], supersample).max(axis=(1, 3))
        W, H = size
    out = np.clip(img, 0, 255).astype(np.uint8)
    if ground_shadow and not silhouette:
        mask = zbuf > -1e29
        yy, xx = np.mgrid[0:H, 0:W]
        rows, cols_ = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        if len(rows) and len(cols_):
            cy, cx = rows.max() + 0.012 * H, (cols_.min() + cols_.max()) / 2
            rx, ry = (cols_.max() - cols_.min()) * 0.48, 0.035 * H
            d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
            sh = np.clip(1 - d, 0, 1) ** 1.5 * 0.30
            sh[mask] = 0
            out = (out * (1 - sh[..., None]) + 90 * sh[..., None]).astype(np.uint8)
    return out


VIEWS = {
    # X = length, front = +X; Y = width; Z = up.
    'side_right': (0, -90, 0),
    'side_left': (0, 90, 0),
    'front': (1, 0, 4),
    'rear': (1, 180, 4),
    'front34_left': (28, 38, 12),
    'front34_right': (28, -38, 12),
    'rear34_left': (28, 142, 12),
    'rear34_right': (28, -142, 12),
    'top': (0, -90, 89.9),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stl')
    ap.add_argument('outdir')
    ap.add_argument('--views', nargs='*', default=None)
    ap.add_argument('--silhouette', action='store_true')
    ap.add_argument('--flat', action='store_true')
    ap.add_argument('--no-ao', action='store_true')
    ap.add_argument('--frame-mesh')
    ap.add_argument('--size', nargs=2, type=int, default=[1400, 1000])
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    mesh = trimesh.load(args.stl)
    frame = trimesh.load(args.frame_mesh) if args.frame_mesh else None
    for name in args.views or list(VIEWS):
        _, az, el = VIEWS[name]
        img = render(mesh, az, el, size=tuple(args.size), flat=args.flat,
                     frame_mesh=frame, silhouette=args.silhouette,
                     ground_shadow=not args.silhouette, ao=not args.no_ao)
        path = os.path.join(args.outdir, f'{name}.png')
        Image.fromarray(img).save(path)
        print('saved', path)


if __name__ == '__main__':
    main()
