#!/usr/bin/env python3
"""Software rasterizer (z-buffer) for STL previews — no OpenGL needed.

Renders orthographic views of a mesh with Lambert shading, optional
silhouette output. Used to visually compare the model against the
reference photos of the VW Tayron.
"""
import argparse
import numpy as np
import trimesh
from numba import njit
from PIL import Image


@njit(cache=True)
def rasterize(tris, cols, W, H, zbuf, img, bg):
    for i in range(H):
        for j in range(W):
            img[i, j, 0] = bg[0]
            img[i, j, 1] = bg[1]
            img[i, j, 2] = bg[2]
            zbuf[i, j] = -1e30
    for t in range(tris.shape[0]):
        x0, y0, z0 = tris[t, 0]
        x1, y1, z1 = tris[t, 1]
        x2, y2, z2 = tris[t, 2]
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        if minx > maxx or miny > maxy:
            continue
        d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(d) < 1e-12:
            continue
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                fx = px + 0.5
                fy = py + 0.5
                w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) / d
                w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) / d
                w2 = 1.0 - w0 - w1
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z > zbuf[py, px]:
                    zbuf[py, px] = z
                    img[py, px, 0] = cols[t, 0]
                    img[py, px, 1] = cols[t, 1]
                    img[py, px, 2] = cols[t, 2]
    return img, zbuf


def camera_basis(az_deg, el_deg):
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    # view direction: from camera towards origin
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return right, up, fwd


def render(mesh, az, el, size=(1400, 1000), margin=0.06, base_color=(225, 228, 232),
           bg=(255, 255, 255), silhouette=False, ground_shadow=False):
    V = mesh.vertices
    F = mesh.faces
    right, up, fwd = camera_basis(az, el)
    R = np.stack([right, up, fwd])  # rows
    P = V @ R.T
    xmin, ymin = P[:, 0].min(), P[:, 1].min()
    xmax, ymax = P[:, 0].max(), P[:, 1].max()
    W, H = size
    avail_w = W * (1 - 2 * margin)
    avail_h = H * (1 - 2 * margin)
    s = min(avail_w / (xmax - xmin), avail_h / (ymax - ymin))
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    px = (P[:, 0] - cx) * s + W / 2
    py = H / 2 - (P[:, 1] - cy) * s
    pz = P[:, 2]
    tris = np.stack([px[F], py[F], pz[F]], axis=2).astype(np.float64)

    # shading
    N = mesh.face_normals
    key = np.array([0.35, -0.45, 0.82]); key /= np.linalg.norm(key)
    fill = np.array([-0.6, 0.55, 0.35]); fill /= np.linalg.norm(fill)
    lam = 0.30 + 0.62 * np.clip(N @ key, 0, 1) + 0.25 * np.clip(N @ fill, 0, 1)
    # slight rim darkening for faces pointing away from camera
    facing = np.clip(N @ (-fwd), 0, 1)
    lam *= (0.75 + 0.25 * facing)
    lam = np.clip(lam, 0, 1.05)
    base = np.array(base_color, dtype=np.float64)
    cols = np.clip(lam[:, None] * base[None, :], 0, 255)
    if silhouette:
        cols[:] = np.array([40, 44, 52], dtype=np.float64)
    cols = np.ascontiguousarray(cols)

    zbuf = np.empty((H, W), dtype=np.float64)
    img = np.empty((H, W, 3), dtype=np.float64)
    img, zbuf = rasterize(np.ascontiguousarray(tris), cols, W, H, zbuf, img,
                          np.array(bg, dtype=np.float64))
    out = img.astype(np.uint8)
    if ground_shadow:
        mask = zbuf > -1e29
        # elliptical soft shadow under the object
        yy, xx = np.mgrid[0:H, 0:W]
        obj_rows = np.where(mask.any(axis=1))[0]
        if len(obj_rows):
            bottom = obj_rows.max()
            shadow_cy = bottom + 0.012 * H
            rows = mask.any(axis=0)
            obj_cols = np.where(rows)[0]
            left, rightx = obj_cols.min(), obj_cols.max()
            scx = (left + rightx) / 2
            srx = (rightx - left) * 0.48
            sry = 0.035 * H
            d = ((xx - scx) / srx) ** 2 + ((yy - shadow_cy) / sry) ** 2
            sh = np.clip(1 - d, 0, 1) ** 1.5 * 0.35
            sh[mask] = 0
            out = (out * (1 - sh[..., None]) + 90 * sh[..., None]).astype(np.uint8)
    return out


VIEWS = {
    # model: X = length (front = +X, to verify), Y = width, Z = up
    'side_right': (0, -90, 2),      # looking from -Y towards +Y: sees right side? adjust after first look
    'side_left': (0, 90, 2),
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
    ap.add_argument('--size', nargs=2, type=int, default=[1400, 1000])
    args = ap.parse_args()
    import os
    os.makedirs(args.outdir, exist_ok=True)
    mesh = trimesh.load(args.stl)
    views = args.views or list(VIEWS.keys())
    for name in views:
        _, az, el = VIEWS[name]
        img = render(mesh, az, el, size=tuple(args.size),
                     silhouette=args.silhouette, ground_shadow=not args.silhouette)
        path = os.path.join(args.outdir, f'{name}.png')
        Image.fromarray(img).save(path)
        print('saved', path)


if __name__ == '__main__':
    main()
