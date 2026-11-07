#!/usr/bin/env python3
"""Photo-guided roof-rail correction.

The source has broad, flat rail pads.  The reference has a slim silver cap
on a recessed base, with tapered end feet.  We fair the old pad into the
local roof shoulder, then union a narrow pedestal and rounded cap.  This
keeps the rail attached (no floating arch) and avoids the blocky feet of the
first replacement pass.
"""
import argparse

import numpy as np
import trimesh
from scipy.interpolate import interp1d
from shapely.geometry import Polygon, box


def _curve(x, z, ay, ylo, yhi, zmin=138):
    xs, zs = [], []
    for x0 in range(-280, 80, 10):
        sel = (x > x0) & (x < x0 + 10) & (ay > ylo) & (ay < yhi) & (z > zmin)
        if sel.sum() > 5:
            xs.append(x0 + 5)
            zs.append(np.median(z[sel]))
    return interp1d(xs, zs, kind='cubic', fill_value='extrapolate')


def roof_curves(body):
    v = body.vertices
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    ay = np.abs(y)
    crown = _curve(x, z, ay, 82, 98)
    shoulder = _curve(x, z, ay, 118, 126)
    return crown, shoulder


def rounded_rect(w, h, r=0.8):
    return box(-w / 2 + r, -h / 2 + r, w / 2 - r, h / 2 - r).buffer(r, resolution=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    args = ap.parse_args()
    mesh = trimesh.load(args.src, process=True)
    comps = sorted(mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True)
    body, wheels = comps[0], comps[1:]
    assert len(wheels) == 4, 'expected four wheel shells'
    crown, shoulder = roof_curves(body)

    V = body.vertices.copy()
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ay = np.abs(y)
    c, s = crown(x), shoulder(x)
    # Local roof shoulder slope, measured away from the old rail cap.
    slope = (s - c) / (122.0 - 90.0)
    roof = c + slope * (ay - 90.0)
    excess = z - roof
    pad = (x > -268) & (x < 68) & (ay > 98) & (ay < 140) & (z > 140)
    w = np.clip((excess - 1.0) / 5.0, 0, 1) * pad
    V[:, 2] = z - w * (excess - 0.6)
    body.vertices = V
    # Fair the flattened pad into the surrounding roof so no rectangular scar
    # remains under the new rail.
    from scipy.sparse import coo_matrix
    e = body.edges_unique
    n = len(V)
    W = coo_matrix((np.ones(len(e) * 2),
                    (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                   shape=(n, n)).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    W = W.multiply((1.0 / np.maximum(deg, 1))[:, None]).tocsr()
    mask = pad
    neigh = W.tocoo()
    allm = np.ones(n, dtype=bool)
    np.logical_and.at(allm, neigh.row, mask[neigh.col])
    interior = mask & allm
    orig = V.copy()
    for _ in range(18):
        for fac in (0.5, -0.53):
            step = fac * (W @ V - V)
            disp = np.linalg.norm(V + step - orig, axis=1)
            scale = np.minimum(1.0, 2.0 / np.maximum(disp, 1e-9))
            V[interior] += step[interior] * scale[interior, None]
    body.vertices = V

    xx = np.linspace(-250, 50, 81)
    t = (xx + 250) / 300
    rail_roof = crown(xx) + ((shoulder(xx) - crown(xx)) / 32.0) * (112.0 - 90.0)
    # Taper the assembly into the roof at both ends (no vertical posts).
    end = np.clip(np.minimum(t, 1.0 - t) / 0.10, 0, 1)
    end = end * end * (3 - 2 * end)
    ped_h = 2.0 + (8.0 + 3.0 * np.sin(np.pi * t) ** 0.7 - 2.0) * end
    new_parts = []
    for side in (-1, 1):
        # Recessed pedestal: attached to the roof along the whole rail.
        ped_path = np.column_stack([xx, np.full_like(xx, side * 112.0), rail_roof + 5.0])
        pedestal = trimesh.creation.sweep_polygon(rounded_rect(6.4, 10.0, 0.7), ped_path)
        pv = pedestal.vertices.copy()
        roof_v = np.interp(pv[:, 0], xx, rail_roof)
        h_v = np.interp(pv[:, 0], xx, ped_h)
        pv[:, 2] = roof_v + (pv[:, 2] - roof_v) * (h_v / 10.0)
        pedestal.vertices = pv
        # Slim rounded cap sitting on the pedestal.
        cap_z = rail_roof + ped_h + 1.0
        cap_path = np.column_stack([xx, np.full_like(xx, side * 112.0), cap_z])
        cap = trimesh.creation.sweep_polygon(rounded_rect(8.0, 4.6, 1.2), cap_path)
        cv = cap.vertices.copy()
        w_v = 0.65 + 0.35 * np.interp(cv[:, 0], xx, end)
        cv[:, 1] = side * 112.0 + (cv[:, 1] - side * 112.0) * w_v
        cap.vertices = cv
        rail_solid = trimesh.boolean.union([pedestal, cap], engine='manifold')
        # Tapered end feet, blended into the pedestal rather than floating boxes.
        for end_x, direction in ((-244, 1), (44, -1)):
            foot = trimesh.creation.box(extents=[24, 9.5, 3.2])
            z0 = float(crown(end_x) + ((shoulder(end_x) - crown(end_x)) / 32.0) * 22.0)
            foot.apply_translation([end_x, side * 112.0, z0 + 1.2])
            rail_solid = trimesh.boolean.union([rail_solid, foot], engine='manifold')
        new_parts.append(rail_solid)
    body = trimesh.boolean.union([body] + new_parts, engine='manifold')
    out = trimesh.util.concatenate([body] + wheels)
    out.export(args.dst)
    print('watertight', out.is_watertight, 'components', len(out.split(only_watertight=False)),
          'faces', len(out.faces))


if __name__ == '__main__':
    main()
