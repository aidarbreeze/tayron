#!/usr/bin/env python3
"""Conservative, photo-guided cleanup of the Tayron STL.

The source mesh is already closed and consistently wound.  This script does
not remesh or boolean it.  It only moves selected vertices, with displacement
caps, to reduce scan-like noise and tidy the roof rails / bumper intake.
The original STL is never modified.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def vertex_sharp_mask(mesh, angle_deg=25):
    sharp = np.zeros(len(mesh.vertices), dtype=bool)
    e = mesh.face_adjacency_edges[mesh.face_adjacency_angles > np.radians(angle_deg)]
    sharp[e.ravel()] = True
    return sharp


def adjacency_matrix(mesh):
    e = mesh.edges_unique
    n = len(mesh.vertices)
    w = np.ones(len(e) * 2)
    rows = np.concatenate([e[:, 0], e[:, 1]])
    cols = np.concatenate([e[:, 1], e[:, 0]])
    W = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(W.sum(axis=1)).ravel()
    W = W.multiply((1.0 / np.maximum(deg, 1))[:, None]).tocsr()
    return W


def taubin_masked(V, W, mask, iterations=8, lam=0.5, mu=-0.53, max_disp=1.5):
    """Taubin smoothing on interior masked vertices; boundary stays fixed."""
    orig = V.copy()
    # Interior = masked vertices whose one-ring neighbours are all masked.
    neigh = W.tocoo()
    all_neigh_mask = np.ones(len(V), dtype=bool)
    np.logical_and.at(all_neigh_mask, neigh.row, mask[neigh.col])
    interior = mask & all_neigh_mask
    for _ in range(iterations):
        for factor in (lam, mu):
            avg = W @ V
            d = avg - V
            step = factor * d
            disp = np.linalg.norm(V + step - orig, axis=1)
            scale = np.minimum(1.0, max_disp / np.maximum(disp, 1e-9))
            V[interior] += step[interior] * scale[interior, None]
    return V, interior


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def repair(src: Path, dst: Path, report_path: Path):
    mesh = trimesh.load(src, process=True)
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    V0 = mesh.vertices.copy()
    V = V0.copy()
    comps = mesh.split(only_watertight=False)
    body = max(comps, key=lambda m: len(m.faces))
    # Map body vertices back to combined mesh by exact coordinates.
    tree_all = cKDTree(V0)
    _, body_idx = tree_all.query(body.vertices)
    is_body = np.zeros(len(V), dtype=bool)
    is_body[body_idx] = True

    sharp = vertex_sharp_mask(mesh, 25)
    W = adjacency_matrix(mesh)
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ay = np.abs(y)

    # 1. Broad-panel denoise. Protect creases and small details.
    detail = (
        ((x > 385) & (z < 25) & (ay < 135)) |          # grille / bumper intakes
        ((x < -395) & (z > 15) & (z < 75) & (ay < 160)) |  # rear lamps / bar
        ((ay > 165) & (z > -20) & (z < 40)) |          # handles / mirrors
        ((z < -25) & (ay > 125)) |                     # arches / sills
        ((z > 145) & (ay > 95) & (ay < 140) & (x > -270) & (x < 70))  # rails
    )
    panel_mask = is_body & ~sharp & ~detail & (z > -125)
    V, panel_interior = taubin_masked(V, W, panel_mask, iterations=10, max_disp=1.2)

    # 2. Roof rails: local smoothing + left/right symmetry.
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    ay = np.abs(y)
    rail_mask = is_body & (x > -265) & (x < 65) & (ay > 98) & (ay < 138) & (z > 145) & ~sharp
    V, rail_interior = taubin_masked(V, W, rail_mask, iterations=14, lam=0.6, mu=-0.63, max_disp=3.0)
    # Symmetrise rail vertices against the opposite side rail pool.
    rail_idx = np.where(rail_mask)[0]
    pos = rail_idx[y[rail_idx] > 0]
    neg = rail_idx[y[rail_idx] < 0]
    if len(pos) and len(neg):
        mirrored = V[pos].copy(); mirrored[:, 1] *= -1
        d, j = cKDTree(V[neg]).query(mirrored)
        ok = d < 6.0
        avg = 0.5 * (V[pos[ok]] + V[neg[j[ok]]] * np.array([1, -1, 1]))
        # Limit symmetry correction.
        delta = avg - V[pos[ok]]
        dn = np.linalg.norm(delta, axis=1, keepdims=True)
        delta *= np.minimum(1.0, 2.0 / np.maximum(dn, 1e-9))
        V[pos[ok]] += delta
        V[neg[j[ok]]] = V[pos[ok]] * np.array([1, -1, 1])
    V, _ = taubin_masked(V, W, rail_mask, iterations=6, lam=0.5, mu=-0.53, max_disp=3.0)

    # 3. Front lower intake: melt chaotic blobs, keep opening boundary fixed.
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    intake = is_body & (x > 424) & (z > -108) & (z < -62) & (np.abs(y) < 116) & ~sharp
    V, intake_interior = taubin_masked(V, W, intake, iterations=16, lam=0.65, mu=-0.68, max_disp=4.0)

    # 4. Rear light bar: reduce waviness, avoid centre emblem.
    x, y, z = V[:, 0], V[:, 1], V[:, 2]
    bar = is_body & (x < -428) & (z > 32) & (z < 62) & (np.abs(y) > 22) & (np.abs(y) < 112) & ~sharp
    V, bar_interior = taubin_masked(V, W, bar, iterations=10, lam=0.55, mu=-0.58, max_disp=2.5)
    # Gently straighten the bar band in Z.
    band = bar & (np.abs(z - 47) < 9)
    weight = 0.35 * smoothstep(9 - np.abs(z[band] - 47), 0, 9)
    V[band, 2] += (47 - z[band]) * weight

    mesh.vertices = V
    mesh.fix_normals()
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)

    disp = np.linalg.norm(V - V0, axis=1)
    moved = disp > 1e-6
    report = {
        'source': str(src), 'output': str(dst),
        'vertices': len(V), 'faces': len(mesh.faces),
        'watertight': bool(mesh.is_watertight),
        'winding_consistent': bool(mesh.is_winding_consistent),
        'moved_vertices': int(moved.sum()),
        'max_displacement': float(disp.max()),
        'mean_displacement_moved': float(disp[moved].mean()) if moved.any() else 0.0,
        'panel_interior_vertices': int(panel_interior.sum()),
        'rail_vertices': int(rail_mask.sum()),
        'intake_interior_vertices': int(intake_interior.sum()),
        'rear_bar_interior_vertices': int(bar_interior.sum()),
        'bounds_before': mesh.bounds.tolist(),
        'notes': [
            'Topology unchanged; only vertex positions were adjusted.',
            'Sharp creases and small details were protected by masks.',
            'Displacements were capped per region to avoid silhouette drift.',
            'This is a visual cleanup, not a certified manufacturing repair.',
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--report', default='reports/repair_report.json')
    args = ap.parse_args()
    repair(Path(args.src), Path(args.dst), Path(args.report))
