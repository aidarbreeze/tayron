#!/usr/bin/env python3
"""
Оптимизация STL-модели Tayron 2023 (FAW-VW, 280TSI/300TSI) по образу эталона Tiguan.

Что делает конвейер:
  1. разбирает входной STL на кузов + 4 колеса;
  2. селективное сглаживание Таубина (сильнее на плоских панелях, слабее на решётке,
     фарах, молдингах) — убирает скан-«крупку» без «заплавляния» деталей;
  3. квадратичная децимация до бюджета граней Tiguan (~88k);
  4. скрытые «цапфы» (цилиндры внутри арок) + булево объединение -> один
     водонепроницаемый корпус вместо 5 отдельных деталей;
  5. финальная лёгкая релаксация, экспорт STL в исходной системе координат.

Зависимости:
    pip install --break-system-packages numpy scipy trimesh fast-simplification manifold3d

Пример:
    python3 tools/optimize_tayron.py "tayron lite 2 (1) (1).stl" tayron_2023_optimized.stl
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import trimesh
from scipy import sparse


def adjacency_average(mesh: trimesh.Trimesh) -> sparse.csr_matrix:
    """Строково-нормированная матрица усреднения по 1-кольцу соседей."""
    v, f = mesh.vertices, mesh.faces
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    row = np.concatenate([e[:, 0], e[:, 1]])
    col = np.concatenate([e[:, 1], e[:, 0]])
    a = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(v), len(v)))
    deg = np.asarray(a.sum(1)).ravel()
    return sparse.diags(1.0 / deg) @ a


def smooth(mesh, iters, lam=0.5, mu=-0.53, selective=True, floor=0.15, k=3.0):
    """Сглаживание Таубина (lambda/mu) с региональными весами.

    selective=True уменьшает силу сглаживания там, где локальная шероховатость
    (|v - центр соседей|) велика, т.е. на кромках, решётке, шильдиках — детали
    остаются читаемыми, а панели выглаживаются.
    """
    v = mesh.vertices.astype(np.float64).copy()
    w_neigh = adjacency_average(mesh)
    if selective:
        disp = np.linalg.norm(w_neigh @ v - v, axis=1)
        feat = np.clip(disp / (k * (np.median(disp) + 1e-12)), 0.0, 1.0)
        w = (1.0 - (1.0 - floor) * feat)[:, None]
    else:
        w = np.ones((len(v), 1))
    for _ in range(iters):
        v = v + w * lam * (w_neigh @ v - v)
        v = v + w * mu * (w_neigh @ v - v)
    out = mesh.copy()
    out.vertices = v
    return out


def largest_part(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    return max(mesh.split(only_watertight=False), key=lambda c: len(c.faces))


def denoise_part(mesh, iters, floor, k, faces, post_iters=4):
    s = smooth(mesh, iters, selective=True, floor=floor, k=k)
    s = largest_part(s.simplify_quadric_decimation(face_count=faces))
    s = smooth(s, post_iters, selective=False)          # убрать фасетки от децимации
    return largest_part(s.simplify_quadric_decimation(face_count=faces))


def hidden_stub(wheel, reach=52.0, start=18.0, radius_frac=0.34, sections=48):
    """Цилиндр внутри арки: соединяет колесо с внутренней стенкой кузова,
    снаружи не виден (радиус меньше радиуса шины)."""
    c, ext = wheel.center_mass, wheel.extents
    inward = 1.0 if c[1] < 0 else -1.0
    y0, y1 = c[1] + inward * start, c[1] + inward * reach
    cyl = trimesh.creation.cylinder(
        radius=min(ext[0], ext[2]) * radius_frac, height=abs(y1 - y0), sections=sections
    )
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cyl.apply_translation([c[0], (y0 + y1) / 2, c[2]])
    return cyl


def optimize(src_path, dst_path, body_faces=62000, wheel_faces=6200, final_faces=88000):
    t0 = time.time()
    scene = trimesh.load(src_path, force="mesh")
    parts = sorted(scene.split(only_watertight=False), key=lambda c: -len(c.faces))
    body, wheels = parts[0], parts[1:]
    print(f"вход: {len(scene.faces)} граней, {len(parts)} деталей "
          f"(кузов {len(body.faces)}, колёса {[len(w.faces) for w in wheels]})")

    body = denoise_part(body, iters=40, floor=0.15, k=3.0, faces=body_faces)
    print(f"кузов: {len(body.faces)} граней, watertight={body.is_watertight}")

    wheels = [denoise_part(w, iters=25, floor=0.35, k=4.0, faces=wheel_faces, post_iters=3)
              for w in wheels]
    print(f"колёса: {[len(w.faces) for w in wheels]}")

    stubs = [hidden_stub(w) for w in wheels]
    merged = trimesh.boolean.union([body] + wheels + stubs, engine="manifold")
    merged = max(merged.split(only_watertight=False), key=lambda c: len(c.faces))
    print(f"объединение: {len(merged.faces)} граней, "
          f"watertight={merged.is_watertight}, деталей=1")

    merged = smooth(merged, 2, lam=0.25, mu=-0.25, selective=True, floor=0.4, k=4.0)
    if len(merged.faces) > final_faces:
        merged = largest_part(merged.simplify_quadric_decimation(face_count=final_faces))
    merged = largest_part(merged)

    vol_delta = (merged.volume / scene.volume - 1.0) * 100.0
    print(f"итог: {len(merged.faces)} граней / {len(merged.vertices)} вершин, "
          f"watertight={merged.is_watertight}, деталей=1, объём {vol_delta:+.2f}%")
    print(f"габариты {np.round(merged.extents, 2)} (было {np.round(scene.extents, 2)})")

    merged.export(dst_path)
    print(f"сохранено: {dst_path}  ({time.time() - t0:.1f} c)")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--body-faces", type=int, default=62000)
    ap.add_argument("--wheel-faces", type=int, default=6200)
    ap.add_argument("--final-faces", type=int, default=88000)
    a = ap.parse_args()
    optimize(a.src, a.dst, a.body_faces, a.wheel_faces, a.final_faces)


if __name__ == "__main__":
    main()
