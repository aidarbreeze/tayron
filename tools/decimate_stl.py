#!/usr/bin/env python3
"""Облегчение готовой модели: квадрическая децимация с сохранением рельефа.

Зачем отдельный шаг, а не ещё один прогон конвейера с меньшим бюджетом:
при шаге сетки ~2 юнита гравировка (штрихи 1–3 юнита) ложится на «волну»
базовых фасеток и тонет в ней. Децимация уже готовой плотной модели,
наоборот, режет ровные панели и оставляет рёбра рельефа (метрика ошибки
квадрик держится за кривизну) — фары и фонари на 100 000 граней читаются.

    python3 tools/decimate_stl.py tayron_2023_optimized_hires.stl \\
        tayron_2023_optimized_lite.stl --faces 100000
"""
import argparse
import time

import numpy as np
import trimesh
import fast_simplification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--faces", type=int, default=100000,
                    help="целевое число граней (100 000 ≈ 5.0 МБ в двоичном STL)")
    a = ap.parse_args()

    t0 = time.time()
    mesh = trimesh.load(a.src, force="mesh")
    vol0 = mesh.volume
    v, f = fast_simplification.simplify(
        mesh.vertices.astype(np.float32), mesh.faces.astype(np.int32),
        target_count=a.faces)
    out = trimesh.Trimesh(vertices=v.astype(np.float64), faces=f, process=True)
    out.merge_vertices()
    # децимация иногда оставляет 2-гранные «иголки» нулевого объёма: они делают
    # меш незамкнутым и разваливают его на «детали» — выкидываем всё, кроме
    # основной оболочки, и завариваем дырки
    parts = out.split(only_watertight=False)
    if len(parts) > 1:
        parts.sort(key=lambda p: len(p.faces), reverse=True)
        keep = parts[0]
        for p in parts[1:]:
            if len(p.faces) < 50 and abs(p.volume) < 1e-6:
                continue
            keep = trimesh.util.concatenate([keep, p])
        out = keep
        trimesh.repair.fill_holes(out)
        trimesh.repair.fix_normals(out)
        out.merge_vertices()
    out.export(a.dst)

    print(f"{len(mesh.faces)} -> {len(out.faces)} граней, "
          f"watertight={out.is_watertight}, деталей={len(out.split(only_watertight=False))}")
    print(f"объём {vol0:.0f} -> {out.volume:.0f} ({100 * (out.volume / vol0 - 1):+.3f} %), "
          f"габариты {np.round(out.extents, 2)}")
    print(f"сохранено: {a.dst}  ({time.time() - t0:.1f} c)")


if __name__ == "__main__":
    main()
