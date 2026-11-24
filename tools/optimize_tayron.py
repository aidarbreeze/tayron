#!/usr/bin/env python3
"""
Оптимизация STL-модели Tayron 2023 (FAW-VW 1.4 TSI, Luxury Plus Advanced)
по образцу эталона Tiguan + акценты по фото реального автомобиля.

Конвейер:
  1. разбор на кузов + 4 колеса;
  2. селективное сглаживание Таубина (снимает скан-«крупку», сохраняет детали);
  3. крыша: гармоническое заполнение между рейлингами — панорамная панель и все
     её швы убираются, крыша становится сплошной, как у Tiguan;
  4. акценты по фото (гравировка по нормалям, неизменная топология):
       • фары — чёткий контур, «линза» с углублением, проектор, хром-бровь;
       • задние фонари — контур блока, углублённая «линза», шевроны-треугольники
         (мотив X), тонкая полоса фонарей через всю корму;
       • бампер Luxury Plus — контуры противотуманных блоков и серебристые
         «X»-лучи по нижнему бамперу;
  5. децимация до бюджета граней Tiguan с защитой зон акцентов;
  6. скрытые «цапфы» в арках + булево объединение -> один водонепроницаемый корпус;
  7. экспорт STL в исходной системе координат.

Зависимости:
    pip install --break-system-packages numpy scipy trimesh fast-simplification manifold3d matplotlib

Пример:
    python3 tools/optimize_tayron.py "tayron lite 2 (1) (1).stl" tayron_2023_optimized.stl
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import trimesh
from matplotlib.path import Path
from scipy import sparse
from scipy.sparse.linalg import spsolve
from trimesh.remesh import subdivide


# ─────────────────────────── базовые утилиты сетки ───────────────────────────

def adjacency_average(mesh):
    """Строково-нормированная матрица усреднения по 1-кольцу соседей."""
    v, f = mesh.vertices, mesh.faces
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    row = np.concatenate([e[:, 0], e[:, 1]])
    col = np.concatenate([e[:, 1], e[:, 0]])
    a = sparse.csr_matrix((np.ones(len(row)), (row, col)), shape=(len(v), len(v)))
    return sparse.diags(1.0 / np.asarray(a.sum(1)).ravel()) @ a


def smooth(mesh, iters, lam=0.5, mu=-0.53, selective=True, floor=0.15, k=3.0, weights=None):
    """Сглаживание Таубина; selective — сила зависит от локальной шероховатости."""
    v = mesh.vertices.astype(np.float64).copy()
    w_neigh = adjacency_average(mesh)
    if weights is not None:
        w = weights[:, None]
    elif selective:
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


def smooth_region(mesh, region, iters=8, lam=0.55, mu=-0.58, feather=6):
    """Локальное сглаживание (Таубин) только внутри region с плавным краем.

    Нужно перед гравировкой: после децимации плоские панели (корма) покрыты
    фасетками амплитудой 1–2 юнита, и мелкий рельеф в них тонет.
    """
    mesh = mesh.copy()
    mesh.merge_vertices()          # после subdivide(process=False) есть дубли вершин
    f = mesh.faces
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]),
                          axis=1), axis=0)
    n = len(mesh.vertices)
    a = sparse.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    a = a + a.T
    deg = np.asarray(a.sum(1)).ravel()
    deg[deg == 0] = 1.0
    w = region.astype(np.float64)
    # плавный край маски
    for _ in range(feather):
        w = 0.5 * w + 0.5 * (a @ w) / deg
    v = mesh.vertices.astype(np.float64)
    for _ in range(iters):
        v = v + lam * w[:, None] * ((a @ v) / deg[:, None] - v)
        v = v + mu * w[:, None] * ((a @ v) / deg[:, None] - v)
    out = mesh.copy()
    out.vertices = v
    return out


def largest_part(mesh):
    return max(mesh.split(only_watertight=False), key=lambda c: len(c.faces))


def decimate_safe(mesh, target, step=0.75):
    """Децимация ступенями с проверкой замкнутости на каждом шаге.

    fast_simplification при сильном сокращении теряет грани и оставляет дырки;
    если шаг сделал меш незамкнутым, пробуем более мягкий шаг, а если и он
    не выходит — останавливаемся на текущем (лучше больше граней, чем дырки).
    """
    m = mesh
    while len(m.faces) > target:
        nxt = max(target, int(len(m.faces) * step))
        done = None
        for s in (nxt, int(len(m.faces) * 0.85), int(len(m.faces) * 0.93), len(m.faces) - 1):
            if s >= len(m.faces):
                continue
            cand = clean_volume(largest_part(m.simplify_quadric_decimation(face_count=s)))
            if cand.is_watertight:
                done = cand
                break
        if done is None:
            break
        m = done
    return m


def clean_volume(mesh):
    """Крупнейшая деталь + заварка дырок, оставшихся от децимации."""
    m = largest_part(mesh)
    if not m.is_watertight:
        m.merge_vertices()
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
    return m


def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ─────────────────────────────── крыша ───────────────────────────────

def fix_roof(mesh, x_rear=-335, x_front=45, y_half=92, z_min=140, verbose=True):
    """Гармоническое заполнение крыши между рейлингами: панорамная панель,
    её рамка и все швы исчезают, поверхность остаётся естественно выпуклой."""
    v = mesh.vertices.astype(np.float64)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    core = (ss(x_rear, x_rear + 25, x) * (1 - ss(x_front - 15, x_front + 15, x)) *
            (1 - ss(y_half - 10, y_half + 4, np.abs(y))) * ss(z_min, z_min + 12, z))
    region = (ss(x_rear - 10, x_rear + 15, x) * (1 - ss(x_front, x_front + 30, x)) *
              (1 - ss(y_half - 16, y_half + 8, np.abs(y))) * ss(z_min - 12, z_min + 10, z))

    f = mesh.faces
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    a = sparse.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v)))
    a = a + a.T
    lap = (sparse.diags(np.asarray(a.sum(1)).ravel()) - a).tocsr()

    idx = region > 0.02
    z_new = z.copy()
    z_new[idx] = spsolve(lap[idx][:, idx].tocsc(), -lap[idx][:, ~idx] @ z[~idx])
    z_out = z + core * (z_new - z)

    out = mesh.copy()
    out.vertices = np.column_stack([x, y, z_out])
    if verbose:
        moved = np.abs(z_out - z)[core > 0.8]
        print(f"  крыша: сдвиг медиана {np.median(moved):.2f}, макс {np.abs(z_out - z).max():.2f}")
    return out


# ──────────────────────── 2D-примитивы для гравировки ────────────────────────

def _poly_sdf(u, v, poly):
    p = np.asarray(poly, float)
    pts = np.column_stack([u, v])
    inside = Path(p).contains_points(pts)
    best = np.full(len(pts), np.inf)
    for i in range(len(p)):
        a_p, b_p = p[i], p[(i + 1) % len(p)]
        ab = b_p - a_p
        l2 = ab @ ab
        if l2 == 0:
            continue
        t = np.clip(((pts - a_p) @ ab) / l2, 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(pts - (a_p + t[:, None] * ab), axis=1))
    return best, inside


def outline(poly, width=4.0, depth=2.2):
    """Канавка по контуру полигона."""
    def f(u, v):
        d, _ = _poly_sdf(u, v, poly)
        return depth * (1 - ss(0, width, d))
    return f


def dish(poly, depth=1.2, edge=4.0):
    """Углубление внутри полигона (со сглаженным краем)."""
    def f(u, v):
        d, ins = _poly_sdf(u, v, poly)
        return depth * ins * ss(0, edge, d)
    return f


def capsule(p0, p1, width=4.0, depth=2.0):
    """Канавка/валик вдоль отрезка (depth<0 — наружу)."""
    a_p = np.array(p0, float)
    b_p = np.array(p1, float)
    ab = b_p - a_p
    l2 = ab @ ab

    def f(u, v):
        pts = np.column_stack([u, v])
        t = np.clip(((pts - a_p) @ ab) / l2, 0.0, 1.0)
        d = np.linalg.norm(pts - (a_p + t[:, None] * ab), axis=1)
        return depth * (1 - ss(0, width, d))
    return f


def polyline(points, width=4.0, depth=2.0):
    """Канавка вдоль ломаной."""
    segs = [capsule(points[i], points[i + 1], width, depth) for i in range(len(points) - 1)]

    def f(u, v):
        return np.maximum.reduce([s(u, v) for s in segs]) if len(segs) > 1 else segs[0](u, v)
    return f


def disk(center, radius, depth=1.2, edge=3.5):
    c = np.array(center, float)

    def f(u, v):
        return depth * (1 - ss(radius, radius + edge, np.hypot(u - c[0], v - c[1])))
    return f


def chevron(apex, base_top, base_bottom, depth=2.0, width=3.5):
    """V-образная канавка (шеврон) — из двух отрезков."""
    return polyline([base_top, apex, base_bottom], width=width, depth=depth)


def apply_features(mesh, features, region, mirror=True, flip_normal=False):
    """Смещение вершины по нормали; u=|Y| (или Y), v=Z."""
    v = mesh.vertices.astype(np.float64)
    n = mesh.vertex_normals
    u = np.abs(v[:, 1]) if mirror else v[:, 1]
    vv = v[:, 2]
    disp = np.zeros(len(v))
    for f in features:
        disp += f(u, vv)
    mask = region & (disp != 0)
    out = mesh.copy()
    out.vertices = v - n * (disp * mask)[:, None]
    return out, mask


# ─────────────────────── фигуры под Tayron 2023 ───────────────────────

def front_features():
    """Фара 1.4 Luxury Plus: широкий клин, высокий у наружного края.
    Глубины 0.4–0.9 юнита (2–4.4 мм при длине модели 900), иначе рельеф «плывёт»."""
    lamp = [(109, -4.1), (132, -0.8), (151, 4.4), (158, 6.3), (161, 7.3),
            (159, -10.5), (144, -14.7), (124, -16.8), (109, -14.7)]
    feats = [
        outline(lamp, width=2.2, depth=2.40),          # чёткий контур фары
        dish(lamp, depth=0.80, edge=3.5),              # стекло утоплено
        disk((136, -8.0), 7.5, depth=1.1, edge=3.0),   # проектор
        polyline([(112, -12.6), (150, -6.0), (157, -4.5)], width=1.8, depth=1.2),  # DRL
        polyline([(110, -2.0), (150, 1.0), (158, 2.2)], width=2.0, depth=-1.0),    # хром-бровь
    ]
    return feats


def rear_features():
    """Фонари Tayron по образцу Тигуана: узкая утопленная полоса через всю
    корму с чётким уступом-контуром (по фото: Z 40–59, |Y| до 168) и
    X-световоды «<» внутри неё — крупные, по два на сторону.

    Глубина полосы 1.8 юнита (≈9 мм), стенка 1.2 юнита; шевроны 2.0 юнита
    шириной 3 — на сетке 3–4 юнита это самые крупные штрихи, которые
    держат форму и не рассыпаются в фасетки."""
    band = [(12, 41.5), (60, 40.5), (110, 40), (150, 40.5), (165, 42.5),
            (170, 48), (167, 55), (152, 58.5), (110, 59), (60, 59), (12, 58)]
    feats = [
        dish(band, depth=1.80, edge=1.2),          # полоса: контур-уступ
        # световоды «<»: пара у шильда и пара у края
        chevron((40, 49.5), (56, 57.0), (56, 42.0), depth=2.00, width=3.0),
        chevron((76, 49.5), (92, 57.0), (92, 42.0), depth=2.00, width=3.0),
        chevron((118, 49.5), (134, 57.0), (134, 42.0), depth=2.00, width=3.0),
        chevron((148, 49.5), (160, 56.0), (160, 43.0), depth=2.00, width=2.8),
        # светящаяся линия вдоль нижней кромки полосы
        polyline([(12, 43.0), (60, 42.5), (110, 42.2), (158, 43.0)],
                 width=1.6, depth=1.30),
    ]
    return feats


def bumper_features():
    """Бампер Luxury Plus: контуры ПТФ и серебристые X-лучи (глубина ≤0.8 юнита)."""
    fog = [(133, -47), (160, -46), (170, -56), (166, -72), (140, -79), (129, -70)]
    feats = [
        outline(fog, width=2.4, depth=2.00),
        dish(fog, depth=0.55, edge=3.5),
        # X-лучи: из-под номерного знака к ПТФ и к нижним углам бампера
        capsule((42, -70), (144, -50), width=6.0, depth=-0.90),
        capsule((42, -70), (146, -94), width=6.0, depth=-0.90),
    ]
    return feats


# ───────────────────────────── сборка ─────────────────────────────

def hidden_stub(wheel, reach=52.0, start=18.0, radius_frac=0.34, sections=48):
    c, ext = wheel.center_mass, wheel.extents
    inward = 1.0 if c[1] < 0 else -1.0
    y0, y1 = c[1] + inward * start, c[1] + inward * reach
    cyl = trimesh.creation.cylinder(radius=min(ext[0], ext[2]) * radius_frac,
                                    height=abs(y1 - y0), sections=sections)
    cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    cyl.apply_translation([c[0], (y0 + y1) / 2, c[2]])
    return cyl


def optimize(src_path, dst_path, body_faces=62000, wheel_faces=6200, final_faces=170000,
             detail_faces=150000,
             with_roof=True, with_details=True, verbose=True):
    t0 = time.time()
    scene = trimesh.load(src_path, force="mesh")
    parts = sorted(scene.split(only_watertight=False), key=lambda c: -len(c.faces))
    body, wheels = parts[0], parts[1:]
    if verbose:
        print(f"вход: {len(scene.faces)} граней, {len(parts)} деталей")

    # 1. сглаживание кузова
    body = smooth(body, 40, selective=True, floor=0.15, k=3.0)
    if verbose:
        print(f"кузов сглажен: {len(body.faces)} граней")

    # 2. крыша
    if with_roof:
        body = fix_roof(body, verbose=verbose)

    # 3. чистовая децимация кузова
    body = decimate_safe(body, body_faces)
    body = smooth(body, 3, selective=True, floor=0.35, k=3.5)
    if len(body.faces) > body_faces:
        body = decimate_safe(body, body_faces)

    # 4. акценты: уплотняем сетку, гравируем, возвращаем бюджет
    if with_details:
        body = largest_part(trimesh.Trimesh(*subdivide(body.vertices, body.faces), process=False))
        v = body.vertices
        n = body.vertex_normals
        front_region = (v[:, 0] > 370) & (n[:, 0] > 0.15)
        rear_region = (v[:, 0] < -370) & (n[:, 0] < -0.15)
        # выравниваем только те панели, куда ляжет гравировка — иначе фасетки
        # глушат рельеф, а сглаживание всей кормы даёт гребешки по кромкам
        vb = body.vertices
        soft = (np.abs(vb[:, 1]) < 178) & (
            ((vb[:, 0] < -390) & (vb[:, 2] > 25) & (vb[:, 2] < 88)) |
            ((vb[:, 0] > 370) & (vb[:, 2] > -92) & (vb[:, 2] < 38)))
        body = smooth_region(body, soft, iters=6, lam=0.45, mu=-0.48, feather=12)
        v = body.vertices
        n = body.vertex_normals
        front_region = (v[:, 0] > 370) & (n[:, 0] > 0.15)
        rear_region = (v[:, 0] < -370) & (n[:, 0] < -0.15)
        body, m1 = apply_features(body, front_features(), front_region, mirror=True)
        body, m2 = apply_features(body, rear_features(), rear_region, mirror=True)
        body, m3 = apply_features(body, bumper_features(), front_region, mirror=True)
        body = decimate_safe(body, detail_faces)
        if verbose:
            print(f"акценты: фары {int(m1.sum())}, фонари {int(m2.sum())}, бампер {int(m3.sum())} вершин")
    if verbose:
        print(f"кузов готов: {len(body.faces)} граней, watertight={body.is_watertight}")

    # 5. колёса
    wheels = [largest_part(smooth(w, 25, selective=True, floor=0.35, k=4.0)
                           .simplify_quadric_decimation(face_count=wheel_faces))
              for w in wheels]
    wheels = [largest_part(smooth(w, 3, selective=False).simplify_quadric_decimation(face_count=wheel_faces))
              for w in wheels]

    # 6. объединение
    stubs = [hidden_stub(w) for w in wheels]
    merged = trimesh.boolean.union([body] + wheels + stubs, engine="manifold")
    merged = max(merged.split(only_watertight=False), key=lambda c: len(c.faces))
    merged = smooth(merged, 2, lam=0.25, mu=-0.25, selective=True, floor=0.4, k=4.0)
    if len(merged.faces) > final_faces:
        merged = decimate_safe(merged, final_faces)

    if verbose:
        vol = (merged.volume / scene.volume - 1.0) * 100.0
        print(f"итог: {len(merged.faces)} граней / {len(merged.vertices)} вершин, "
              f"деталей=1, watertight={merged.is_watertight}, объём {vol:+.2f}%")
        print(f"габариты {np.round(merged.extents, 2)} (было {np.round(scene.extents, 2)})")
    merged.export(dst_path)
    if verbose:
        print(f"сохранено: {dst_path}  ({time.time() - t0:.1f} c)")
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--body-faces", type=int, default=62000)
    ap.add_argument("--wheel-faces", type=int, default=6200)
    ap.add_argument("--final-faces", type=int, default=170000)
    ap.add_argument("--detail-faces", type=int, default=150000)
    ap.add_argument("--no-roof", action="store_true")
    ap.add_argument("--no-details", action="store_true")
    a = ap.parse_args()
    optimize(a.src, a.dst, a.body_faces, a.wheel_faces, a.final_faces, a.detail_faces,
             with_roof=not a.no_roof, with_details=not a.no_details)


if __name__ == "__main__":
    main()
