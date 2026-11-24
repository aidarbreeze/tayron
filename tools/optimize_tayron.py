#!/usr/bin/env python3
"""
Оптимизация STL-модели Tayron 2023 (FAW-VW 1.4 TSI, Luxury Plus Advanced)
по образцу эталона Tiguan + акценты по фото реального автомобиля.

Конвейер:
  1. разбор на кузов + 4 колеса;
  2. селективное сглаживание Таубина (снимает скан-«крупку», сохраняет детали);
  3. крыша: гармоническое заполнение между рейлингами — панорамная панель и все
     её швы убираются, крыша становится сплошной, как у Tiguan;
  4. акценты по фото (рельеф смещением вершин, топология не меняется):
       • фары — чёткий контур, «линза» с углублением, проектор, хром-бровь;
       • задние фонари — утопленный блок с швом-контуром, красная линия через
         корму, шеврон и красная полоса на крыле (координаты сняты с фото);
       • бампер Luxury Plus — тёмная вставка с линиями по верху и низу,
         серебристая накладка-«крыло» и полоса по низу; корпус ДХО берётся из
         базовой модели (он уже вылеплен и совпадает с фото);
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


def refine_faces(mesh, face_idx, levels=1):
    """Локальное уплотнение выбранных треугольников без T-стыков.

    Обычный subdivide(face_index=...) сшивает только внутри зоны и оставляет
    щели по её границе (меш перестаёт быть замкнутым). Здесь применяется
    red-green схема: у выбранных граней дробятся все три ребра, у соседних —
    только те, что общие с выбранными, поэтому новых «висячих» рёбер нет.
    """
    m = mesh
    for _ in range(levels):
        V = m.vertices.astype(np.float64)
        F = m.faces
        sel = np.zeros(len(F), bool)
        sel[np.asarray(face_idx, dtype=int)] = True
        if not sel.any():
            return m
        # рёбра: (nF, 3) -> уникальные
        e = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
        ue, inv = np.unique(e, axis=0, return_inverse=True)
        eid = inv.reshape(-1, 3)
        marked = np.zeros(len(ue), bool)
        marked[eid[sel].ravel()] = True
        mid_index = -np.ones(len(ue), dtype=np.int64)
        ids = np.where(marked)[0]
        mids = 0.5 * (V[ue[ids, 0]] + V[ue[ids, 1]])
        mid_index[ids] = np.arange(len(V), len(V) + len(ids))
        V2 = np.vstack([V, mids]) if len(ids) else V

        newf = []
        for face, e3 in zip(F, eid):
            mk = marked[e3]
            k = int(mk.sum())
            if k == 0:
                newf.append(face)
                continue
            # поворачиваем так, чтобы отмеченное ребро(а) шли подряд
            i0, i1, i2 = face
            m01 = mid_index[e3[0]]; m12 = mid_index[e3[1]]; m20 = mid_index[e3[2]]
            if k == 3:
                newf += [(i0, m01, m20), (i1, m12, m01), (i2, m20, m12), (m01, m12, m20)]
            elif k == 1:
                if mk[0]:   # ребро 0-1
                    newf += [(i0, m01, i2), (m01, i1, i2)]
                elif mk[1]:  # ребро 1-2
                    newf += [(i1, m12, i0), (m12, i2, i0)]
                else:        # ребро 2-0
                    newf += [(i2, m20, i1), (m20, i0, i1)]
            else:
                if not mk[2]:    # рёбра 0-1 и 1-2 -> угол в i1
                    newf += [(i1, m12, m01), (i0, m01, m12), (i0, m12, i2)]
                elif not mk[0]:  # рёбра 1-2 и 2-0 -> угол в i2
                    newf += [(i2, m20, m12), (i1, m12, m20), (i1, m20, i0)]
                else:            # рёбра 2-0 и 0-1 -> угол в i0
                    newf += [(i0, m01, m20), (i2, m20, m01), (i2, m01, i1)]
        m = trimesh.Trimesh(vertices=V2, faces=np.array(newf, dtype=np.int64), process=False)
    return m


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


def apply_features(mesh, features, region, mirror=True, flip_normal=False, normal=None):
    """Смещение вершины по нормали; u=|Y| (или Y), v=Z.

    normal задаёт единое направление смещения вместо по-вершинных нормалей:
    для почти плоской панели (корма) это единственный способ получить ровный
    рельеф — по-вершинные нормали на плотной сетке дают «рябь» из фасеток.
    """
    v = mesh.vertices.astype(np.float64)
    if normal is None:
        n = mesh.vertex_normals
    else:
        n = np.tile(np.asarray(normal, dtype=np.float64), (len(v), 1))
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
        outline(lamp, width=1.6, depth=2.40),          # чёткий контур фары
        dish(lamp, depth=0.80, edge=2.2),              # стекло утоплено
        disk((136, -8.0), 7.5, depth=1.1, edge=3.0),   # проектор
        polyline([(112, -12.6), (150, -6.0), (157, -4.5)], width=1.2, depth=1.2),  # DRL
        polyline([(110, -2.0), (150, 1.0), (158, 2.2)], width=1.4, depth=-1.0),    # хром-бровь
    ]
    return feats


def rear_features():
    """Фонари Tayron 2023 — рисунок по фото, собранный из примитивов.

    Координаты сняты с фотографии кормы: u = (px − 1157.5)·0.3266,
    v = (1164 − py)/3.102. Фотомаска красного рисунка тоньше шага сетки
    (≈0.9 юнита) и давала «рябь», поэтому тот же рисунок перерисован по
    измерениям: утопленный блок фонаря с тонким швом-контуром (как у Тигуана),
    сквозная красная линия через корму, крупный шеврон и вертикальная
    красная полоса на крыле. Штрихи взяты шириной 3.5–5 юнитов — только такие
    сетка ещё держит без фасеток.
    """
    lamp = [(24, 42.0), (24, 55.0), (95, 57.5), (150, 58.5), (167, 55.0),
            (170, 45.0), (166, 41.5)]
    feats = [
        dish(lamp, depth=0.90, edge=1.2),       # блок фонаря
        outline(lamp, width=1.7, depth=1.30),   # тонкий шов-контур по краю
        polyline([(20.0, 43.0), (166.0, 43.0)], width=4.0, depth=1.35),   # красная линия
        polyline([(146.0, 44.5), (158.5, 51.0), (146.0, 57.5)],
                 width=3.4, depth=1.20),        # крупный шеврон
        capsule((165.0, 42.5), (165.0, 57.0), width=4.6, depth=1.35),     # красная полоса
    ]
    return feats


def bumper_features():
    """Бампер Luxury Plus по фото (пиксели фото → u=|Y|, v=Z):
    u = (1196 − px)·0.247, v = −46 − (py − 1120)·0.2435.

    Корпус ДХО в базовой модели уже вылеплен (ободок + линза + диагональный
    световод), и он почти совпадает с фото — поэтому здесь он НЕ перерисовывается:
    второй контур поверх первого давал «двойную» мешанину и ромбы.
    Добавляются только элементы, которых в базе нет: тёмная вставка-панель
    с тонкими линиями по верху и низу, серебристая накладка-«крыло» под
    нижней решёткой и серебристая полоса по низу бампера.
    """
    insert = [(18, -46), (150, -46), (166, -58), (166, -104), (150, -114),
              (-150, -114), (-166, -104), (-166, -58), (-150, -46), (-18, -46)]
    silver = [(22, -101), (105, -108), (140, -101), (150, -108), (110, -114), (22, -107)]
    bottom = [(150, -116), (172, -116), (176, -124), (-176, -124), (-170, -116), (-150, -116)]
    feats = [
        dish(insert, depth=0.25, edge=1.5),      # чёрная вставка — едва утоплена
        polyline([(-150, -46), (150, -46)], width=1.6, depth=0.70),   # верхняя линия
        polyline([(-150, -114.5), (150, -114.5)], width=1.6, depth=0.70),  # нижняя линия
        dish(silver, depth=-0.30, edge=1.2),     # серебристая накладка-«крыло»
        dish(bottom, depth=-0.30, edge=1.2),     # серебристая полоса по низу
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


def optimize(src_path, dst_path, body_faces=62000, wheel_faces=6200, final_faces=340000,
             detail_faces=330000, rear_refine=3,
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
        # локальное уплотнение там, где нужны тонкие штрихи: фонари (×2) и фары (×1)
        c = body.triangles_center
        zr = (c[:, 0] < -392) & (np.abs(c[:, 1]) < 182) & (c[:, 2] > 34) & (c[:, 2] < 66)
        body = refine_faces(body, np.where(zr)[0], levels=rear_refine)
        c = body.triangles_center
        zf = ((c[:, 0] > 372) & (np.abs(c[:, 1]) > 95) & (np.abs(c[:, 1]) < 186)
              & (c[:, 2] > -24) & (c[:, 2] < 14))
        body = refine_faces(body, np.where(zf)[0], levels=1)
        # выравниваем только те панели, куда ляжет гравировка — иначе фасетки
        # глушат рельеф, а сглаживание всей кормы даёт гребешки по кромкам
        vb = body.vertices
        soft = (np.abs(vb[:, 1]) < 178) & (
            ((vb[:, 0] < -390) & (vb[:, 2] > 25) & (vb[:, 2] < 88)) |
            ((vb[:, 0] > 360) & (vb[:, 2] > -104) & (vb[:, 2] < 38)))
        body = smooth_region(body, soft, iters=6, lam=0.45, mu=-0.48, feather=12)
        v = body.vertices
        n = body.vertex_normals
        front_region = (v[:, 0] > 370) & (n[:, 0] > 0.15)
        rear_region = (v[:, 0] < -370) & (n[:, 0] < -0.15)
        body, m1 = apply_features(body, front_features(), front_region, mirror=True)
        body, m2 = apply_features(body, rear_features(), rear_region, mirror=True,
                                  normal=(-1.0, 0.0, 0.0))
        body, m3 = apply_features(body, bumper_features(), front_region, mirror=True)
        # после гравировки гасим фасеточную «рябь» по самому рельефу: Таубин
        # с малым весом снимает высокочастотные ступеньки, глубины остаются
        vg = body.vertices
        etched_front = ((np.abs(vg[:, 1]) < 178) & (vg[:, 0] > 356)
                        & (vg[:, 2] > -104) & (vg[:, 2] < -34))
        etched_rear = ((np.abs(vg[:, 1]) < 178) & (vg[:, 0] < -390)
                       & (vg[:, 2] > 36) & (vg[:, 2] < 64))
        # после гравировки гасим фасеточную «рябь»: Таубин снимает высокочастотные
        # ступеньки сетки, а штрихи шириной 3.5–5 юнитов остаются
        body = smooth_region(body, etched_front, iters=2, lam=0.32, mu=-0.34, feather=4)
        body = smooth_region(body, etched_rear, iters=5, lam=0.45, mu=-0.47, feather=5)
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
    ap.add_argument("--final-faces", type=int, default=340000)
    ap.add_argument("--detail-faces", type=int, default=330000)
    ap.add_argument("--rear-refine", type=int, default=3,
                    help="уровни уплотнения полосы задних фонарей (2–4)")
    ap.add_argument("--no-roof", action="store_true")
    ap.add_argument("--no-details", action="store_true")
    a = ap.parse_args()
    optimize(a.src, a.dst, a.body_faces, a.wheel_faces, a.final_faces, a.detail_faces,
             rear_refine=a.rear_refine,
             with_roof=not a.no_roof, with_details=not a.no_details)


if __name__ == "__main__":
    main()
