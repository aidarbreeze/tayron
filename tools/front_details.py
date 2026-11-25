#!/usr/bin/env python3
"""Передок из фотографии: решётка с сотами и полосами, эмблема, накладки ПТФ, воздухозаборник.

Зачем: в скане решётка — крупные «блины» вместо сотовой сетки, эмблема — купол,
накладки ПТФ — бугры. Форму взять не из чего (фотограмметрия мелкий рельеф
теряет), поэтому детали строятся заново по разметке с референсного фото, а
поверхность скана под ними выравнивается и утапливается в плоскость панели —
старая каша не торчит сквозь новую геометрию.

Всё ставится «по панели»: для каждой зоны по вершинам скана подгоняется
плоскость X = aY + bZ + c, детали строятся в её плоскости и сдвигаются по X по
её наклону — поэтому они сидят на кузове, а не висят в воздухе и не режут его.

    python3 tools/front_details.py --in tayron_2023_optimized_lite.stl \\
        --out tayron_2023_optimized_v5.stl
"""
import argparse
import os
import sys
from functools import lru_cache

import numpy as np
import trimesh
from scipy import ndimage as ndi
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_vs_photo as A                      # noqa: E402
import photo_texture as P                       # noqa: E402

ROOT = A.ROOT
VIEW = 'front'
BOX = A.BBOX[VIEW]                              # кадр кропа: x0, x1, y0, y1
W = int(BOX[1] - BOX[0])                        # ширина кадра, px
H = int(BOX[3] - BOX[2])                        # высота кадра, px

# ---------------------------------------------------------------- разметка
GRILLE = (458, 1008, 548, 734)                  # проём решётки (кадр, px)
GRILLE_BARS = [(578, 591), (619, 631)]          # два хромовых прута поперёк
GRILLE_LIP = (712, 732)                         # нижняя хромовая накладка
EMBLEM = (751, 632, 57.0)                       # центр и внешний радиус, px
PTF = [(112, 822, 358, 1006), (1124, 822, 1370, 1006)]   # окна накладок ПТФ
PTF_STRIP = [(268, 964, 360, 980), (1166, 964, 1214, 982)]   # хром-полоска ПТФ
INTAKE = (250, 1252, 878, 1014)                 # нижний воздухозаборник (x0,x1,y0,y1)
INTAKE_DARK = 70                                # порог тёмного в кадре, 0..255
INTAKE_FINS = [(984, 994)]                      # горизонтальные рёбра в нише, строки
TRIM_BRIGHT = 75                                # порог серебра обвеса
PLATE_HOLE = (505, 795, 1035, 912)              # площадка номера — не обвес

# ---------------------------------------------------------------- параметры
RECESS = 4.2                                    # утопление скана под деталями
TAPER = 24.0                                    # плавный край утопления, px
BAR_OUT = 2.4                                   # выступ прутов над дном, юниты
LIP_OUT = 2.0
CELL_PX = 74.0                                  # шаг сот по кадру, px
CELL_H = 46.0                                   # высота ряда сот, px
CELL_WALL = 3.6                                 # толщина ребра, px
EMB_OUT = 2.8                                   # выступ эмблемы
TRIM_OUT = 1.6                                  # выступ серебра обвеса


# ---------------------------------------------------------------- кадр → модель
def frame_maps(mesh):
    u_of, v_of, sx, sy = A.frame(mesh, VIEW)
    u, v, _ = A.project(mesh, VIEW)
    u_max, v_min = u.max(), v.min()

    def yz(px, py):
        return u_max - np.asarray(px) / sx, v_min + (H - np.asarray(py)) / sy
    return yz, sx, sy, u_max, v_min


def plane_fit(mesh, sel_mask, front_only=True):
    """X = aY + bZ + c по передним вершинам скана в зоне."""
    px, py = _px_py(mesh)
    ix = np.clip(px.astype(int), 0, W - 1)
    iy = np.clip(py.astype(int), 0, H - 1)
    sel = np.asarray(sel_mask, bool)[iy, ix]
    if front_only:
        sel &= mesh.vertex_normals[:, 0] > 0.55
    V = mesh.vertices[sel]
    if len(V) < 20:
        raise RuntimeError(f'в зоне мало вершин ({len(V)})')
    M = np.stack([V[:, 1], V[:, 2], np.ones(len(V))], 1)
    coef, *_ = np.linalg.lstsq(M, V[:, 0], rcond=None)
    # малые и кривые зоны (углы бампера) — плоской плитой на средней глубине
    ys = np.where(np.asarray(sel_mask, bool).any(0))[0]
    if len(ys) and (ys.max() - ys.min()) < 320:
        coef = np.array([0.0, 0.0, float(np.median(V[:, 0]))])
    coef[0] = float(np.clip(coef[0], -0.45, 0.45))
    coef[1] = float(np.clip(coef[1], -0.45, 0.45))
    # после обрезки наклонов пересчитать смещение, иначе плоскость уезжает
    coef[2] = float(np.median(V[:, 0] - coef[0] * V[:, 1] - coef[1] * V[:, 2]))
    return coef


def _px_py(mesh):
    """Вершины → пиксели кадра. Считается заново: кэш по id(mesh) отдавал
    пиксели старого меша после каждого пересоздания."""
    _, sx, sy, u_max, v_min = frame_maps(mesh)
    u, v, _ = A.project(mesh, VIEW)
    return (u_max - u) * sx, H - (v - v_min) * sy


# ---------------------------------------------------------------- полигоны кадра
def win_mask(win):
    x0, x1, y0, y1 = win
    m = np.zeros((H, W), bool)
    m[int(y0):int(y1), int(x0):int(x1)] = True
    return m


def pbox(x0, y0, x1, y1):
    return box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def disc(cx, cy, r, n=64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))


def line(pts, w):
    return LineString([(float(x), float(y)) for x, y in pts]).buffer(w / 2, cap_style=2)


@lru_cache(maxsize=None)
def gray_photo():
    """Фото в градациях серого (кэш один на процесс, меша не касается)."""
    return np.asarray(P.photo_crop(VIEW, A.BBOX[VIEW]).convert('L'), float)


def mask_polygon(mask, simplify=1.4):
    """Маска кадра → полигон (ступеньки сглажены buffer−/+, углы подрезаны)."""
    ys = np.where(mask.any(1))[0]
    if len(ys) < 4:
        return None
    boxes = []
    for y in range(ys.min(), ys.max() + 1):
        row = np.where(mask[y])[0]
        if not len(row):
            continue
        s = p = row[0]
        for x in row[1:]:
            if x - p > 1:
                boxes.append(box(float(s), float(y), float(p) + 1, float(y) + 1))
                s = x
            p = x
        boxes.append(box(float(s), float(y), float(p) + 1, float(y) + 1))
    poly = unary_union(boxes).buffer(-1.5, join_style=1).buffer(1.5, join_style=1)
    if poly.geom_type == 'MultiPolygon':                 # оставить самые крупные куски
        parts = sorted(poly.geoms, key=lambda g: -g.area)
        poly = parts[0] if len(parts) == 1 else unary_union(
            [g for g in parts if g.area > 0.25 * parts[0].area])
    return poly.simplify(simplify)


def to_units(poly, yz):
    def ring(c):
        a = np.asarray(c, float)
        Y, Z = yz(a[:, 0], a[:, 1])
        return np.stack([Y, Z], 1)
    return Polygon(ring(poly.exterior.coords), [ring(i.coords) for i in poly.interiors])


def conform(part, coef, y0, z0):
    """Сдвинуть деталь по X по наклону панели: X += a(Y−Y0) + b(Z−Z0)."""
    V = part.vertices.copy()
    V[:, 0] += coef[0] * (V[:, 1] - y0) + coef[1] * (V[:, 2] - z0)
    return trimesh.Trimesh(vertices=V, faces=part.faces, process=False)


def plate(poly_px, yz, thickness, x_off, coef):
    """Полигон кадра → плита по панели (толщина по +X)."""
    poly = to_units(poly_px, yz)
    if poly.is_empty or poly.area < 1e-6:
        return None
    m = trimesh.creation.extrude_polygon(poly, height=thickness)
    V = m.vertices.copy()
    out = np.stack([V[:, 2], V[:, 0], V[:, 1]], 1)      # (x,y,z) → (X,Y,Z)
    out[:, 0] += x_off
    part = trimesh.Trimesh(vertices=out, faces=m.faces, process=False)
    Yc, Zc = yz(EMBLEM[0], EMBLEM[1])
    part = conform(part, coef, Yc, Zc)
    if part.volume < 0:                      # зеркальная карта кадра → вывернуть
        part.invert()
    return part


def hex_ring(r, wall, thick, squash=0.62):
    """Шестиугольная ячейка из 6 брусков в плоскости yz, толщина по x."""
    boxes = []
    a = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi / 6
    pts = [(r * np.cos(t), r * np.sin(t) * squash) for t in a]
    for i in range(6):
        p0, p1 = np.array(pts[i]), np.array(pts[(i + 1) % 6])
        d = p1 - p0
        L = float(np.linalg.norm(d))
        # брусок: толщина по x, длина по y; поворот в плоскости yz
        b = trimesh.creation.box(extents=[thick, L + wall, wall])
        ang = float(np.arctan2(d[1], d[0]))
        b.apply_transform(trimesh.transformations.rotation_matrix(ang, [1, 0, 0]))
        b.apply_translation([0.0, *(p0 + p1) / 2])
        boxes.append(b)
    return trimesh.util.concatenate(boxes)


def honeycomb(yz, coef, x_off, win, keep=None, pitch=CELL_PX, cell_h=CELL_H,
              wall=CELL_WALL, thick=1.2):
    """Поле сот в проёме: ряды по кадру, ячейки режутся по маске keep."""
    x0, x1, y0, y1 = win
    u_per_px = abs(float(yz(1, 0)[0] - yz(0, 0)[0]))       # юнитов на пиксель
    r = pitch * u_per_px / 2
    wall_u = wall * u_per_px
    cells = []
    ny = int((y1 - y0) / cell_h) + 1
    nx = int((x1 - x0) / pitch) + 2
    for j in range(ny):
        for i in range(nx):
            px = x0 + i * pitch + (pitch / 2 if j % 2 else 0) + pitch / 2
            py = y0 + j * cell_h + cell_h / 2
            if not (x0 + 3 < px < x1 - 3 and y0 + 3 < py < y1 - 3):
                continue
            if keep is not None and not keep(px, py):
                continue
            c = hex_ring(r, wall_u, thick)
            Y, Z = yz(px, py)
            c.apply_translation([x_off, float(Y), float(Z)])
            cells.append(c)
    if not cells:
        return None
    part = trimesh.util.concatenate(cells)
    return conform(part, coef, *yz(EMBLEM[0], EMBLEM[1]))


def sink(mesh, mask, coef, recess, taper=TAPER):
    """Выровнять и утопить поверхность скана в зоне маски (под детали)."""
    px, py = _px_py(mesh)
    dist = ndi.distance_transform_edt(mask)
    ix = np.clip(px.astype(int), 0, W - 1)
    iy = np.clip(py.astype(int), 0, H - 1)
    t = np.clip(dist[iy, ix] / taper, 0, 1)
    w = t * t * (3 - 2 * t)
    w *= np.clip(mesh.vertex_normals[:, 0] / 0.06, 0, 1)   # не тянуть изнанку
    V = mesh.vertices.copy()
    target = coef[0] * V[:, 1] + coef[1] * V[:, 2] + coef[2] - recess
    V[:, 0] += (target - V[:, 0]) * w
    return trimesh.Trimesh(vertices=V, faces=mesh.faces, process=False), int((w > 0).sum())


def X_at(coef, yz, px, py):
    Y, Z = yz(px, py)
    return coef[0] * float(Y) + coef[1] * float(Z) + coef[2]


# ---------------------------------------------------------------- сборка
def build(mesh, verbose=True):
    yz = frame_maps(mesh)[0]
    work, parts = mesh, []

    def zone(mask, recess, name):
        nonlocal work
        coef = plane_fit(work, mask)
        work, n = sink(work, mask, coef, recess)
        if verbose:
            print(f'  {name}: X = {coef[0]:+.3f}Y {coef[1]:+.3f}Z {coef[2]:+.1f}, '
                  f'утоплено вершин {n}')
        return coef

    # ---- решётка: дно + соты + два прута + нижняя накладка
    gm = win_mask((GRILLE[0], GRILLE[1], GRILLE[2], GRILLE[3]))
    coef = zone(gm, RECESS, 'решётка')
    floor = X_at(coef, yz, EMBLEM[0], EMBLEM[1]) - RECESS

    add = lambda p: parts.append(p) if p is not None else None
    cells = honeycomb(yz, coef, floor + 0.9, GRILLE)
    add(cells)
    for by0, by1 in GRILLE_BARS:
        add(plate(pbox(GRILLE[0], by0, GRILLE[1], by1), yz, 1.8, floor + BAR_OUT, coef))
    add(plate(pbox(GRILLE[0] - 4, GRILLE_LIP[0], GRILLE[1] + 4, GRILLE_LIP[1]), yz, 2.2,
              floor + LIP_OUT, coef))

    # ---- эмблема: кольцо + VW + тёмная подложка
    ex, ey, er = EMBLEM
    ring = disc(ex, ey, er).difference(disc(ex, ey, er - 11))
    v = unary_union([line([(ex - 33, ey - 26), (ex - 5, ey + 22)], 11),
                     line([(ex + 33, ey - 26), (ex + 5, ey + 22)], 11)])
    w = unary_union([line([(ex - 40, ey - 22), (ex - 27, ey + 24)], 11),
                     line([(ex - 27, ey + 24), (ex, ey - 8)], 11),
                     line([(ex, ey - 8), (ex + 27, ey + 24)], 11),
                     line([(ex + 27, ey + 24), (ex + 40, ey - 22)], 11)])
    add(plate(disc(ex, ey, er + 3), yz, 0.9, floor + EMB_OUT - 1.4, coef))     # подложка
    add(plate(unary_union([ring, v, w]), yz, 1.8, floor + EMB_OUT, coef))      # знак

    # ---- накладки ПТФ: утопленная ниша + хром-полоска
    for (wx0, wy0, wx1, wy1), (sx0, sy0, sx1, sy1) in zip(PTF, PTF_STRIP):
        c2 = plane_fit(work, win_mask((wx0, wx1, wy0, wy1)))
        c2 = np.array([c2[0] * 0.5, c2[1] * 0.5, c2[2]])    # угол — наклон вполовину
        f2 = X_at(c2, yz, EMBLEM[0], EMBLEM[1])
        add(plate(pbox(wx0 + 5, wy0 + 5, wx1 - 5, wy1 - 5), yz, 1.1, f2 + 0.35, c2))
        add(plate(pbox(sx0, sy0, sx1, sy1), yz, 1.5, f2 + 1.3, c2))
        if verbose:
            print(f'  ПТФ: ниша X = {f2:.1f}')

    # ---- нижний воздухозаборник: тёмная маска кадра, сотовое поле, серебро обвеса
    g = gray_photo()
    dark = np.zeros((H, W), bool)
    x0, x1, y0, y1 = INTAKE
    dark[int(y0):int(y1), int(x0):int(x1)] = True
    dark &= g < INTAKE_DARK
    dark = ndi.binary_opening(dark, np.ones((7, 7)))
    dark = ndi.binary_closing(dark, np.ones((11, 11)))
    lab, n = ndi.label(dark)
    if n:
        sz = ndi.sum(dark, lab, range(1, n + 1))
        dark = lab == (int(np.argmax(sz)) + 1)
        dark = ndi.binary_fill_holes(dark)
    poly_dark = mask_polygon(dark, simplify=2.0)
    if verbose:
        print(f'  тёмная зона: {int(dark.sum())} px, полигон '
              f'{"нет" if poly_dark is None else round(poly_dark.area)}')
    if poly_dark is not None:
        cu = zone(dark, RECESS * 0.8, 'воздухозаборник')
        fu = X_at(cu, yz, EMBLEM[0], EMBLEM[1]) - RECESS * 0.8
        bnds = poly_dark.bounds
        for fy0, fy1 in INTAKE_FINS:
            fin = pbox(bnds[0] + 12, fy0, bnds[2] - 12, fy1).intersection(poly_dark)
            if not fin.is_empty:
                add(plate(fin, yz, 1.4, fu + 1.4, cu))
        # серебро обвеса вокруг ниши
        trim = np.zeros((H, W), bool)
        trim[int(y0 - 90):int(y1 + 70), int(x0 - 110):int(x1 + 110)] = True
        trim &= g > TRIM_BRIGHT
        trim[int(PLATE_HOLE[1]):int(PLATE_HOLE[3]), int(PLATE_HOLE[0]):int(PLATE_HOLE[2])] = False
        trim = ndi.binary_opening(trim, np.ones((3, 3)))
        trim &= ~dark
        # только вокруг ниши: не дальше 60 px от её края
        trim &= (ndi.distance_transform_edt(~dark) < 60) & ~dark
        poly_trim = mask_polygon(trim, simplify=1.6)
        ct = plane_fit(work, trim)
        if poly_trim is not None:
            add(plate(poly_trim, yz, 2.4, X_at(ct, yz, EMBLEM[0], EMBLEM[1]) + TRIM_OUT, ct))

    parts = [p for p in parts if p is not None]
    out = trimesh.util.concatenate([work] + parts)
    if verbose:
        print(f'  деталей: {len(parts)}, граней добавлено {len(out.faces) - len(work.faces)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', required=True)
    a = ap.parse_args()
    mesh = trimesh.load(a.src, force='mesh')
    print(f'вход: {a.src}: {len(mesh.faces)} граней')
    out = build(mesh)
    out.export(a.dst)
    print(f'выход: {a.dst}: {len(out.faces)} граней, watertight {out.is_watertight}, '
          f'{os.path.getsize(a.dst) / 1e6:.2f} МБ')


if __name__ == '__main__':
    main()
