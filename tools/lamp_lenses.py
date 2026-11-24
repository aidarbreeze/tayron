#!/usr/bin/env python3
"""Две линзы в каждой фаре — по фотографии, как это читалось в модели Тигуана.

В скане внутренности фары не восстановились (глянцевый пластик), поэтому фара
остаётся «слепой»: в геометрии нет ни линз, ни отражателя. Сюда добавляются
две круглые линзы на каждую фару — большая (основной модуль) и малая
(дополнительный модуль), каждая с хромовым ободком — по местам, снятым с
референсного фото (кадр тот же, что у аудита).

Линза — отдельная замкнутая оболочка (эллипсоид-стекло + тор-ободок), она
добавляется к корпусу как есть: булевы операции на 300–900 тыс. граней дают
вырожденные грани, а на стык линзы с панелью смотреть не нужно.

    python3 tools/lamp_lenses.py --in tayron_2023_optimized_hires_v3.stl \\
        --out tayron_2023_optimized_hires_v4.stl
"""
import argparse
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_vs_photo as A                      # noqa: E402
import photo_texture as P                       # noqa: E402
import fix_front_lamps as L                     # noqa: E402  (маски фар с фото)

ROOT = A.ROOT
# (метка, центр x, центр y, радиус) в пикселях кадра «перед»; кадр 1482 px
LENSES = [('A', 272, 561, 42.0),     # основной модуль, ближе к крылу
          ('B', 346, 573, 27.0)]     # дополнительный, ближе к решётке
FRAME_W = 1482
DOME_DEPTH = 0.34                   # сплющивание эллипсоида по оси (0.34 → линза)
DOME_SETBACK = 2.2                  # центр эллипсоида за поверхностью, юниты
BEZEL_TUBE = 1.0                    # толщина хромового ободка, юниты


FLATTEN = 0.75                  # насколько выровнять «дно» фары под линзы
FLATTEN_TAPER = 10.0            # плавный вход в выравнивание, пиксели кадра


def flatten_lamp(mesh, view='front', flatten=FLATTEN, taper=FLATTEN_TAPER):
    """Выровнять дно ниши фары: скан там бугристый, линзы на нём «плавают».

    Плоскость подгоняется по самим вершинам ниши (X = aY + bZ + c), затем
    вершины подтягиваются к ней с весом smoothstep(расстояние до края контура) —
    на границе вес нулевой, поэтому панель не рвётся.
    """
    from scipy import ndimage as ndi
    px, py, sx, sy = L._px_py(mesh, view)
    V = mesh.vertices.copy()
    n = mesh.vertex_normals
    front = n[:, 0] / 0.25
    front = np.clip(front, 0, 1)
    masks = L.photo_masks(view, None, verbose=False)
    for name, m in masks:
        dist = ndi.distance_transform_edt(m)
        ix = np.clip(px.astype(int), 0, m.shape[1] - 1)
        iy = np.clip(py.astype(int), 0, m.shape[0] - 1)
        inside = m[iy, ix] & (front > 0.01)
        if inside.sum() < 50:
            continue
        A_ = np.stack([V[inside, 1], V[inside, 2], np.ones(int(inside.sum()))], 1)
        coef, *_ = np.linalg.lstsq(A_, V[inside, 0], rcond=None)
        plane = coef[0] * V[:, 1] + coef[1] * V[:, 2] + coef[2]
        t = np.clip(dist[iy, ix] / taper, 0, 1)
        w = t * t * (3 - 2 * t) * np.where(inside, 1.0, 0.0)
        V[:, 0] += (plane - V[:, 0]) * flatten * w
        print(f'  выравнивание ниши {name}: вершин {int((w > 0).sum())}')
    return trimesh.Trimesh(vertices=V, faces=mesh.faces, process=False)


def px_to_world(mesh, px, py):
    """Пиксель кадра → луч в мир: (начало, направление)."""
    box = A.BBOX['front']
    x0, x1, y0, y1 = box
    u_of, v_of, sx, sy = A.frame(mesh, 'front')
    u, v, _ = A.project(mesh, 'front')
    Y = u.max() - px / sx
    Z = v.min() + ((y1 - y0) - py) / sy
    origin = np.array([mesh.vertices[:, 0].max() + 20.0, Y, Z])
    return origin, np.array([-1.0, 0.0, 0.0])


def surface_hit(mesh, px, py):
    """Точка и нормаль поверхности фары под данным пикселем кадра."""
    origin, direction = px_to_world(mesh, px, py)
    loc, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=[origin], ray_directions=[direction], multiple_hits=True)
    if len(loc) == 0:
        raise RuntimeError(f'луч не попал в модель: пиксель ({px}, {py})')
    # multiple_hits: берём самое переднее (ближе к камере)
    i = int(np.argmax(loc[:, 0]))
    return loc[i], mesh.face_normals[index_tri[i]]


def barrel_mesh(center, normal, radius):
    """«Стакан» за линзой: сужается в глубину, даёт объём в 3/4."""
    n = normal / np.linalg.norm(normal)
    f = frustum(radius * 0.98, radius * BARREL_BACK, BARREL_LEN, sections=48)
    R = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), -n)
    f.apply_transform(R)
    f.apply_translation(center - n * 0.4)
    return f


def lens_mesh(center, normal, radius, detail=3):
    """Стекло-эллипсоид + хромовый ободок, ось — по нормали к панели."""
    n = normal / np.linalg.norm(normal)
    glass = trimesh.creation.icosphere(subdivisions=detail, radius=radius * 0.94)
    glass.apply_scale([1.0, 1.0, DOME_DEPTH])
    axis = np.array([0.0, 0.0, 1.0])
    R = trimesh.geometry.align_vectors(axis, n)
    glass.apply_transform(R)
    glass.apply_translation(center - n * DOME_SETBACK)
    seg = 64 if detail >= 3 else 32
    bezel = trimesh.creation.torus(major_radius=radius * 0.98,
                                   minor_radius=BEZEL_TUBE,
                                   major_segments=seg, minor_segments=12 if detail >= 3 else 8)
    bezel.apply_transform(R)
    bezel.apply_translation(center)          # ободок сидит на панели
    return trimesh.util.concatenate([glass, bezel])


# --------------------------------------------------------------- обвязка фары
DRL_STEP_PX = 2.5               # шаг вдоль контура, пиксели кадра
DRL_INSET_PX = 17.0             # отступ полосы ДХО от верхней кромки фары
DRL_WIDTH = 3.0                 # ширина полосы ДХО, юниты
DRL_OUT, DRL_IN = 1.2, 0.8      # выступ/заглубление полосы относительно панели
CHROME_INSET_PX = 9.0           # отступ нижней хромовой накладки, пиксели
CHROME_WIDTH = 2.6              # ширина накладки, юниты
CHROME_OUT, CHROME_IN = 0.9, 0.7   # выступ/заглубление накладки
BARREL_LEN = 7.0                # глубина «стакана» за линзой, юниты
BARREL_BACK = 0.50              # радиус дна стакана от радиуса линзы


def frustum(r0, r1, h, sections=48):
    """Замкнутая усечённая воронка по оси Z: от r0 (z=0) к r1 (z=h)."""
    ang = np.linspace(0, 2 * np.pi, sections, endpoint=False)
    c, s = np.cos(ang), np.sin(ang)
    v = np.concatenate([np.stack([r0 * c, r0 * s, np.zeros_like(ang)], 1),
                        np.stack([r1 * c, r1 * s, np.full_like(ang, h)], 1)])
    f = []
    for i in range(sections):
        j = (i + 1) % sections
        f.append([i, j, sections + j])
        f.append([i, sections + j, sections + i])
    cb = len(v)
    v = np.vstack([v, [[0, 0, 0], [0, 0, h]]])
    for i in range(sections):
        j = (i + 1) % sections
        f.append([cb, j, i])                      # дно
        f.append([cb + 1, sections + i, sections + j])   # горло
    return trimesh.Trimesh(vertices=v, faces=np.array(f), process=True)


def ribbon_solid(stations, width, d_out, d_in):
    """Замкнутая лента по станциям (точка, нормаль, поперечное направление)."""
    verts, faces = [], []
    for p, n, b in stations:
        half = b * (width * 0.5)
        verts.extend([p + half + n * d_out, p + half + n * d_in,
                      p - half + n * d_in, p - half + n * d_out])
    k = len(stations)
    for i in range(k - 1):
        a, c = i * 4, (i + 1) * 4
        for m in range(4):
            m2 = (m + 1) % 4
            faces.append([a + m, a + m2, c + m2])
            faces.append([a + m, c + m2, c + m])
    faces.extend([[0, 1, 2], [0, 2, 3]])                       # торец
    e = (k - 1) * 4
    faces.extend([[e + 3, e + 2, e + 1], [e + 3, e + 1, e + 0]])
    return trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)


def _stations(mesh, pts_px):
    """Точки кадра -> станции ленты (точка на панели, нормаль, поперечное)."""
    st = []
    for (px, py) in pts_px:
        c, n = surface_hit(mesh, px, py)
        n = n / np.linalg.norm(n)
        st.append((c, n, None))
    # сглаживаем путь и нормали: панель под полосой бугристая, иначе лента «пилит»
    def smooth(arr, k=7):
        pad = np.pad(arr, ((k // 2, k // 2), (0, 0)), mode='edge')
        ker = np.ones(k) / k
        return np.stack([np.convolve(pad[:, j], ker, mode='valid') for j in range(arr.shape[1])], 1)
    P_ = smooth(np.array([c for c, _, _ in st]))
    N_ = smooth(np.array([n for _, n, _ in st]))
    st = [(P_[i], N_[i], None) for i in range(len(st))]
    out = []
    for i, (c, n, _) in enumerate(st):
        j0, j1 = max(0, i - 1), min(len(st) - 1, i + 1)
        t = st[j1][0] - st[j0][0]
        t /= np.linalg.norm(t) + 1e-9
        b = np.cross(n, t)
        b /= np.linalg.norm(b) + 1e-9
        out.append((c, n, b))
    return out


def contour_path(mesh, side_shift=0, edge='top', inset=DRL_INSET_PX, turn=False):
    """Путь вдоль верхней или нижней кромки фары, с отступом внутрь контура."""
    from scipy import ndimage as ndi
    top, bot = L._envelopes(np.asarray(P.photo_crop('front', A.BBOX['front']).convert('L'),
                                       float), 130, 470, 470, 617, 130)
    top = ndi.median_filter(L._fill_nan(top), 21)   # контур с фото шумный
    bot = ndi.median_filter(L._fill_nan(bot), 21)
    xs = np.arange(150, 452, DRL_STEP_PX)
    pts = []
    for x in xs[::-1]:                              # от решётки к крылу
        i = int(x - 130)
        y = top[i] + inset if edge == 'top' else bot[i] - inset
        pts.append((float(x), float(y)))
    if turn:                                        # загиб вниз у крыла
        i0 = int(150 - 130)
        y0 = top[i0] + inset
        y1 = bot[i0] - inset * 1.3
        for y in np.arange(y0, y1, DRL_STEP_PX * 1.6):
            pts.append((150.0, float(y)))
    if side_shift:                                  # правая фара: зеркало по x
        pts = [(FRAME_W - px, py) for (px, py) in pts]
    return _stations(mesh, pts)


def drl_strip(mesh, side_shift=0):
    """Полоса ДХО по верхней кромке фары с загибом вниз у крыла."""
    return contour_path(mesh, side_shift, 'top', DRL_INSET_PX, turn=True)


def chrome_strip(mesh, side_shift=0):
    """Хромовая накладка по нижней кромке фары (в фото она есть)."""
    return contour_path(mesh, side_shift, 'bot', CHROME_INSET_PX, turn=False)


def build(mesh, verbose=True, flatten=True, detail=3):
    pivot = flatten_lamp(mesh) if flatten else mesh
    parts = []
    for side, sgn in (('лево', 0), ('право', 1)):
        for name, px, py, r in LENSES:
            p = px if sgn == 0 else FRAME_W - px
            center, normal = surface_hit(pivot, p, py)
            # радиус в юнитах: пиксели по горизонтали и вертикали дают разный мм/px
            _, _, sx, sy = A.frame(pivot, 'front')
            ru = r / np.sqrt(sx * sy)
            parts.append(lens_mesh(center, normal, ru, detail))
            parts.append(barrel_mesh(center, normal, ru))
            if verbose:
                print(f'  фара {side}, линза {name}: центр пикс ({p}, {py}) → '
                      f'точка ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), '
                      f'R = {ru:.2f} юнита ({ru * 5.10 * 2:.0f} мм)')
    for side_shift in (0, 1):
        st = drl_strip(pivot, side_shift)
        if verbose:
            print(f'  ДХО {"правая" if side_shift else "левая"}: станций {len(st)}')
        parts.append(ribbon_solid(st, DRL_WIDTH, DRL_OUT, DRL_IN))
        st2 = chrome_strip(pivot, side_shift)
        if verbose:
            print(f'  хром по низу {"правая" if side_shift else "левая"}: станций {len(st2)}')
        parts.append(ribbon_solid(st2, CHROME_WIDTH, CHROME_OUT, CHROME_IN))
    out = trimesh.util.concatenate([pivot] + parts)
    if verbose:
        print(f'  добавлено оболочек: {len(parts)}, граней к корпусу: '
              f'{len(out.faces) - len(mesh.faces)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', default=ROOT + 'tayron_2023_optimized_hires_v3.stl')
    ap.add_argument('--out', dest='dst', default=ROOT + 'tayron_2023_optimized_hires_v4.stl')
    ap.add_argument('--lens-detail', dest='detail', type=int, default=3,
                    help='2 — грубее (для lite), 3 — гладко')
    a = ap.parse_args()
    mesh = trimesh.load(a.src, force='mesh')
    print(f'вход: {a.src}: {len(mesh.faces)} граней')
    out = build(mesh, detail=a.detail)
    out.export(a.dst)
    print(f'выход: {a.dst}: {len(out.faces)} граней, watertight {out.is_watertight}, '
          f'частей {len(out.split(only_watertight=False))}, '
          f'{os.path.getsize(a.dst) / 1e6:.2f} МБ')


if __name__ == '__main__':
    main()
