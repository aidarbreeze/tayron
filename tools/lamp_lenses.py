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
            if verbose:
                print(f'  фара {side}, линза {name}: центр пикс ({p}, {py}) → '
                      f'точка ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}), '
                      f'R = {ru:.2f} юнита ({ru * 5.10 * 2:.0f} мм)')
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
