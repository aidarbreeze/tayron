#!/usr/bin/env python3
"""Перенос элементов с Tiguan (`obj_1_tiguan.stl`) на Tayron.

Что переносится:
  --mirrors   боковые зеркала: корпус с заводской линией разъёма из Tiguan
              сажается на родную заготовку-конус Tayron вместо «булыжника»
  --wiper-r   задний дворник на стекло кормы (строится по месту, см. ниже)
  --wiper-f   пара передних дворников в корыте у кромки лобового

Системы координат
  Tiguan после выравнивания по OBB: столбцы (axis0 вверх, axis1 ширина,
  axis2 длина; +axis2 = перед). 1 юнит ≈ 58.5 мм.
  Tayron: (X длина, +X = перед; Y ширина; Z вверх). 1 юнит ≈ 5.10 мм.
  Пересчёт «юнит Tiguan -> юнит Tayron» без искажений: 58.5 / 5.10 = 11.47.

Схема сборки
  1. из Tiguan булевым пересечением с параллелепипедом вырезается замкнутый
     блок детали — корпус зеркала (в комле 37 × 33 юнита Tayron);
  2. у Tayron заготовка зеркала подрезается двумя призмами (вершина конуса и
     «нога» под корпусом) и корпус сажается комлем на её место; шов убирается
     локальным сглаживанием в поясе комля;
  3. дворники в доноре отдельной геометрией не выделены (на стёклах только паз),
     поэтому они строятся по месту — лезвие + поводок + ось — на плоскостях
     стёкол, подобранных по крупным граням кузова Tayron;
  4. всё объединяется булево: на выходе одно замкнутое тело, объём и число
     граней расходятся с исходником на доли процента.

Результат — новый файл; исходники не меняются.
"""
import argparse
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
TIGUAN = ROOT + 'obj_1_tiguan.stl'
TAYRON = ROOT + 'tayron_2023_optimized.stl'

MM_TIGUAN = 58.5
MM_TAYRON = 5.10
S = MM_TIGUAN / MM_TAYRON          # 11.47: юниты Tiguan -> юниты Tayron


# ------------------------------------------------------------------ утилиты
def box_between(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = trimesh.creation.box(extents=hi - lo)
    b.apply_translation((lo + hi) / 2)
    return b


def as_volume(mesh):
    """Замкнутый корректно ориентированный объём (нужно для булевых операций).

    Если меш уже корректный объём — возвращается как есть: повторная склейка
    вершин на сложной сетке рвёт её на куски.
    """
    if mesh.is_volume:
        return mesh
    m = mesh.copy()
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    m.fix_normals()
    if not m.is_watertight:
        m.fill_holes()
    return m


def finalize(mesh):
    """Чистка перед экспортом: выкидывает вырожденные и дублирующиеся грани.

    Булевы операции иногда оставляют нулевые треугольники; в памяти меш ещё
    замкнут, но STL не хранит индексы вершин, и после перезагрузки такой меш
    распадается.
    """
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    m.fix_normals()
    if not m.is_watertight:
        m.fill_holes()
    m.merge_vertices()
    return m


def verify(path):
    """Проверка уже записанного файла: перезагрузка и контроль замкнутости."""
    m = trimesh.load(path, force='mesh')
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    parts = len(m.split(only_watertight=False)) if len(m.faces) < 500_000 else '—'
    print(f'  проверка файла: граней {len(m.faces)}, watertight {m.is_watertight}, '
          f'частей {parts}, объём {round(float(m.volume), 1)}')
    return m


def tiguan_aligned():
    """Tiguan в осях OBB: (axis0 вверх, axis1 ширина, axis2 длина)."""
    m = trimesh.load(TIGUAN, force='mesh')
    T, _ = trimesh.bounds.oriented_bounds(m)
    V = trimesh.transform_points(m.vertices, T)
    return as_volume(trimesh.Trimesh(vertices=V, faces=m.faces, process=True))


# ------------------------------------------------------------------ зеркала
# Tiguan: корпус зеркала (axis0 вверх, axis1 ширина, axis2 длина; +axis2 = перед)
MIR_LO = (4.4, 15.9, 6.6)
MIR_HI = (7.6, 18.2, 11.2)

# В модели Тайрона на двери уже стоит грубый конус-заготовка зеркала:
#   |Y| 172: ~39 x 48   |Y| 182: 35.6 x 34.7   |Y| 194: 33.6 x 31.8   вершина |Y| 213
# Корпус Tiguan в комле 37 x 33 — тот же калибр. Заготовку не выбрасываем,
# а обрезаем воронкой: у двери широкая манжета, к комлю корпуса — узкое горло,
# которое корпус наглухо проглатывает. Так стык получается без ступеньки.
# Манжета: гладкий лофт от овала на двери к комлю корпуса. Кольца подобраны так,
# чтобы накрыть заготовку модели (её сечения: |Y| 172 -> ~39x48, 182 -> 35.6x34.7,
# 186 -> 35x33) и в комле (|Y| 186) уйти внутрь корпуса (37x33).
# (|Y|, центр X, центр Z, полуось X, полуось Z, показатель суперэллипса)
# Кольца идут по профилю заготовки (см. её сечения выше) чуть снаружи неё,
# поэтому манжета — «носок» на конусе: заготовка не протыкает её, а стык у
# корпуса уходит внутрь комля.
CUFF = [(164.0, 118.0, 74.0, 20.5, 19.5, 2.3),
        (170.0, 118.0, 74.5, 19.5, 18.5, 2.2),
        (175.0, 116.5, 74.0, 18.0, 18.0, 2.2),
        (179.0, 115.5, 74.0, 16.5, 16.0, 2.1),
        (182.5, 114.0, 74.0, 12.5, 11.5, 2.1)]
CROP_LO = (80.0, 181.5, 42.0)      # подрезка вершины заготовки
CROP_HI = (142.0, 305.0, 104.0)
LEG_LO = (80.0, 175.0, 33.0)        # срез «ноги» — того, что торчит под корпусом
LEG_HI = (142.0, 305, 56.4)
ROOT_Y = 180.0                      # плоскость комля корпуса (заготовка на 181.5)
CENTRE_XZ = (115.0, 74.0)           # центр комля = центр заготовки на этой глубине
SWEEP = 0.10                        # заворот корпуса назад вокруг Z, рад


def mir_cut(body, lo, hi, side):
    """Вычитает из кузова призму [lo, hi] на данном борту (zip по |Y|)."""
    y0, y1 = lo[1] * side, hi[1] * side
    c = box_between([min(lo[0], hi[0]), min(y0, y1), min(lo[2], hi[2])],
                    [max(lo[0], hi[0]), max(y0, y1), max(lo[2], hi[2])])
    return as_volume(trimesh.boolean.difference([body, c], engine='manifold'))


def sup_ring(rx, rz, n, seg=72):
    """Суперэллипс: n = 2 — эллипс, больше — скруглённый прямоугольник."""
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s_ = np.cos(th), np.sin(th)
    x = rx * np.sign(c) * np.abs(c) ** (2.0 / n)
    z = rz * np.sign(s_) * np.abs(s_) ** (2.0 / n)
    return np.stack([x, z], 1)


def cuff_mesh(side, seg=72):
    """Замкнутая манжета вдоль Y: кольца CUFF, торцы закрыты."""
    verts, faces = [], []
    for y, cx, cz, rx, rz, n in CUFF:
        for x, z in sup_ring(rx, rz, n, seg):
            verts.append([cx + x, y * side, cz + z])
    nr = len(CUFF)
    for k in range(nr - 1):
        for i in range(seg):
            a = k * seg + i; b = k * seg + (i + 1) % seg
            c = (k + 1) * seg + (i + 1) % seg; d = (k + 1) * seg + i
            faces += [[a, b, c], [a, c, d]] if side > 0 else [[a, c, b], [a, d, c]]
    c0 = len(verts); verts.append([CUFF[0][1], CUFF[0][0] * side, CUFF[0][2]])
    c1 = len(verts); verts.append([CUFF[-1][1], CUFF[-1][0] * side, CUFF[-1][2]])
    last = (nr - 1) * seg
    for i in range(seg):
        faces.append([c0, (i + 1) % seg, i] if side > 0 else [c0, i, (i + 1) % seg])
        faces.append([c1, last + i, last + (i + 1) % seg] if side > 0
                     else [c1, last + (i + 1) % seg, last + i])
    return as_volume(trimesh.Trimesh(vertices=np.array(verts, float),
                                     faces=np.array(faces), process=False))


def carva_foot(body, side):
    """Подрезает заготовку (вершина конуса и «нога») и возвращает (кузов, манжету).

    Сама манжета — накладка поверх конуса: срезать по ней кузов нельзя, иначе
    в двери открывается прямоугольный вырез по окну реза.
    """
    cuff = cuff_mesh(side)
    body = mir_cut(body, CROP_LO, CROP_HI, side)
    body = mir_cut(body, LEG_LO, LEG_HI, side)
    return body, cuff


def housing_piece(tg, side):
    """Корпус зеркала Tiguan в осях Тайрона; +Y — наружу, +X — перед машины."""
    piece = as_volume(trimesh.boolean.intersection(
        [tg, box_between(list(MIR_LO), list(MIR_HI))], engine='manifold'))
    W = np.stack([piece.vertices[:, 2], piece.vertices[:, 1], piece.vertices[:, 0]], 1) * S
    F = piece.faces
    if side < 0:
        W[:, 1] *= -1
        F = F[:, ::-1]
    piece = as_volume(trimesh.Trimesh(vertices=W, faces=F, process=False))
    b = piece.bounds
    dy = -b[0][1] if side > 0 else -b[1][1]        # комель в Y = 0, корпус наружу
    piece.apply_translation([-(b[0][0] + b[1][0]) / 2, dy, -(b[0][2] + b[1][2]) / 2])
    th = -SWEEP if side > 0 else SWEEP              # нос корпуса заворачиваем назад
    R = trimesh.transformations.rotation_matrix(th, [0, 0, 1])[:3, :3]
    piece.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    piece.apply_translation([CENTRE_XZ[0], ROOT_Y * side, CENTRE_XZ[1]])
    return as_volume(piece)


def smooth_band(mesh, select, iterations=2, lamb=0.3):
    V = mesh.vertices.copy()
    idx = np.where(select(V))[0]
    if len(idx) == 0:
        return mesh
    neigh = [[] for _ in range(len(V))]
    for i, j in mesh.edges_unique:
        neigh[i].append(j); neigh[j].append(i)
    for _ in range(iterations):
        new = V.copy()
        for i in idx:
            if neigh[i]:
                new[i] = V[i] * (1 - lamb) + V[neigh[i]].mean(0) * lamb
        V = new
    return trimesh.Trimesh(vertices=V, faces=mesh.faces, process=False)


def build_mirrors(body, tg):
    for side in (+1, -1):
        body, cuff = carva_foot(body, side)
        piece = housing_piece(tg, side)
        body = as_volume(trimesh.boolean.union([body, cuff, piece], engine='manifold'))
        ring_c = np.array([CENTRE_XZ[0], ROOT_Y * side, CENTRE_XZ[1]])
        body = smooth_band(body, lambda V, c=ring_c: (np.linalg.norm(V - c, axis=1) < 9.0),
                           iterations=2, lamb=0.22)
        body = as_volume(body)
        print(f'  зеркало {side:+d}: граней {len(body.faces)}, watertight {body.is_watertight}, '
              f'|Y|max {abs(body.vertices[:,1]).max():.1f}')
    return body


# ---------------------------------------------------------------- дворники
# В модели Tiguan отдельной геометрии дворников нет — на стёклах только паз под
# них (проверено: булевы вырезки дают линзу стекла, а не лезвие). Поэтому
# дворники строятся по месту, но по схеме донора: задний — лезвие на стекле
# кормы с поводком и осью, передние — пара в корыте у кромки лобового.
#
# Плоскости стёкол Тайрона (нормаль наружу, точка на стекле), подобраны по
# крупным граням кузова:
REAR_GLASS = ((-0.613, 0.0, 0.79), (-374.3, -1.5, 107.0))     # наклонная корма
FRONT_GLASS = ((0.54, 0.0, 0.842), (136.0, 3.0, 110.0))       # лобовое

WIPER_SINK = 1.2                  # насколько основание утоплено в стекло

# задний: (длина лезвия, смещение оси вдоль лезвия, смещение оси вниз по стеклу)
REAR_WIPER = dict(blade=68.0, hub_dx=30.0, hub_dy=-16.0, y=0.0, slope=-6.0)
# передние: два взмаха, оси у корыта
FRONT_WIPERS = [dict(blade=112.0, hub_dx=42.0, hub_dy=-13.0, y=44.0, slope=-16.0,
                     flip=False),
                dict(blade=104.0, hub_dx=40.0, hub_dy=-12.0, y=-42.0, slope=-20.0,
                     flip=True)]


def rot_between(a, b):
    a = np.asarray(a, float); a /= np.linalg.norm(a)
    b = np.asarray(b, float); b /= np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _along(seg_len, radius, a, b, sections=16):
    """Цилиндр между точками a и b."""
    d = np.asarray(b, float) - np.asarray(a, float)
    ln = float(np.linalg.norm(d))
    cyl = trimesh.creation.cylinder(radius=radius, height=ln, sections=sections)
    R = rot_between([0, 0, 1], d / ln)
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = (np.asarray(a) + np.asarray(b)) / 2
    cyl.apply_transform(M)
    return cyl


def make_wiper(blade, hub_dx, hub_dy, flip=False):
    """Дворник в локальных осях: x — вдоль лезвия, y — вверх по стеклу, z — наружу."""
    sx = -1.0 if flip else 1.0
    r_b = 1.15
    z_b = r_b * 0.95
    lez = trimesh.creation.cylinder(radius=r_b, height=blade, sections=28)
    lez.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    lez.apply_scale([1.0, 1.7, 0.85])
    lez.apply_translation([0.0, 0.0, z_b])
    hub = np.array([sx * hub_dx, hub_dy, 1.0])
    arm = _along(1.0, 0.62, hub, np.array([sx * hub_dx * 0.25, 0.0, z_b * 0.9]))
    axis = trimesh.creation.cylinder(radius=2.3, height=1.5, sections=24)
    axis.apply_translation([hub[0], hub[1], 0.75])
    return as_volume(trimesh.boolean.union([lez, arm, axis], engine='manifold'))


def place_on_glass(piece, plane, y_off, slope_off, body, sink=WIPER_SINK):
    """Кладёт деталь на плоскость стекла: локальный z -> нормаль, x -> вдоль борта."""
    n_t = np.asarray(plane[0], float); n_t /= np.linalg.norm(n_t)
    p0 = np.asarray(plane[1], float)
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(n_t, e1); e2 /= np.linalg.norm(e2)      # вверх по стеклу
    R = np.stack([e1, e2, n_t], 1)
    piece.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1]]))
    target = p0 + e1 * y_off + e2 * slope_off
    cp, _, _ = trimesh.proximity.closest_point(body, [target])
    target = target + n_t * (float(cp[0] @ n_t) - float(target @ n_t))
    piece.apply_translation(target - n_t * sink)
    return as_volume(piece)


def build_wiper_rear(body, tg=None):
    cfg = REAR_WIPER
    piece = make_wiper(cfg['blade'], cfg['hub_dx'], cfg['hub_dy'])
    n0 = len(body.faces)
    piece = place_on_glass(piece, REAR_GLASS, cfg['y'], cfg['slope'], body)
    body = as_volume(trimesh.boolean.union([body, piece], engine='manifold'))
    print(f'  задний дворник: граней {len(body.faces)} (+{len(body.faces)-n0}), '
          f'watertight {body.is_watertight}')
    return body


def build_wiper_front(body, tg=None):
    for i, cfg in enumerate(FRONT_WIPERS):
        piece = make_wiper(cfg['blade'], cfg['hub_dx'], cfg['hub_dy'], cfg['flip'])
        n0 = len(body.faces)
        piece = place_on_glass(piece, FRONT_GLASS, cfg['y'], cfg['slope'], body)
        body = as_volume(trimesh.boolean.union([body, piece], engine='manifold'))
        print(f'  передний дворник {i+1}: граней {len(body.faces)} (+{len(body.faces)-n0}), '
              f'watertight {body.is_watertight}')
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=ROOT + 'tayron_2023_optimized_v2.stl')
    ap.add_argument('--body', default=TAYRON, help='сборка-основа (lite / 16 МБ / hires)')
    ap.add_argument('--mirrors', action='store_true')
    ap.add_argument('--wiper-r', action='store_true', help='задний дворник')
    ap.add_argument('--wiper-f', action='store_true', help='передние дворники')
    ap.add_argument('--preview', default=None, help='префикс для png-превью')
    a = ap.parse_args()

    tg = tiguan_aligned()
    print('Tiguan выровнен: extents', np.round(tg.extents, 2), 'граней', len(tg.faces))
    body = as_volume(trimesh.load(a.body, force='mesh'))
    print('Tayron:', os.path.basename(a.body), 'граней', len(body.faces), 'watertight', body.is_watertight,
          'объём', round(float(body.volume), 1))

    if a.mirrors:
        body = build_mirrors(body, tg)
    if a.wiper_r:
        body = build_wiper_rear(body, tg)
    if a.wiper_f:
        body = build_wiper_front(body, tg)
    if a.mirrors or a.wiper_r or a.wiper_f:
        body = finalize(body)
        print('итог: граней', len(body.faces), 'watertight', body.is_watertight,
              'объём', round(float(body.volume), 1),
              'частей', len(body.split(only_watertight=False)))
        body.export(a.out)
        print('записано', a.out, round(os.path.getsize(a.out) / 1e6, 2), 'МБ')
        verify(a.out)

    if a.preview:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
        import preview_parts
        preview_parts.render_regions(body, a.preview)


if __name__ == '__main__':
    main()
