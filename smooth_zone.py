#!/usr/bin/env python3
"""Доводка зон, сглаженных в браузере: соты и мелкий рельеф сносятся на гладкую
панель, сетка и топология остаются прежними.

Зачем отдельный шаг. В браузере сглаживание — это диффузия: вершины тянутся к
среднему соседей. Рельеф она убирает, но остаются «пятна» — вершины стоят на
стенках ячеек, где их густо, а между ячейками редко, поэтому поверхность
получается волнистой, а не панелью.

Что делает скрипт (только сдвиг вершин, ни одной новой/удалённой вершины или
грани):

1. Находит зоны, которые правили в браузере, — вершины, сдвинутые относительно
   исходного кузова v14.
2. По «полу» зоны (вершины, у которых нормаль смотрит вперёд, то есть это
   панель, а не стенки ниши) строит гладкое поле x = H(y, z): робастная
   квадратичная аппроксимация с отбрасыванием сот как выбросов. Поле описывает
   изогнутую панель кузова — по y и z, а не плоскость.
3. Сносит на это поле вершины рельефа (|высота| до 1.0 мм — это соты, шаг 0.3-1.2
   мм): вес 1 в глубине зоны, плавно к нулю у кромки. Стенки ниши и высокие рёбра
   (выше 1.6 мм) не трогаются — они остаются деталями кузова.
4. Пишет 3MF с теми же 120 376 гранями.

    python3 smooth_zone.py "сгладил решеткуTayron_body_smooth.3mf" \
        --out Tayron_body_smooth_v2.3mf
"""
import argparse

import numpy as np

import m3mf

U = 5.8824           # единиц модели на мм печати
ZONE_MIN_MM = 0.01   # сдвиг, с которого вершина считается «в зоне»
FALLOFF_RINGS = 3    # плавный вход в зону
SNAP_FULL_MM = 1.0   # рельеф до этой высоты сносится на панель полностью
SNAP_ZERO_MM = 1.6   # выше — не трогаем


def weld(V):
    """canon: индекс файла → индекс уникальной точки (точно совпадающие)."""
    key = np.round(V / 1e-6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return inv.astype(np.int32), len(first)


def adjacency(F, vk):
    e = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    deg = np.bincount(e.ravel(), minlength=vk)
    off = np.concatenate([[0], np.cumsum(deg)])
    nbr = np.empty(off[-1], np.int32)
    fill = off[:-1].copy()
    for a, b in e:
        nbr[fill[a]] = b; fill[a] += 1
        nbr[fill[b]] = a; fill[b] += 1
    return e, off, nbr


def vertex_normals(P, F):
    n = np.zeros_like(P)
    tri = P[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for i in range(3):
        np.add.at(n, F[:, i], fn)
    ln = np.linalg.norm(n, axis=1)
    ln[ln == 0] = 1
    return n / ln[:, None]


def border_distance(mask, off, nbr):
    """Сколько рёбер от вершины до кромки зоны (0 — сама кромка)."""
    dist = np.full(len(mask), -1, np.int32)
    border = [v for v in np.where(mask)[0] if not mask[nbr[off[v]:off[v + 1]]].all()]
    dist[border] = 0
    cur = np.array(border, np.int32)
    for d in range(1, 64):
        nxt = []
        for v in cur:
            for u in nbr[off[v]:off[v + 1]]:
                if mask[u] and dist[u] < 0:
                    dist[u] = d; nxt.append(u)
        if not nxt:
            break
        cur = np.array(nxt, np.int32)
    dist[(dist < 0) & mask] = 63
    return dist


def panel_field(y, z, x, rounds=4):
    """Гладкое поле панели x = H(y, z): робастная квадратичная аппроксимация.

    Возвращает H(y, z) с обрезкой аргументов по габаритам «пола» (чтобы не
    экстраполировать в стенки ниши) и маску точек, по которым считали.
    """
    y0, z0 = y.mean(), z.mean()
    sy, sz = max(y.std(), 1e-6), max(z.std(), 1e-6)
    u, w = (y - y0) / sy, (z - z0) / sz
    A = np.column_stack([np.ones_like(u), u, w, u * u, u * w, w * w])
    beta = np.linalg.lstsq(A, x, rcond=None)[0]
    for _ in range(rounds):
        r = x - A @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        wt = np.minimum(1.0, 1.5 * s / np.maximum(np.abs(r), 1e-12))
        Aw = A * wt[:, None]
        beta = np.linalg.lstsq(Aw.T @ A, Aw.T @ x, rcond=None)[0]
    ylo, yhi, zlo, zhi = y.min(), y.max(), z.min(), z.max()

    def H(yy, zz):
        yy = np.clip(yy, ylo, yhi)
        zz = np.clip(zz, zlo, zhi)
        uu, ww = (yy - y0) / sy, (zz - z0) / sz
        return np.column_stack([np.ones_like(uu), uu, ww, uu * uu, uu * ww, ww * ww]) @ beta

    return H


def smooth_weights(w, loc_idx, eu_sub):
    """Размыть веса по сетке: переход в зону не «ступенькой»."""
    for _ in range(4):
        sw = np.zeros(len(w)); cnt = np.zeros(len(w))
        la, lb = loc_idx[eu_sub[:, 0]], loc_idx[eu_sub[:, 1]]
        ok = (la >= 0) & (lb >= 0)
        np.add.at(sw, la[ok], w[lb[ok]]); np.add.at(sw, lb[ok], w[la[ok]])
        np.add.at(cnt, la[ok], 1); np.add.at(cnt, lb[ok], 1)
        w = np.where(cnt > 0, 0.5 * w + 0.5 * sw / np.maximum(cnt, 1), w)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='3MF, сохранённый из сайта')
    ap.add_argument('--base', default='Tayron5_flex.3mf', help='исходный кузов (v14)')
    ap.add_argument('--out', default='Tayron_body_smooth_v2.3mf')
    ap.add_argument('--full', action='store_true',
                    help='снести на панель всю зону целиком, включая рёбра внутри неё')
    a = ap.parse_args()

    V = F = V0 = None
    (n1, usr), = m3mf.read(a.source)
    V = np.asarray(usr.vertices, float)
    F = np.asarray(usr.faces, np.int32)
    for name, part in m3mf.read(a.base):        # в v14 несколько деталей — ищем кузов
        v = np.asarray(part.vertices, float)
        if len(v) == len(V):
            V0 = v
            break
    assert V0 is not None, 'в базовом файле нет кузова с тем же числом вершин'

    d = np.linalg.norm(V - V0, axis=1) / U
    zone_f = d > ZONE_MIN_MM
    print('зон(вершин): %d, макс сдвиг %.2f мм' % (zone_f.sum(), d.max()))

    canon, vk = weld(V)
    Fc = canon[F]
    eu, off, nbr = adjacency(Fc, vk)
    P = np.zeros((vk, 3)); P[canon] = V
    nrm = vertex_normals(P, Fc)

    zonec = np.zeros(vk, bool); zonec[canon[zone_f]] = True
    rings = border_distance(zonec, off, nbr)
    weight = np.clip(rings / FALLOFF_RINGS, 0, 1).astype(float)
    weight = weight * weight * (3 - 2 * weight)          # smoothstep

    comp = -np.ones(vk, np.int32)
    cid = 0
    for s in np.where(zonec)[0]:
        if comp[s] >= 0:
            continue
        stack, comp[s] = [s], cid
        while stack:
            v = stack.pop()
            for u in nbr[off[v]:off[v + 1]]:
                if zonec[u] and comp[u] < 0:
                    comp[u] = cid; stack.append(u)
        cid += 1
    print('компонент зон:', cid)

    moved = 0
    for k in range(cid):
        sel = comp == k
        if sel.sum() < 200:                              # мелочь не трогаем
            continue
        floorv = np.abs(nrm[sel, 0]) > 0.65               # панель, не стенки ниши
        if floorv.sum() < 100:
            floorv = np.ones(sel.sum(), bool)
        H = panel_field(P[sel][floorv, 1], P[sel][floorv, 2], P[sel][floorv, 0])
        tgt = H(P[sel, 1], P[sel, 2])
        h = (P[sel, 0] - tgt) / U                         # высота над панелью, мм
        near = (np.ones_like(h) if a.full
                else np.clip((SNAP_ZERO_MM - np.abs(h)) / (SNAP_ZERO_MM - SNAP_FULL_MM), 0, 1))
        w2 = weight[sel] * near
        loc = -np.ones(vk, np.int32); loc[np.where(sel)[0]] = np.arange(sel.sum())
        w2 = smooth_weights(w2, loc, eu)
        shift = w2 * (tgt - P[sel, 0])
        P[sel, 0] += shift
        # в глубине зоны — ровно на поле (без остатка), иначе остаются пятна
        # от неровной плотности вершин
        exact = w2 > 0.98
        if exact.any():
            P[np.where(sel)[0][exact], 0] = tgt[exact]
        moved += int(np.count_nonzero(np.abs(shift) > 1e-6))
        print('  компонент %d: %d вершин (панель: %d), снесено %.2f мм в среднем, '
              'макс %.2f мм' % (k, sel.sum(), floorv.sum(),
                                np.abs(shift[shift != 0]).mean() / U if (shift != 0).any() else 0,
                                np.abs(shift).max() / U))

    m3mf.write(a.out, [('Кузов', __import__('trimesh').Trimesh(P, Fc, process=False))],
               meta='Кузов: зоны, сглаженные в браузере, снесены на гладкую панель')
    print('вершин %d, граней %d, сдвинуто вершин %d' % (len(P), len(Fc), moved))
    print('записано:', a.out)


if __name__ == '__main__':
    main()
