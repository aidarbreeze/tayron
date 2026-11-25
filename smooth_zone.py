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


def panel_field(y, z, x, rounds=4, clip=True):
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
        if clip:
            yy = np.clip(yy, ylo, yhi)
            zz = np.clip(zz, zlo, zhi)
        uu, ww = (yy - y0) / sy, (zz - z0) / sz
        return np.column_stack([np.ones_like(uu), uu, ww, uu * uu, uu * ww, ww * ww]) @ beta

    return H


def fill_recess(P, F, nrm, off, nbr, zone_mm=(-18.0, 18.0), z_mm=(21.3, 31.7),
                depth_mm=0.30, badge=(25.7, 4.6), skin=None):
    """Ниша решётки: что заливать и до какой поверхности.

    Область — не прямоугольник, а связная часть оболочки: всё, что в полосе
    решётки лежит ниже «арки» (поверхности по гребням сот) и связано с полом
    сот. Так в неё попадают и стенки ячеек, и горизонтальные плашки, но не
    попадают кромка ниши (она выше арки), эмблема и внутренняя сторона оболочки
    (это отдельный лист сетки, связанный с наружным только по краю кузова).
    """
    yy, zz = P[:, 1] / U, P[:, 2] / U
    box = ((yy >= zone_mm[0]) & (yy <= zone_mm[1]) &
           (zz >= z_mm[0]) & (zz <= z_mm[1]))
    front = nrm[:, 0] > 0.5
    cand = box & front
    if cand.sum() < 200:
        return None, None
    if skin is not None:
        A = skin
    else:
        crest = cand & (P[:, 0] >= np.percentile(P[cand, 0], 80))
        A = panel_field(P[crest, 1], P[crest, 2], P[crest, 0], clip=False)
    lev = A(P[:, 1], P[:, 2])
    low = box & (P[:, 0] < lev - depth_mm * U)
    # связные куски «ниже арки», в которых есть пол сот
    comp = -np.ones(len(P), np.int32)
    cid = 0
    for s in np.where(low)[0]:
        if comp[s] >= 0:
            continue
        stack, comp[s], has_front = [s], cid, bool(front[s])
        while stack:
            v = stack.pop()
            for u in nbr[off[v]:off[v + 1]]:
                if low[u] and comp[u] < 0:
                    comp[u] = cid
                    has_front = has_front or bool(front[u])
                    stack.append(u)
        if not has_front:
            comp[comp == cid] = -2          # кусок без пола сот — не ниша
        cid += 1
    R = comp >= 0
    # «закрыть» область по строкам z: граница тогда идёт по рамке решётки, а не
    # зигзагом по сотам (иначе на стыке остаются заломы)
    dz = 0.25
    bins = np.floor((zz - zz.min()) / dz).astype(int)
    lo = np.full(bins.max() + 1, np.inf); hi = np.full(bins.max() + 1, -np.inf)
    for b in np.unique(bins[R]):
        yy_r = yy[R & (bins == b)]
        lo[b] = yy_r.min(); hi[b] = yy_r.max()
    ok = np.isfinite(lo)
    if ok.sum() >= 3:
        k = np.ones(5) / 5
        for arr in (lo, hi):
            good = np.where(ok, arr, np.nan)
            fill = np.interp(np.arange(len(good)), np.where(ok)[0], good[ok])
            arr[:] = np.convolve(np.pad(fill, 2, mode='edge'), k, mode='valid')
    inside_rows = np.zeros(len(P), bool)
    for b in np.unique(bins):
        row = bins == b
        if ok[b]:
            inside_rows |= row & (yy >= lo[b]) & (yy <= hi[b])
    # только геометрия у самой решётки: иначе «строка» захватит заднюю часть кузова
    near_lev = (P[:, 0] > lev - 6.0 * U) & (P[:, 0] < lev + 3.0 * U)
    R = R | (inside_rows & box & near_lev)
    # добить треугольники, у которых в области 2 из 3 вершин: иначе на кромке
    # остаются «зубцы» из недорубленных граней
    for _ in range(3):
        add = np.zeros(len(P), bool)
        for f in F:
            if R[f[0]] or R[f[1]] or R[f[2]]:
                add[f] = True
        add &= box & near_lev
        R = R | add
    return R, A


def refine_patch(P, F, patch, level_fn, max_edge):
    """Дробить крупные треугольники заплаты, новые точки — сразу на поле.

    Плотность вершин в бывших сотах неравномерная: на стенках ячеек вершин густо,
    внутри ячеек — ни одной. После сноса на панель получаются длинные
    треугольники («фасетки»). Здесь они дробятся по рёбрам длиннее max_edge
    (1→2, 1→3, 1→4 — по числу раздробленных рёбер), новые вершины ставятся на то
    же поле. Кромка заплаты не дробится, поэтому T-стыков с остальной сеткой нет.
    """
    pts = [list(map(float, p)) for p in P]
    mid = {}

    def midpoint(a, b):
        key = (a, b) if a < b else (b, a)
        if key in mid:
            return mid[key]
        pts.append([0.5 * (pts[a][i] + pts[b][i]) for i in range(3)])
        mid[key] = len(pts) - 1
        return mid[key]

    faces = [list(map(int, f)) for f in patch]
    for _ in range(6):
        cnt = {}
        for f in faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                k = (a, b) if a < b else (b, a)
                cnt[k] = cnt.get(k, 0) + 1
        split = {k for k, c in cnt.items()
                 if c == 2 and float(np.linalg.norm(np.subtract(pts[k[0]], pts[k[1]]))) > max_edge}
        if not split:
            break
        out = []
        for f in faces:
            e = [((f[0], f[1]) if f[0] < f[1] else (f[1], f[0])),
                 ((f[1], f[2]) if f[1] < f[2] else (f[2], f[1])),
                 ((f[2], f[0]) if f[2] < f[0] else (f[0], f[2]))]
            hit = [t for t in range(3) if e[t] in split]
            if len(hit) == 0:
                out.append(f)
            elif len(hit) == 3:
                m0, m1, m2 = (midpoint(*e[t]) for t in range(3))
                out += [[f[0], m0, m2], [m0, f[1], m1], [m2, m1, f[2]], [m0, m1, m2]]
            elif len(hit) == 1:
                t = hit[0]
                a, b, c = f[t], f[(t + 1) % 3], f[(t + 2) % 3]
                m = midpoint(*e[t])
                out += [[a, m, c], [m, b, c]]
            else:
                t = next(t for t in range(3) if t in hit and (t + 1) % 3 in hit)
                a, b, c = f[t], f[(t + 1) % 3], f[(t + 2) % 3]
                m0 = midpoint(*e[t])
                m1 = midpoint(*e[(t + 1) % 3])
                out += [[a, m0, c], [m0, b, m1], [m0, m1, c]]
        faces = out
    return pts, faces


def slab_solid(A, body, y0, y1, z0, z1, step, thickness_mm, drop_mm=0.05,
               only_recess=False, mask=None):
    """Плита «по уровню»: заполняет только те места, где кузов ниже поля A.

    Сетка в плоскости (y, z); для каждой ячейки пускаем луч в кузов и смотрим,
    ниже ли поверхность поля. Из подходящих ячеек собирается тело с передней
    гранью на поле. Кромка такой плиты лежит там, где поверхность кузова выходит
    на уровень поля, поэтому стенки не видны — получается не «пластина сверху»,
    а залитая впадина.
    """
    ys = np.arange(y0, y1, step); ys = np.append(ys, y1)
    zs = np.arange(z0, z1, step); zs = np.append(zs, z1)
    ny, nz = len(ys), len(zs)
    Yc, Zc = np.meshgrid((ys[:-1] + ys[1:]) / 2, (zs[:-1] + zs[1:]) / 2, indexing='ij')
    org = np.column_stack([np.full(Yc.size, 400.0 * U), Yc.ravel(), Zc.ravel()])
    loc, ir, _ = body.ray.intersects_location(
        ray_origins=org, ray_directions=np.tile([-1.0, 0, 0], (Yc.size, 1)), multiple_hits=False)
    xsurf = np.full(Yc.size, -np.inf)
    xsurf[ir] = loc[:, 0]
    lev = A(Yc.ravel(), Zc.ravel())
    if mask is not None:
        mask = np.asarray(mask, bool)
        assert mask.shape == (ny - 1, nz - 1), 'маска не по сетке плиты'
    elif only_recess:
        mask = (xsurf < lev - drop_mm * U).reshape(ny - 1, nz - 1)
    else:
        mask = np.ones((ny - 1, nz - 1), bool)
    # убираем «диагональные» контакты: они дают неманифолдные рёбра
    for _ in range(50):
        ch = False
        for i in range(ny - 2):
            for j in range(nz - 2):
                a, b = mask[i, j], mask[i, j + 1]
                c, d = mask[i + 1, j], mask[i + 1, j + 1]
                if a and d and not b and not c:
                    mask[i, j + 1] = mask[i + 1, j] = True; ch = True
                elif b and c and not a and not d:
                    mask[i, j] = mask[i + 1, j + 1] = True; ch = True
        if not ch:
            break

    # вершины: передняя и задняя сетки по узлам, используемым маской
    t = thickness_mm * U
    used = np.zeros((ny, nz), bool)
    for i in range(ny - 1):
        for j in range(nz - 1):
            if mask[i, j]:
                used[i:i + 2, j:j + 2] = True
    idx = -np.ones((ny, nz), np.int32)
    Vf, Vb, faces = [], [], []

    def vert(i, j, front):
        key = (i, j)
        if idx[i, j] < 0:
            push(np.array([[ys[i], zs[j]]]))
        return idx[i, j]

    # проще: сначала узлы, потом грани
    pts = np.argwhere(used)
    for i, j in pts:
        idx[i, j] = len(Vf)
        lev0 = float(A(np.array([ys[i]]), np.array([zs[j]]))[0])
        Vf.append([lev0, ys[i], zs[j]])
        Vb.append([lev0 - t, ys[i], zs[j]])
    n = len(Vf)
    V = np.asarray(Vf + Vb, float)

    def quad(a, b, c, d):
        faces.append([a, b, c]); faces.append([a, c, d])

    for i in range(ny - 1):
        for j in range(nz - 1):
            if not mask[i, j]:
                continue
            a, b = idx[i, j], idx[i + 1, j]
            c, d = idx[i + 1, j + 1], idx[i, j + 1]
            quad(a, b, c, d)                                   # перед
            quad(a + n, d + n, c + n, b + n)                   # зад
            # стенки там, где соседняя ячейка не заполнена
            if i == 0 or not mask[i - 1, j]:
                quad(a, d, d + n, a + n)
            if i == ny - 2 or not mask[i + 1, j]:
                quad(b, b + n, c + n, c)
            if j == 0 or not mask[i, j - 1]:
                quad(a, a + n, b + n, b)
            if j == nz - 2 or not mask[i, j + 1]:
                quad(d, c, c + n, d + n)
    out = __import__('trimesh').Trimesh(V, np.asarray(faces), process=True)
    out.merge_vertices()
    out.update_faces(out.nondegenerate_faces())
    out.fix_normals()
    if out.volume < 0:
        out.invert()
    return out


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
    ap.add_argument('--level', choices=['floor', 'crest', 'skin'], default='floor',
                    help='к какому уровню приводить: floor — впадины сот (панель в нише), '
                         'crest — гребни сот (видимая поверхность решётки, вровень с кузовом), '
                         'skin — наружная панель вокруг зоны')
    ap.add_argument('--offset-mm', type=float, default=0.0,
                    help='сдвинуть панель вперёд (+) или назад (-) на столько мм')
    ap.add_argument('--slab', action='store_true',
                    help='залить решётку плитой с гладкой передней гранью (булево объединение)')
    ap.add_argument('--slab-step-mm', type=float, default=0.6)
    ap.add_argument('--slab-half-mm', type=float, default=21.0,
                    help='полуширина плиты по y, мм (по умолчанию — кромка решётки)')
    ap.add_argument('--slab-z-min-mm', type=float, default=21.4)
    ap.add_argument('--slab-z-max-mm', type=float, default=31.2)
    ap.add_argument('--slab-level-mm', type=float, default=71.6,
                    help='высота панели по x, мм (по умолчанию — гребни сот исходной решётки)')
    ap.add_argument('--slab-sink-mm', type=float, default=1.8,
                    help='на сколько плита уходит под поверхность у кромки зоны')
    ap.add_argument('--slab-edge-cells', type=int, default=3,
                    help='на сколько ячеек расширить плиту за закрашенную зону')
    ap.add_argument('--slab-blend-mm', type=float, default=1.2,
                    help='ширина плавного схода плиты под кузов у кромок, мм')
    ap.add_argument('--slab-zone', action='store_true', default=True,
                    help='плита строго по закрашенной зоне (по умолчанию включено)')
    ap.add_argument('--slab-recess', action='store_true',
                    help='заливать только те ячейки, где кузов ниже уровня (кромка по «ватерлинии»)')
    ap.add_argument('--slab-hole-mm', type=float, default=3.4,
                    help='радиус лунки под эмблему (0 — без лунки)')
    ap.add_argument('--slab-thick-mm', type=float, default=1.5)
    ap.add_argument('--fill-recess', action='store_true',
                    help='залить нишу решётки до наружной поверхности (ровная панель под X)')
    ap.add_argument('--fill-depth-mm', type=float, default=0.30,
                    help='что считать нишей: всё, что ниже арки на столько мм '
                         '(отрицательное — захватить и гребни сот)')
    ap.add_argument('--diffuse-iters', type=int, default=0,
                    help='итераций диффузии внутри заплаты (0 — не трогать, заломов меньше)')
    ap.add_argument('--subdiv-mm', type=float, default=0.0,
                    help='дробить треугольники заплаты длиннее стольких мм печати')
    ap.add_argument('--protect-badge', action='store_true', default=True,
                    help='не трогать эмблему (по умолчанию включено)')
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

    if a.slab:
        # поле берём по гребням сот ИСХОДНОГО кузова: там поверхность решётки
        # симметричная и ровная, в отличие от уже сглаженного файла
        Pb = np.zeros((len(V0), 3)); Pb[:] = V0
        nb = vertex_normals(Pb, np.asarray(max(m3mf.read(a.base), key=lambda q: len(q[1].faces))[1].faces, np.int32))
        yy0, zz0 = Pb[:, 1] / U, Pb[:, 2] / U
        cand = ((np.abs(yy0) <= 18) & (zz0 >= 21.3) & (zz0 <= 31.7) & (nb[:, 0] > 0.5))
        crest = cand & (Pb[:, 0] >= np.percentile(Pb[cand, 0], 80))
        A = panel_field(Pb[crest, 1], Pb[crest, 2], Pb[crest, 0], clip=False)
        from scipy import ndimage as _ndi
        from scipy.interpolate import RegularGridInterpolator as _RGI
        from numpy.lib.stride_tricks import sliding_window_view as _swv

        _hs = a.slab_step_mm
        _ys = np.append(np.arange(-a.slab_half_mm, a.slab_half_mm, _hs), a.slab_half_mm)
        _zs = np.append(np.arange(a.slab_z_min_mm, a.slab_z_max_mm, _hs), a.slab_z_max_mm)
        _cy = (_ys[:-1] + _ys[1:]) / 2; _cz = (_zs[:-1] + _zs[1:]) / 2

        # маска плиты: строго по закрашенной зоне, промежутки между сотами залиты
        if a.slab_zone:
            zy = V[zone_f, 1] / U; zz = V[zone_f, 2] / U
            iy = np.clip(np.searchsorted(_ys, zy) - 1, 0, len(_cy) - 1)
            iz = np.clip(np.searchsorted(_zs, zz) - 1, 0, len(_cz) - 1)
            ok = (zy >= _ys[0]) & (zy <= _ys[-1]) & (zz >= _zs[0]) & (zz <= _zs[-1])
            mask = np.zeros((len(_cy), len(_cz)), bool)
            mask[iy[ok], iz[ok]] = True
            mask = _ndi.binary_closing(mask, np.ones((3, 3)), border_value=0)
            for arr in (mask, mask.T):          # сплошные полосы по строкам и столбцам
                any_ = arr.any(axis=1)
                lo = np.argmax(arr, axis=1)
                hi = arr.shape[1] - 1 - np.argmax(arr[:, ::-1], axis=1)
                idx = np.arange(arr.shape[1])[None, :]
                arr[(idx >= lo[:, None]) & (idx <= hi[:, None]) & any_[:, None]] = True
            mask = _ndi.binary_fill_holes(_ndi.binary_dilation(mask, np.ones((3, 3))))
            # рабочая кромка чуть шире закрашенного: сход уходит в соты, а зона
            # остаётся ровной до самого края
            mask = _ndi.binary_fill_holes(
                _ndi.binary_dilation(mask, np.ones((3, 3)), iterations=a.slab_edge_cells))
            # жёстко не выходим за габарит закрашенного: ниже полосы начинаются
            # фонари/губа бампера, выше — рамка
            _y_lim = np.abs(zy).max() + 0.6
            _z_lo, _z_hi = zz.min() - 0.2, zz.max() + 0.2
            _ok_cells = ((np.abs(_cy)[:, None] <= _y_lim) &
                         (_cz[None, :] >= _z_lo) & (_cz[None, :] <= _z_hi))
            mask &= _ok_cells
            print('  маска плиты: ячеек %d, y ±%.1f, z %.1f…%.1f мм' % (
                mask.sum(), _y_lim, _z_lo, _z_hi))
        else:
            mask = np.ones((len(_cy), len(_cz)), bool)

        # уровень панели — ровная плоскость на высоте гребней сот исходной
        # решётки (перцентиль по зоне): ровно то, что нужно под гравировку
        _GY, _GZ = np.meshgrid(_ys * U, _zs * U, indexing='ij')
        lev = np.full(_GY.shape, (a.slab_level_mm + a.offset_mm) * U)

        # у кромки маски плита плавно уходит под соты — обрыва-ступеньки не видно
        _d = _ndi.distance_transform_edt(mask) * _hs
        _d = np.pad(_d, ((0, 1), (0, 1)), mode='edge')
        _t = np.clip(_d / max(a.slab_blend_mm, 1e-6), 0.0, 1.0)
        lev = lev - (1.0 - _t * _t * (3.0 - 2.0 * _t)) * a.slab_sink_mm * U
        _rgi = _RGI((_ys * U, _zs * U), lev, bounds_error=False, fill_value=None)

        def Aenv(y, z):
            yy = np.atleast_1d(np.asarray(y, float)); zz = np.atleast_1d(np.asarray(z, float))
            yy, zz = np.broadcast_arrays(yy, zz)
            out = _rgi(np.column_stack([yy.ravel(), zz.ravel()])).reshape(yy.shape)
            return out if (np.ndim(y) or np.ndim(z)) else out.item()

        slab = slab_solid(Aenv, __import__('trimesh').Trimesh(V, F, process=False),
                          -a.slab_half_mm * U, a.slab_half_mm * U,
                          a.slab_z_min_mm * U, a.slab_z_max_mm * U,
                          a.slab_step_mm * U, a.slab_thick_mm,
                          only_recess=a.slab_recess, mask=mask)
        if a.slab_hole_mm > 0:                    # лунка под эмблему
            cyl = __import__('trimesh').creation.cylinder(
                radius=a.slab_hole_mm * U, height=6 * U, sections=64)
            cyl.apply_transform(__import__('trimesh').transformations.rotation_matrix(
                np.pi / 2, [0, 1, 0]))
            cyl.apply_translation([72.5 * U, 0, 25.7 * U])
            slab = __import__('trimesh').boolean.difference([slab, cyl], engine='manifold')
        print('  плита: граней %d, объём %.2f см³' % (len(slab.faces), slab.volume / U ** 3 / 1000))
        body = __import__('trimesh').Trimesh(V, F, process=False)   # как в файле, без склейки
        out = __import__('trimesh').boolean.union([body, slab], engine='manifold')
        out.remove_unreferenced_vertices()
        m3mf.write(a.out, [('Кузов', out)],
                   meta='Кузов: ниша решётки залита ровной панелью под гравировку')
        print('итог: вершин %d, граней %d, объём %.2f см³, watertight %s' % (
            len(out.vertices), len(out.faces), out.volume / U ** 3 / 1000, out.is_watertight))
        print('записано:', a.out)
        return

    moved = 0
    if a.fill_recess:
        skin = None
        if a.level == 'skin':
            yy0, zz0 = P[:, 1] / U, P[:, 2] / U
            outside = ((~zonec) & (np.abs(nrm[:, 0]) > 0.65) &
                       (np.abs(yy0) <= 27) & (zz0 >= 19.0) & (zz0 <= 33.0) &
                       (P[:, 0] > 65.0 * U))
            skin = panel_field(P[outside, 1], P[outside, 2], P[outside, 0])
        R, A = fill_recess(P, Fc, nrm, off, nbr, depth_mm=a.fill_depth_mm, skin=skin)
        if R is None:
            raise SystemExit('не нашёл нишу решётки в этом файле')
        lev = A(P[:, 1], P[:, 2]) + a.offset_mm * U
        # вес 1 в глубине, плавно к нулю у кромки области и у эмблемы: иначе на
        # стыке вылезают «зубцы» из растянутых граней
        rr = border_distance(R, off, nbr)
        w = np.where(R, np.clip(rr / 2.5, 0, 1), 0.0)
        w = w * w * (3 - 2 * w)
        yy0, zz0 = P[:, 1] / U, P[:, 2] / U
        bz, br = 25.7, 4.6                 # центр и радиус обода эмблемы, мм
        rad = np.hypot(yy0, zz0 - bz)
        w *= np.clip((rad - br) / 1.2, 0, 1)
        shift = w[R] * (lev[R] - P[R, 0])
        P[R, 0] += shift
        moved = int(R.sum())
        print('  ниша решётки: %d вершин, поднято на %.2f мм в среднем, макс %.2f мм' % (
            R.sum(), np.abs(shift).mean() / U, np.abs(shift).max() / U))
        if a.subdiv_mm > 0:
            Hf = A
            patch = [f for f in Fc if R[f[0]] and R[f[1]] and R[f[2]]]
            rest = [f for f in Fc if not (R[f[0]] and R[f[1]] and R[f[2]])]
            before = len(patch)
            Pout, patch2 = refine_patch(P, Fc, patch, lambda yy, zz: Hf(yy, zz) + a.offset_mm * U,
                                        a.subdiv_mm * U)
            P = np.asarray(Pout, float)
            # лёгкая диффузия внутри заплаты: убирает остаточные морщины,
            # кромка (где заплата стыкуется с кузовом) закреплена
            patch_arr = np.asarray(patch2, np.int64)
            inner = np.zeros(len(P), bool)
            inner[np.unique(patch_arr)] = True
            border = set()
            ecount = {}
            for f in patch_arr:
                for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                    k = (min(u, v), max(u, v))
                    ecount[k] = ecount.get(k, 0) + 1
            for (u, v), c in ecount.items():
                if c == 1:
                    border.add(u); border.add(v)
            free = inner.copy()
            free[list(border)] = False
            e2, off2, nbr2 = adjacency(patch_arr, len(P))
            for _ in range(max(0, a.diffuse_iters)):
                acc = np.zeros_like(P); cnt = np.zeros(len(P))
                for a0, b0 in e2:
                    acc[a0] += P[b0]; cnt[a0] += 1
                    acc[b0] += P[a0]; cnt[b0] += 1
                idx = np.where(free & (cnt > 0))[0]
                P[idx] += 0.3 * (acc[idx] / cnt[idx, None] - P[idx])
            Fc = np.asarray(rest + patch2, np.int32)
            print('  заплата раздроблена: %d -> %d граней, вершин %d' % (
                before, len(patch2), len(P)))
            m3mf.write(a.out, [('Кузов', __import__('trimesh').Trimesh(P, Fc, process=False))],
                       meta='Кузов: ниша решётки залита, панель под гравировку')
            print('вершин %d, граней %d' % (len(P), len(Fc)))
            print('записано:', a.out)
            return
    for k in ([] if a.fill_recess else range(cid)):
        sel = comp == k
        if sel.sum() < 200:                              # мелочь не трогаем
            continue
        if a.level == 'crest':
            # видимая поверхность решётки — гребни сот: аппроксимируем по верхней
            # части зоны, тогда панель встаёт вровень с кузовом, а не в нишу
            q = np.percentile(P[sel, 0], 80)
            floorv = P[sel, 0] >= q
        elif a.level == 'skin':
            # наружная панель вокруг зоны: кольца снаружи зоны, смотрящие вперёд
            yy0, zz0 = P[:, 1] / U, P[:, 2] / U
            outside = ((~zonec) & (np.abs(nrm[:, 0]) > 0.65) &
                       (np.abs(yy0) <= 27) & (zz0 >= 19.0) & (zz0 <= 33.0) &
                       (P[:, 0] > 65.0 * U))
            selx = np.where(outside)[0]
            floorv = None
            Hs = panel_field(P[selx, 1], P[selx, 2], P[selx, 0])
        else:
            floorv = np.abs(nrm[sel, 0]) > 0.65           # панель, не стенки ниши
            if floorv.sum() < 100:
                floorv = np.ones(sel.sum(), bool)
        if a.level != 'skin':
            Hs = panel_field(P[sel][floorv, 1], P[sel][floorv, 2], P[sel][floorv, 0])
        tgt = Hs(P[sel, 1], P[sel, 2]) + a.offset_mm * U
        h = (P[sel, 0] - tgt) / U
        near = (np.ones_like(h) if a.full
                else np.clip((SNAP_ZERO_MM - np.abs(h)) / (SNAP_ZERO_MM - SNAP_FULL_MM), 0, 1))
        if a.protect_badge:                                # эмблему не трогаем
            yy, zz = P[sel, 1] / U, P[sel, 2] / U
            badge = (np.abs(yy) <= 4.3) & (np.abs(zz - 25.7) <= 3.4)
            near = np.where(badge, 0.0, near)
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
