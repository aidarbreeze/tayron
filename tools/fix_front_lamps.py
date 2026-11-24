#!/usr/bin/env python3
"""Фары и ПТФ по фотографии: выбрать контур с фото и вдавить его в передок.

Зачем: в исходном скане фары не восстановились (зеркальный чёрный пластик для
фотограмметрии «невидим»), поэтому на модели вместо фары — гладкая выпуклость,
смещённая вниз на 30–40 px кадра. Конвейер уплотняет сетку, но форму не меняет.
Здесь форма фары берётся с фотографии: контур тёмного корпуса фары (и накладки
ПТФ) проецируется на передок и внутрь контура вырезается плавная ниша.

Геометрия не перестраивается: вершины внутри контура смещаются по −X на
глубину, которая плавно гаснет к границе контура (smoothstep), поэтому сетка
остаётся замкнутой и связной.

    python3 tools/fix_front_lamps.py --in tayron_2023_optimized_v2.stl \
        --out tayron_2023_optimized_v3.stl --depth 4.0
"""
import argparse
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_vs_photo as A                      # noqa: E402
import photo_texture as P                       # noqa: E402

ROOT = A.ROOT
# окна поиска (кадр фото, пиксели) и порог «тёмного» для каждого элемента
# окно (x0, x1, y0, y1) — где ищем фонарь/фару, и полоса строк, внутри которой
# берутся тёмные прогоны. Контур строится по колонкам: верх и низ корпуса.
ELEMENTS = [
    ('фара слева', (130, 470, 470, 617), 130),
    ('фара справа', (1030, 1370, 470, 617), 130),
]
TAPER_PX = 12.0         # ширина плавного перехода у границы контура


def _envelopes(ph, x0, x1, y0, y1, thr, min_run=8):
    """Верх и низ корпуса фары по колонкам: границы тёмных прогонов."""
    top = np.full(x1 - x0, np.nan)
    bot = np.full(x1 - x0, np.nan)
    dark = ph < thr
    for i, x in enumerate(range(x0, x1)):
        col = dark[y0:y1, x]
        runs, k = [], 0
        while k < len(col):
            if col[k]:
                j = k
                while j < len(col) and col[j]:
                    j += 1
                if j - k >= min_run:
                    runs.append((y0 + k, y0 + j - 1))
                k = j
            else:
                k += 1
        if runs:
            top[i], bot[i] = runs[0][0], runs[-1][1]
    return top, bot


def _fill_nan(a):
    ok = ~np.isnan(a)
    if not ok.any():
        return a
    idx = np.where(ok)[0]
    return np.interp(np.arange(len(a)), idx, a[idx])


def photo_masks(view='front'):
    """Маски элементов по фото: полоса между верхом и низом корпуса фары."""
    ph = np.asarray(P.photo_crop(view, A.BBOX[view]).convert('L'), float)
    masks = []
    for name, (x0, x1, y0, y1), thr in ELEMENTS:
        top, bot = _envelopes(ph, x0, x1, y0, y1, thr)
        if np.isnan(top).all():
            print(f'  {name}: маска пустая — пропуск')
            continue
        top, bot = _fill_nan(top), _fill_nan(bot)
        top = ndi.median_filter(top, 25)
        bot = ndi.median_filter(bot, 25)
        m = np.zeros_like(ph, bool)
        for i, x in enumerate(range(x0, x1)):
            m[int(top[i]):int(bot[i]) + 1, x] = True
        m = ndi.binary_closing(m, np.ones((3, 3)))
        m = ndi.binary_fill_holes(m)
        ys, xs = np.where(m)
        print(f'  {name}: пикс {m.sum()}, x {xs.min()}..{xs.max()}, '
              f'y {ys.min()}..{ys.max()} (h ≈ {int(np.nanmedian(bot - top))})')
        masks.append((name, m))
    return masks


def masks_png(masks, out):
    """Проверка масок: контуры поверх фото."""
    ph = P.photo_crop('front', A.BBOX['front']).convert('RGB')
    a = np.asarray(ph).copy()
    for i, (name, m) in enumerate(masks):
        edge = m & ~ndi.binary_erosion(m, np.ones((3, 3)))
        a[edge] = [(255, 40, 40), (40, 200, 40), (40, 120, 255), (255, 200, 40)][i % 4]
    Image.fromarray(a).save(out)
    print('маски:', out)


def apply_depth(mesh, view, masks, depth, taper_px=TAPER_PX):
    """Сместить вершины внутри контуров по −X: ниша с плавным краем."""
    x0, x1, y0, y1 = A.BBOX[view]
    u_of, v_of, sx, sy = A.frame(mesh, view)
    u, v, _ = A.project(mesh, view)
    # пиксель каждого пикселя кадра -> u,v: обратное к u_of/v_of
    # кадр = кроп фото, поэтому координаты кропа:
    # u_of(px) = u.max() − (px − x0)/sx  =>  px_кропа = (u.max() − u)·sx
    px = (u.max() - u) * sx
    py = (y1 - y0) - (v - v.min()) * sy
    V = mesh.vertices.copy()
    moved = np.zeros(len(V), bool)
    total = 0.0
    for name, m in masks:
        dist = ndi.distance_transform_edt(m)
        ix = np.clip(px.astype(int), 0, m.shape[1] - 1)
        iy = np.clip(py.astype(int), 0, m.shape[0] - 1)
        inside = m[iy, ix]
        if not inside.any():
            continue
        d_px = dist[iy, ix]
        # smoothstep по расстоянию до границы контура
        t = np.clip(d_px / taper_px, 0, 1)
        wgt = t * t * (3 - 2 * t)
        wgt = np.where(inside, wgt, 0.0)
        # только передние поверхности: нормаль смотрит на камеру (+X)
        front = mesh.vertex_normals[:, 0] > 0.05
        wgt = wgt * front
        n = int((wgt > 0).sum())
        V[:, 0] -= depth * wgt
        moved |= wgt > 0
        total += n
        print(f'  {name}: смещено вершин {n}, макс. глубина {depth * wgt.max():.2f}')
    out = trimesh.Trimesh(vertices=V, faces=mesh.faces, process=False)
    out.merge_vertices()
    print(f'  всего вершин смещено: {int(moved.sum())} из {len(V)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', default=ROOT + 'tayron_2023_optimized_v2.stl')
    ap.add_argument('--out', dest='dst', default=ROOT + 'tayron_2023_optimized_v3.stl')
    ap.add_argument('--depth', type=float, default=4.0, help='глубина ниши, юниты')
    ap.add_argument('--taper', type=float, default=TAPER_PX,
                    help='ширина плавного края ниши, пиксели кадра')
    ap.add_argument('--masks-only', action='store_true')
    a = ap.parse_args()
    print('маски по фото:')
    masks = photo_masks('front')
    masks_png(masks, ROOT + 'lamp_masks_front.png')
    if a.masks_only:
        return
    mesh = trimesh.load(a.src, force='mesh')
    print(f'вход: {a.src}: {len(mesh.faces)} граней, watertight={mesh.is_watertight}')
    fixed = apply_depth(mesh, 'front', masks, a.depth, a.taper)
    fixed.export(a.dst)
    print(f'выход: {a.dst}: {len(fixed.faces)} граней, watertight={fixed.is_watertight}, '
          f'объём {fixed.volume:.1f}')


if __name__ == '__main__':
    main()
