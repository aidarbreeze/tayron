#!/usr/bin/env python3
"""Аудит соответствия модели фотографиям: сравнение в одном кадре по зонам.

Регистрация: габаритный прямоугольник модели (по осям проекции) совмещается с
габаритным прямоугольником машины на фото (bbox не-белых пикселей). Проекция
ортогональная, поэтому для каждой зоны достаточно линейного отображения, и
кадр модели совпадает с кропом фото пиксель в пиксель (с учётом небольшой
анизотропии, которая есть в самих референсах).

Единицы модели: 1 юнит ≈ 4.9 мм (габарит 899.9 × 427.1 × 341.4 юнита).

Использование:
    python3 tools/audit_vs_photo.py all          # 4 сводных листа в корень
    python3 tools/audit_vs_photo.py overlay      # только наложения
    python3 tools/audit_vs_photo.py sheets       # только листы фото|модель
    python3 tools/audit_vs_photo.py zone <имя> [--model X.stl] [--out Y.png]
"""
import argparse
import os

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
MODEL_MAIN = ROOT + 'tayron_2023_optimized.stl'
MODEL_LITE = ROOT + 'tayron_2023_optimized_lite.stl'
PHOTO = {'front': ROOT + 'reference/tayron_2023_luxury_plus_front.png',
         'side': ROOT + 'reference/tayron_2023_luxury_plus_side.png',
         'rear': ROOT + 'reference/tayron_2023_luxury_plus_rear.png'}

# габарит машины на фото (x0, x1, y0, y1): не-белые пиксели без тени/водяного знака
BBOX = {'front': (450, 1932, 394, 1580),   # y1 — пятно контакта шин
        'side': (97, 2286, 568, 1308),
        'rear': (563, 1761, 608, 1687)}

# свет: от наблюдателя, чуть сверху и в сторону
LIGHT = {'front': (0.82, 0.18, 0.54), 'rear': (-0.82, -0.18, 0.54),
         'side': (0.15, 0.92, 0.36)}
VIEW_DIR = {'front': (1., 0., 0.), 'rear': (-1., 0., 0.), 'side': (0., 1., 0.)}

VIEW_RU = {'front': 'перед', 'side': 'бок', 'rear': 'корма'}
_MESH = {}


def load(path):
    if path not in _MESH:
        _MESH[path] = trimesh.load(path, force='mesh')
    return _MESH[path]


def project(mesh, view):
    """u, v — координаты в плоскости кадра, d — глубина (больше = ближе к камере)."""
    V = mesh.vertices
    if view == 'side':
        return V[:, 0], V[:, 2], V[:, 1]
    if view == 'rear':
        return -V[:, 1], V[:, 2], -V[:, 0]
    return V[:, 1], V[:, 2], V[:, 0]


def frame(mesh, view):
    """Масштабы и обратное отображение «пиксель фото -> (u, v) модели»."""
    u, v, _ = project(mesh, view)
    bx0, bx1, by0, by1 = BBOX[view]
    sx = (bx1 - bx0) / (u.max() - u.min())
    sy = (by1 - by0) / (v.max() - v.min())

    def u_of(px):
        return u.max() - (px - bx0) / sx if view != 'side' else u.min() + (px - bx0) / sx

    def v_of(py):
        return v.min() + (by1 - py) / sy

    return u_of, v_of, sx, sy


def shade(mesh, view):
    """Цвета граней: ламберт от наблюдателя + блик, с учётом сортировки по глубине."""
    u, v, d = project(mesh, view)
    F = mesh.faces
    tri = np.stack([np.stack([u[F[:, i]], v[F[:, i]]], 1) for i in range(3)], 1)
    order = np.argsort(d[F].mean(1))
    L = np.array(LIGHT[view]) / np.linalg.norm(LIGHT[view])
    lam = np.clip(mesh.face_normals[order] @ L, 0, 1)
    sh = np.clip(0.10 + 0.88 * lam ** 0.8 + 0.08 * lam ** 12, 0, 1)
    base = np.array([0.88, 0.90, 0.93])
    return tri[order], base[None, :] * sh[:, None]


def render(mesh, view, box, width_px, out, ss=1.7, dpi=110, light=None):
    """Рендер зоны модели так, чтобы кадр совпал с кропом фото (пиксель в пиксель)."""
    x0, x1, y0, y1 = box
    u_of, v_of, _, _ = frame(mesh, view)
    tri, colors = shade(mesh, view)
    w = width_px * ss / dpi
    h = width_px * ss * (y1 - y0) / (x1 - x0) / dpi
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    ax.add_collection(PolyCollection(tri, facecolors=colors, edgecolors='none'))
    ax.set_xlim(u_of(x0), u_of(x1))
    ax.set_ylim(v_of(y1), v_of(y0))
    ax.set_aspect('auto')
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def overlay(view, out, edge_lw=0.35, area_alpha=0.18):
    """Силуэт и рёбра перегиба модели поверх фото — проверка регистрации."""
    m = load(MODEL_MAIN)
    u, v, d = project(m, view)
    F = m.faces
    tri = np.stack([np.stack([u[F[:, i]], v[F[:, i]]], 1) for i in range(3)], 1)
    order = np.argsort(d[F].mean(1))
    tri = tri[order]
    VD = np.array(VIEW_DIR[view])
    N = m.face_normals
    fa = m.face_adjacency
    both = np.minimum(N[fa[:, 0]] @ VD, N[fa[:, 1]] @ VD) > 0.30
    keep = (m.face_adjacency_angles > np.radians(18)) & both
    ed = m.face_adjacency_edges[keep]
    segs = np.stack([np.stack([u[ed[:, 0]], v[ed[:, 0]]], 1),
                     np.stack([u[ed[:, 1]], v[ed[:, 1]]], 1)], 1)

    im = Image.open(PHOTO[view]).convert('RGB')
    W, H = im.size
    bx0, bx1, by0, by1 = BBOX[view]
    sx = (bx1 - bx0) / (u.max() - u.min())
    sy = (by1 - by0) / (v.max() - v.min())
    fu = (lambda a: bx0 + (a - u.min()) * sx) if view == 'side' else (lambda a: bx0 + (u.max() - a) * sx)
    fv = lambda a: by1 - (a - v.min()) * sy

    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im, interpolation='lanczos')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    T = tri.copy()
    T[:, :, 0] = fu(T[:, :, 0])
    T[:, :, 1] = fv(T[:, :, 1])
    ax.add_collection(PolyCollection(T, facecolors=(0.9, 0.1, 0.1, area_alpha), edgecolors='none'))
    S = segs.copy()
    S[:, :, 0] = fu(S[:, :, 0])
    S[:, :, 1] = fv(S[:, :, 1])
    ax.add_collection(LineCollection(S, colors=(0.75, 0.0, 0.0, 0.9), linewidths=edge_lw))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------- зоны и листы
CELL_W = 620
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_R = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

ROWS = {
    'front': [
        ('Фары + решётка (слева)', (560, 1180, 860, 1110)),
        ('Центр решётки, эмблема', (950, 1520, 900, 1120)),
        ('Накладка ПТФ/ДХО, нижняя решётка', (520, 1080, 1080, 1470)),
        ('Номерной знак, защита', (930, 1530, 1100, 1480)),
        ('Зеркало (вид спереди)', (400, 780, 620, 830)),
    ],
    'rear': [
        ('Задний фонарь (левый)', (600, 1020, 930, 1080)),
        ('Центр световой полосы, эмблема', (1000, 1420, 930, 1080)),
        ('Надписи TAYRON / 330 TSI', (1000, 1660, 1055, 1240)),
        ('Задний бампер', (590, 1180, 1300, 1580)),
        ('Рейлинги и плавник', (620, 1500, 560, 690)),
    ],
    'side': [
        ('Задний фонарь, заднее крыло', (60, 480, 740, 950)),
        ('Лючок бензобака, ручка задней двери', (300, 640, 760, 920)),
        ('Ручка передней двери, шов', (1050, 1420, 780, 910)),
        ('Зеркало, A-стойка', (1320, 1620, 620, 840)),
        ('Переднее колесо, арка', (1650, 2100, 950, 1320)),
        ('D-стойка, форточка', (330, 640, 620, 840)),
    ],
}


def _cell(im, caption, w, h, font):
    im = im.convert('RGB').resize((w, h - 30), Image.LANCZOS)
    c = Image.new('RGB', (w, h), 'white')
    c.paste(im, (0, 30))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, w, 29], fill='#222')
    d.text((8, 4), caption, fill='white', font=font)
    d.rectangle([0, 0, w - 1, h - 1], outline='#b0b0b0')
    return c


def sheet(view, out, model=MODEL_MAIN, tag='16 МБ'):
    fnt = ImageFont.truetype(FONT_B, 19)
    rows, tmp = [], '/tmp/_audit_mod.png'
    for title, box in ROWS[view]:
        x0, x1, y0, y1 = box
        h = int(CELL_W * (y1 - y0) / (x1 - x0)) + 30
        ph = Image.open(PHOTO[view]).crop((x0, y0, x1, y1))
        render(load(model), view, box, CELL_W, tmp)
        rows.append((_cell(ph, f'ФОТО · {title}', CELL_W, h, fnt),
                     _cell(Image.open(tmp), f'МОДЕЛЬ ({tag}) · {title}', CELL_W, h, fnt)))
    W = CELL_W * 2 + 10
    H = 46 + sum(r[0].height for r in rows)
    img = Image.new('RGB', (W, H), '#e8e8e8')
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 44], fill='#0d0d0d')
    d.text((10, 11), f'АУДИТ · {VIEW_RU[view]}: фото и модель в одном кадре', fill='white', font=fnt)
    y = 46
    for a, b in rows:
        img.paste(a, (0, y))
        img.paste(b, (CELL_W + 10, y))
        y += a.height
    img.quantize(colors=200, dither=Image.FLOYDSTEINBERG).save(out, optimize=True)
    print(out, img.size)


def overlay_sheet(out):
    fnt = ImageFont.truetype(FONT_R, 17)
    tiles = []
    for v in ('front', 'side', 'rear'):
        tmp = f'/tmp/_ov_{v}.png'
        overlay(v, tmp)
        im = Image.open(tmp).convert('RGB')
        im = im.resize((760, int(760 * im.height / im.width)), Image.LANCZOS)
        c = Image.new('RGB', (760, im.height + 28), 'white')
        c.paste(im, (0, 28))
        d = ImageDraw.Draw(c)
        d.rectangle([0, 0, 760, 26], fill='#222')
        d.text((8, 3), f'НАЛОЖЕНИЕ · {VIEW_RU[v]}: силуэт и рёбра модели поверх фото', fill='white', font=fnt)
        tiles.append(c)
    H = sum(t.height for t in tiles) + 4 * len(tiles)
    img = Image.new('RGB', (760, H), '#ddd')
    y = 0
    for t in tiles:
        img.paste(t, (0, y))
        y += t.height + 4
    img.quantize(colors=200, dither=Image.FLOYDSTEINBERG).save(out, optimize=True)
    print(out, img.size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['all', 'overlay', 'sheets'])
    ap.add_argument('--model', default=MODEL_MAIN)
    ap.add_argument('--tag', default='16 МБ')
    a = ap.parse_args()
    if a.cmd in ('all', 'overlay'):
        overlay_sheet(ROOT + 'audit_overlay.png')
    if a.cmd in ('all', 'sheets'):
        for v in ('front', 'rear', 'side'):
            sheet(v, ROOT + f'audit_{v}.png', a.model, a.tag)
