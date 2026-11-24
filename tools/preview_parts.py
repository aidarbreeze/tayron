#!/usr/bin/env python3
"""Быстрые превью зон модели (зеркала, дворники) без фото-регистрации."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
LIGHT = {'side': (0.15, 0.92, 0.36), 'top': (0.05, 0.35, 0.93), 'front': (0.82, 0.18, 0.54),
         'rear': (-0.82, -0.18, 0.54)}


def project(V, view):
    if view == 'side':
        return V[:, 0], V[:, 2], V[:, 1]
    if view == 'top':
        return V[:, 0], V[:, 1], V[:, 2]
    if view == 'rear':
        return -V[:, 1], V[:, 2], -V[:, 0]
    return V[:, 1], V[:, 2], V[:, 0]


def tile(mesh, view, ulim, vlim, width=620, dpi=115):
    u, v, d = project(mesh.vertices, view)
    F = mesh.faces
    tri = np.stack([np.stack([u[F[:, i]], v[F[:, i]]], 1) for i in range(3)], 1)
    order = np.argsort(d[F].mean(1))
    L = np.array(LIGHT[view]); L /= np.linalg.norm(L)
    lam = np.clip(mesh.face_normals[order] @ L, 0, 1)
    sh = np.clip(0.10 + 0.88 * lam ** 0.8 + 0.08 * lam ** 12, 0, 1)
    h = width * (vlim[1] - vlim[0]) / (ulim[1] - ulim[0])
    fig, ax = plt.subplots(figsize=(width / dpi, h / dpi), dpi=dpi)
    ax.add_collection(PolyCollection(tri[order],
                                     facecolors=np.array([0.88, 0.90, 0.93])[None, :] * sh[:, None],
                                     edgecolors='none'))
    ax.set_xlim(*ulim); ax.set_ylim(*vlim); ax.set_aspect('auto'); ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    tmp = '/tmp/_prev.png'
    fig.savefig(tmp, dpi=dpi); plt.close(fig)
    return Image.open(tmp).convert('RGB').copy()


def sheet(tiles, out, titles, heading):
    w = max(t.width for t in tiles); h = sum(t.height + 30 for t in tiles) + 44
    img = Image.new('RGB', (w, h), '#e8e8e8')
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 42); fs = ImageFont.truetype(FONT, 34)
    d.rectangle([0, 0, w, 40], fill='#111'); d.text((8, 4), heading, fill='white', font=fs)
    y = 44
    for t, title in zip(tiles, titles):
        d.rectangle([0, y, w, y + 28], fill='#333'); d.text((8, y + 1), title, fill='white', font=fs)
        img.paste(t, (0, y + 30)); y += t.height + 30
    img.save(out, optimize=True)
    print(out, img.size)


def render_regions(mesh, prefix):
    tiles, titles = [], []
    tiles.append(tile(mesh, 'side', (55, 165), (30, 115)))
    titles.append('сбоку: X × Z (зеркало)')
    tiles.append(tile(mesh, 'top', (55, 165), (150, 235)))
    titles.append('сверху: X × Y (зеркало)')
    tiles.append(tile(mesh, 'front', (-230, 230), (20, 130)))
    titles.append('спереди: Y × Z (оба зеркала)')
    sheet(tiles, prefix + '_mirror_side.png', titles, 'ЗЕРКАЛА')
