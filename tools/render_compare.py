#!/usr/bin/env python3
"""Рендеры «до/после» переноса деталей с Tiguan на Tayron.

    python3 tools/render_compare.py [--before ...stl] [--after ...stl]
                                    [--out-prefix /tmp/compare]

Пишет два листа: <prefix>_before_after.png (машина целиком) и
<prefix>_details.png (крупно: зеркало, задний дворник, передние дворники).
"""
import argparse
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
HEAD = {'side': (0.15, 0.92, 0.36), 'top': (0.05, 0.35, 0.93),
        'front': (0.82, 0.18, 0.54), 'iso': (0.20, 0.55, 0.80),
        'iso_front': (0.35, 0.62, 0.70), 'rear': (-0.75, -0.15, 0.60)}


def load(path):
    m = trimesh.load(path, force='mesh')
    m.merge_vertices()
    return m


def rotated(mesh, th_deg):
    R = trimesh.transformations.rotation_matrix(np.deg2rad(th_deg), [0, 0, 1])[:3, :3]
    return mesh.vertices @ R.T, mesh.face_normals @ R.T


def projection(mesh, kind):
    """u, v, глубина, повёрнутые нормали."""
    if kind == 'iso':
        V, N = rotated(mesh, 35.0)
        return V[:, 0], V[:, 2] + 0.42 * V[:, 1], V[:, 1], N
    if kind == 'iso_front':
        V, N = rotated(mesh, -35.0)
        return V[:, 0], V[:, 2] + 0.42 * V[:, 1], V[:, 1], N
    V, N = mesh.vertices, mesh.face_normals
    if kind == 'side':
        return V[:, 0], V[:, 2], V[:, 1], N
    if kind == 'top':
        return V[:, 0], V[:, 1], V[:, 2], N
    if kind == 'rear':
        return -V[:, 1], V[:, 2], -V[:, 0], N
    return V[:, 1], V[:, 2], V[:, 0], N


def crop(mesh, lo, hi):
    """Чистый вырез боксом (булево), чтобы края были ровные."""
    box = trimesh.creation.box(extents=np.asarray(hi, float) - np.asarray(lo, float))
    box.apply_translation((np.asarray(lo, float) + np.asarray(hi, float)) / 2)
    sub = trimesh.boolean.intersection([mesh, box], engine='manifold')
    if sub is None or len(sub.faces) == 0:
        raise SystemExit(f'пустой кроп {lo} .. {hi}')
    return sub


def tile(mesh, kind, ulim=None, vlim=None, width=880, dpi=110, pad=0.04):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    u, v, depth, N = projection(mesh, kind)
    if ulim is None:
        du = (u.max() - u.min()) * pad
        ulim = (u.min() - du, u.max() + du)
    if vlim is None:
        dv = (v.max() - v.min()) * pad
        vlim = (v.min() - dv, v.max() + dv)
    F = mesh.faces
    tri = np.stack([np.stack([u[F[:, i]], v[F[:, i]]], 1) for i in range(3)], 1)
    order = np.argsort(depth[F].mean(1))
    L = np.array(HEAD[kind]); L /= np.linalg.norm(L)
    lam = np.clip(N[order] @ L, 0, 1)
    sh = np.clip(0.10 + 0.86 * lam ** 0.8 + 0.06 * lam ** 10, 0, 1)
    h = width * (vlim[1] - vlim[0]) / (ulim[1] - ulim[0])
    fig, ax = plt.subplots(figsize=(width / dpi, h / dpi), dpi=dpi)
    ax.add_collection(PolyCollection(tri[order],
                                     facecolors=np.array([0.86, 0.89, 0.93])[None, :] * sh[:, None],
                                     edgecolors='none'))
    ax.set_xlim(*ulim); ax.set_ylim(*vlim); ax.set_aspect('auto'); ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    tmp = '/tmp/_cmp.png'
    fig.savefig(tmp, dpi=dpi); plt.close(fig)
    return Image.open(tmp).convert('RGB').copy()


def sheet(rows, out, heading):
    w = max(a.width + b.width for _, a, b in rows)
    h = sum(max(a.height, b.height) + 34 for _, a, b in rows) + 46
    img = Image.new('RGB', (w, h), '#e8e8e8')
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 40], fill='#111')
    d.text((10, 3), heading, fill='white', font=ImageFont.truetype(FONT, 36))
    d.text((w // 2 - 60, 3), 'ДО      |      ПОСЛЕ', fill='#bbb', font=ImageFont.truetype(FONT, 30))
    y = 44
    for title, a, b in rows:
        d.rectangle([0, y, w, y + 30], fill='#333')
        d.text((10, y + 1), title, fill='white', font=ImageFont.truetype(FONT, 30))
        img.paste(a, (0, y + 32)); img.paste(b, (a.width, y + 32))
        y += max(a.height, b.height) + 34
    img.save(out, optimize=True)
    print(out, img.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', default=ROOT + 'tayron_2023_optimized.stl')
    ap.add_argument('--after', default=ROOT + 'tayron_2023_optimized_v2.stl')
    ap.add_argument('--out-prefix', default='/tmp/compare')
    a = ap.parse_args()
    before, after = load(a.before), load(a.after)
    print('до:', len(before.faces), 'граней; после:', len(after.faces))

    full = [('сбоку', 'side', (-470, 470), (-195, 195)),
            ('сзади-сбоку', 'iso', (-505, 505), (-190, 215)),
            ('спереди-сбоку', 'iso_front', (-505, 505), (-190, 215))]
    rows = []
    for title, kind, ul, vl in full:
        rows.append((title, tile(before, kind, ul, vl), tile(after, kind, ul, vl)))
    sheet(rows, a.out_prefix + '_before_after.png', 'TAYRON: ДО / ПОСЛЕ')

    det = [('зеркало (правый борт), сбоку', 'side', (78, 150), (32, 112), (60, 150, 30), (155, 235, 115)),
           ('задний дворник, сзади', 'rear', None, None, (-436, -60, 78), (-300, 60, 155)),
           ('передние дворники', 'top', None, None, (110, -160, 70), (200, 160, 135))]
    rows = []
    for title, kind, ul, vl, lo, hi in det:
        cb = crop(before, lo, hi)
        ca = crop(after, lo, hi)
        rows.append((title, tile(cb, kind, ul, vl, width=620), tile(ca, kind, ul, vl, width=620)))
    sheet(rows, a.out_prefix + '_details.png', 'TAYRON: КРУПНО')


if __name__ == '__main__':
    main()
