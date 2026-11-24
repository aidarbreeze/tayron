#!/usr/bin/env python3
"""Текстура с фотографии на модель + наложение «фото 30 % поверх модели».

Зачем: проверить, попадают ли элементы геометрии (фары, решётка, фонари,
бампер) под рисунок реальной машины. Регистрация та же, что в аудите
(`tools/audit_vs_photo.py`): габарит модели совмещается с габаритом машины на
фото, проекция ортогональная, кадр модели совпадает с кропом фото пиксель в
пиксель.

Что делает:
  1. рендерит модель в кадре фотографии (вид «перед» / «корма»);
  2. переносит цвет фотографии на пиксели модели — это и есть текстура:
     цвет берётся из фото, форма и свет — из геометрии (ламберт по нормалям);
  3. складывает фотографию поверх рендера модели с прозрачностью 30 %;
  4. строит карту совпадения контуров: рёбра модели (красные) и контуры фото
     (синие) в одном кадре + численная оценка «доля рёбер модели, попавших на
     контуры фото» по каждой зоне.

Использование:
    python3 tools/photo_texture.py both
    python3 tools/photo_texture.py front --model tayron_2023_optimized_v2.stl
    python3 tools/photo_texture.py zones --src "tayron lite 2 (1) (1).stl"
"""
import argparse
import os
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_vs_photo as A                     # noqa: E402  (та же регистрация)

ROOT = A.ROOT
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_R = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
VIEW_RU = {'front': 'ПЕРЕД', 'rear': 'КОРМА'}
ALPHA = 0.30                                   # прозрачность фото поверх модели
WATERMARK = {'front': (1880, 1490, 2320, 1720), 'rear': (1850, 1520, 2300, 1760),
             'side': (1880, 1490, 2330, 1720)}


# ------------------------------------------------------------------ рендеры
def render_rgb(mesh, view, box, dpi=100):
    """Рендер модели в кадре кропа фото: массив (h, w, 3) uint8, ровно по пикселям."""
    x0, x1, y0, y1 = box
    u_of, v_of, _, _ = A.frame(mesh, view)
    tri, colors = A.shade(mesh, view)
    w, h = int(x1 - x0), int(y1 - y0)
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_collection(PolyCollection(tri, facecolors=colors, edgecolors='none'))
    ax.set_xlim(u_of(x0), u_of(x1))
    ax.set_ylim(v_of(y1), v_of(y0))
    ax.set_aspect('auto')
    ax.axis('off')
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if buf.shape[:2] != (h, w):
        buf = np.asarray(Image.fromarray(buf).resize((w, h), Image.LANCZOS))
    return buf


def photo_crop(view, box, drop_watermark=True):
    """Кроп фото по габариту машины; box = (x0, x1, y0, y1), как в аудите."""
    im = Image.open(A.PHOTO[view]).convert('RGB')
    x0, x1, y0, y1 = box
    crop = im.crop((int(x0), int(y0), int(x1), int(y1)))
    if drop_watermark:
        wx0, wy0, wx1, wy1 = WATERMARK[view]
        a = np.asarray(crop).copy()
        x0, y0 = int(box[0]), int(box[2])
        mx0, my0 = max(0, wx0 - x0), max(0, wy0 - y0)
        mx1, my1 = min(a.shape[1], wx1 - x0), min(a.shape[0], wy1 - y0)
        if mx1 > mx0 and my1 > my0:
            patch = a[my0:my1, mx0:mx1].astype(float)
            fill = np.percentile(patch.reshape(-1, 3), 90, axis=0)   # тон фона
            a[my0:my1, mx0:mx1] = fill.astype(np.uint8)
        crop = Image.fromarray(a)
    return crop


def model_mask(rgb, thresh=250):
    """Где модель: пиксели, отличные от белого фона."""
    return np.any(np.asarray(rgb, float) < thresh, axis=2)


def texture_from_photo(photo, model_rgb):
    """Цвет — из фото, свет и форма — из геометрии модели."""
    ph = np.asarray(photo, float) / 255.0
    sh = np.asarray(model_rgb, float).mean(2) / 255.0
    mask = model_mask(model_rgb)
    if mask.any():
        ref = np.percentile(sh[mask], 88)
        sh = np.clip(sh / max(ref, 1e-3), 0, 1)
    lit = 0.55 + 0.45 * sh
    out = np.clip(ph * lit[:, :, None], 0, 1)
    bg = np.ones_like(out)
    out = np.where(mask[:, :, None], out, bg)
    return Image.fromarray((out * 255).astype(np.uint8))


def blend(photo, model_rgb, alpha=ALPHA):
    """Фото с прозрачностью alpha поверх рендера модели."""
    a = np.asarray(photo, float) / 255.0
    b = np.asarray(model_rgb, float) / 255.0
    return Image.fromarray((((1 - alpha) * b + alpha * a).clip(0, 1) * 255).astype(np.uint8))


# ------------------------------------------------------------------ контуры
def model_edges(mesh, view, box, size, lw=0.9, angle_deg=28.0, facing=0.30):
    """Маска рёбер перегиба модели в кадре фото (белое — ребро)."""
    x0, x1, y0, y1 = box
    u, v, _ = A.project(mesh, view)
    u_of, v_of, _, _ = A.frame(mesh, view)
    VD = np.array(A.VIEW_DIR[view])
    N = mesh.face_normals
    fa = mesh.face_adjacency
    both = np.minimum(N[fa[:, 0]] @ VD, N[fa[:, 1]] @ VD) > facing
    keep = (mesh.face_adjacency_angles > np.radians(angle_deg)) & both
    ed = mesh.face_adjacency_edges[keep]
    segs = np.stack([np.stack([u[ed[:, 0]], v[ed[:, 0]]], 1),
                     np.stack([u[ed[:, 1]], v[ed[:, 1]]], 1)], 1)
    w, h = size
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.add_collection(LineCollection(segs, colors=(0, 0, 0, 1), linewidths=lw))
    ax.set_xlim(u_of(x0), u_of(x1))
    ax.set_ylim(v_of(y1), v_of(y0))
    ax.set_aspect('auto')
    ax.axis('off')
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if buf.shape[:2] != (h, w):
        buf = np.asarray(Image.fromarray(buf).resize((w, h), Image.LANCZOS))
    return buf.mean(2) < 128


def photo_edges(photo, thr=25):
    g = np.asarray(photo.convert('L').filter(ImageFilter.GaussianBlur(1.2)), float)
    gx = np.abs(np.gradient(g, axis=1))
    gy = np.abs(np.gradient(g, axis=0))
    return np.hypot(gx, gy) > thr


def dilate(mask, r=3):
    im = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.MaxFilter(2 * r + 1))) > 0


def contours_over_photo(photo, medges, pedges_mask):
    """Фото (приглушённое) + контуры фото синим + рёбра модели красным."""
    base = np.asarray(photo, float) / 255.0
    base = 0.35 + 0.65 * base                        # приглушить, чтобы линии читались
    ped = dilate(pedges_mask, 1)
    base[ped] = [0.10, 0.35, 0.95]                   # контуры фото — синие
    med = dilate(medges, 0)
    base[med] = [0.90, 0.10, 0.10]                   # рёбра модели — красные
    return Image.fromarray((np.clip(base, 0, 1) * 255).astype(np.uint8))


def coverage(medges, pedges_mask, sl=None, r=3):
    """Две доли: рёбра модели на контурах фото и наоборот (допуск r пикселей)."""
    m = medges if sl is None else medges[sl]
    p = pedges_mask if sl is None else pedges_mask[sl]
    out = {}
    if m.any() and p.any():
        out['model_on_photo'] = float((m & dilate(p, r)).sum()) / float(m.sum())
        out['photo_on_model'] = float((p & dilate(m, r)).sum()) / float(p.sum())
    else:
        out['model_on_photo'] = out['photo_on_model'] = None
    return out


def photo_mask(photo, thr=242):
    """Машина на фото: всё, что темнее фона."""
    a = np.asarray(photo.convert('L'), float)
    return a < thr


def footprint(mask):
    """Силуэт целиком: заливаем внутренние светлые пятна (стекло, блики)."""
    from scipy import ndimage as ndi
    m = ndi.binary_fill_holes(mask)
    return ndi.binary_opening(m, np.ones((3, 3)))


def silhouette_diff(photo, model_rgb, split=0.55):
    """Силуэты: совпало — серым, только модель — красным, только фото — синим.
    split — граница «верх/низ» кадра для отдельной оценки (перспектива)."""
    pm = footprint(photo_mask(photo))
    mm = footprint(model_mask(model_rgb))
    both = pm & mm
    img = np.ones((pm.shape[0], pm.shape[1], 3), float)
    img[both] = [0.80, 0.80, 0.80]
    img[mm & ~pm] = [0.90, 0.15, 0.15]
    img[pm & ~mm] = [0.15, 0.35, 0.95]
    def _iou(a, b):
        return float((a & b).sum()) / max(float((a | b).sum()), 1.0)
    H = pm.shape[0]
    cut = int(H * split)
    iou = _iou(pm, mm)
    return (Image.fromarray((img * 255).astype(np.uint8)), iou,
            {'full': iou, 'top': _iou(pm[:cut], mm[:cut]), 'bottom': _iou(pm[cut:], mm[cut:])})


# ------------------------------------------------------------------ листы
def _cell(im, caption, w, h, font):
    im = im.convert('RGB').resize((w, h - 30), Image.LANCZOS)
    c = Image.new('RGB', (w, h), 'white')
    c.paste(im, (0, 30))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, w, 29], fill='#222')
    d.text((8, 4), caption, fill='white', font=font)
    d.rectangle([0, 0, w - 1, h - 1], outline='#b0b0b0')
    return c


def sheet_view(view, out, model=ROOT + 'tayron_2023_optimized_v2.stl', tag='v2'):
    box = A.BBOX[view]
    mesh = A.load(model) if model == A.MODEL_MAIN else trimesh.load(model, force='mesh')
    photo = photo_crop(view, box)
    mrgb = render_rgb(mesh, view, box)
    tex = texture_from_photo(photo, mrgb)
    bl = blend(photo, mrgb)
    medges = model_edges(mesh, view, box, photo.size)
    pedges = photo_edges(photo)
    cont = contours_over_photo(photo, medges, pedges)
    cov = coverage(medges, pedges)
    score = cov['model_on_photo']

    W = 900
    hh = int(W * photo.height / photo.width)
    fnt = ImageFont.truetype(FONT_B, 19)
    cells = [('ФОТО', photo), (f'МОДЕЛЬ ({tag}), рендер', Image.fromarray(mrgb)),
             ('МОДЕЛЬ + ТЕКСТУРА С ФОТО', tex)]
    row1 = [_cell(im, cap, W, hh + 30, fnt) for cap, im in cells]
    sdiff, iou, iou_parts = silhouette_diff(photo, mrgb)
    cells2 = [(f'ФОТО {int(ALPHA * 100)} % ПОВЕРХ МОДЕЛИ', bl),
              ('КОНТУРЫ: фото — синие, рёбра модели — красные', cont),
              (f'СИЛУЭТЫ: IoU {iou * 100:.1f} % (верх {iou_parts["top"] * 100:.0f} %, низ {iou_parts["bottom"] * 100:.0f} %)', sdiff)]
    row2 = [_cell(im, cap, W, hh + 30, fnt) for cap, im in cells2]
    img = Image.new('RGB', (W * 3 + 8, 2 * hh + 60 + 48), '#e8e8e8')
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 44], fill='#0d0d0d')
    d.text((10, 8), f'ТЕКСТУРА С ФОТО · {VIEW_RU[view]}: модель {tag} в кадре реальной машины',
           fill='white', font=fnt)
    for i, c in enumerate(row1):
        img.paste(c, (i * (W + 4), 46))
    for i, c in enumerate(row2):
        img.paste(c, (i * (W + 4), 46 + hh + 30))
    img.save(out, optimize=True)
    print(out, img.size,
          '| рёбра модели на контурах фото:',
          '—' if cov['model_on_photo'] is None else f"{cov['model_on_photo'] * 100:.0f} %",
          '| контуры фото на рёбрах модели:',
          '—' if cov['photo_on_model'] is None else f"{cov['photo_on_model'] * 100:.0f} %",
          f'| IoU силуэтов {iou * 100:.1f} % (верх {iou_parts["top"] * 100:.0f} / низ {iou_parts["bottom"] * 100:.0f})')
    return cov


def _short(title):
    """Короткое имя зоны для подписи в узкой колонке."""
    return title.split(' (')[0].split(',')[0].strip()


def sheet_zones(view, out, model, src, tag='v2'):
    """Зоны крупно: фото | исходный скан | модель | фото 30 % поверх + цифры."""
    box = A.BBOX[view]
    photo_full = photo_crop(view, box)
    mesh = trimesh.load(model, force='mesh')
    mrgb = render_rgb(mesh, view, box)
    bl = blend(photo_full, mrgb)
    src_rgb = render_rgb(trimesh.load(src, force='mesh'), view, box) if src else None
    medges = model_edges(mesh, view, box, photo_full.size)
    pedges = photo_edges(photo_full)
    x0, y0 = int(box[0]), int(box[2])
    W = 540
    fnt = ImageFont.truetype(FONT_B, 15)
    rows = []
    for title, zb in A.ROWS[view]:
        zx0, zx1, zy0, zy1 = zb          # зоны в A.ROWS тоже (x0, x1, y0, y1)
        Wf, Hf = photo_full.size
        sy0 = max(0, int(zy0 - y0)); sy1 = min(Hf, int(zy1 - y0))
        sx0 = max(0, int(zx0 - x0)); sx1 = min(Wf, int(zx1 - x0))
        sl = (sy0, sy1, sx0, sx1)
        h = int(W * (sl[1] - sl[0]) / max(sl[3] - sl[2], 1)) + 30
        cov = coverage(medges, pedges, (slice(sl[0], sl[1]), slice(sl[2], sl[3])), r=4)
        st = _short(title)
        tiles = [_cell(photo_full.crop((sl[2], sl[0], sl[3], sl[1])), f'ФОТО · {st}', W, h, fnt)]
        if src_rgb is not None:
            tiles.append(_cell(Image.fromarray(src_rgb).crop((sl[2], sl[0], sl[3], sl[1])),
                               f'ИСХОДНЫЙ СКАН · {st}', W, h, fnt))
        cap_cov = ('—' if cov['model_on_photo'] is None else
                   f"{cov['model_on_photo'] * 100:.0f} % / {cov['photo_on_model'] * 100:.0f} %")
        tiles.append(_cell(Image.fromarray(mrgb).crop((sl[2], sl[0], sl[3], sl[1])),
                           f'МОДЕЛЬ {tag} · {st} · рёбра↔контуры {cap_cov}', W, h, fnt))
        tiles.append(_cell(bl.crop((sl[2], sl[0], sl[3], sl[1])),
                           f'ФОТО {int(ALPHA * 100)} % · {st}', W, h, fnt))
        rows.append(tiles)
        print(f'  {view}: {title}: рёбра→контуры '
              f'{"—" if cov["model_on_photo"] is None else round(cov["model_on_photo"] * 100)} %, '
              f'контуры→рёбра {"—" if cov["photo_on_model"] is None else round(cov["photo_on_model"] * 100)} %')
    ncol = len(rows[0])
    Wtot = ncol * (W + 4)
    Htot = 46 + sum(r[0].height for r in rows)
    img = Image.new('RGB', (Wtot, Htot), '#e8e8e8')
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, Wtot, 44], fill='#0d0d0d')
    d.text((10, 8), f'ЗОНЫ КРУПНО · {VIEW_RU[view]}: как геометрия ложится под фотографию',
           fill='white', font=ImageFont.truetype(FONT_B, 20))
    y = 46
    for tiles in rows:
        for i, t in enumerate(tiles):
            img.paste(t, (i * (W + 4), y))
        y += tiles[0].height
    img.save(out, optimize=True)
    print(out, img.size)



# --------------------------------------------------- наложение в одном кадре
# Окна зон в пикселях кадра (x0, x1, y0, y1) — те же, что в A.ROWS, но
# обрезанные по самим элементам, чтобы наложение читалось глазами.
OVERLAY_ZONES = {
    'front': [('ЛЕВАЯ ФАРА', (110, 470, 430, 640)),
              ('ПРАВАЯ ФАРА', (1010, 1370, 430, 640)),
              ('РЕШЁТКА, ЭМБЛЕМА', (560, 920, 520, 760)),
              ('ПТФ + НИЖНЯЯ РЕШЁТКА', (240, 700, 700, 1120))],
    'rear': [('ЛЕВЫЙ ФОНАРЬ', (20, 470, 290, 500)),
             ('ПРАВЫЙ ФОНАРЬ', (720, 1170, 290, 500)),
             ('СВЕТОВАЯ ПОЛОСА, ЭМБЛЕМА', (430, 770, 320, 500)),
             ('БАМПЕР', (40, 640, 670, 980))],
}
ZOOM_W = 560
RULER_STEP, RULER_LABEL = 25, 50


def _ruler(im, x0, y0, zoom, fnt):
    """Наложить сетку с подписями строк кадра — чтобы сдвиг читался в пикселях."""
    d = ImageDraw.Draw(im)
    h = im.height
    r = (y0 // RULER_STEP) * RULER_STEP
    while True:
        yy = int((r - y0) * zoom)
        if yy > h:
            break
        if yy >= 0:
            major = (r % RULER_LABEL == 0)
            d.line([0, yy, im.width, yy], fill=(235, 60, 60) if major else (150, 150, 255), width=1)
            if major:
                d.text((3, max(0, yy - 9)), str(r), fill=(200, 0, 0), font=fnt)
        r += RULER_STEP
    return im


def sheet_overlay(view, out, model=ROOT + 'tayron_2023_optimized_v2.stl',
                  src=None, tag='v2', alpha=0.40, zoom=2.6):
    """Одна картинка: фото поверх модели в одном кадре + зоны крупно с линейкой.

    Верх — ровно то, что просили: фотография с прозрачностью поверх рендера
    модели, один кадр. Низ — тот же кадр крупно по зонам, с сеткой в пикселях
    кадра; при желании со столбцом исходного скана (--src), чтобы видеть
    источник расхождения.
    """
    box = A.BBOX[view]
    mesh = trimesh.load(model, force='mesh')
    photo = photo_crop(view, box)
    mrgb = render_rgb(mesh, view, box)
    a = np.asarray(photo, float) / 255.0
    b = np.asarray(mrgb, float) / 255.0
    ov = Image.fromarray((((1 - alpha) * b + alpha * a).clip(0, 1) * 255).astype(np.uint8))
    src_rgb = render_rgb(trimesh.load(src, force='mesh'), view, box) if src else None

    fnt = ImageFont.truetype(FONT_B, 18)
    fnt_s = ImageFont.truetype(FONT_B, 14)
    cols = 3 if src_rgb is not None else 2
    W = cols * (ZOOM_W + 4)
    hh = int(W * photo.height / photo.width)
    top = _cell(ov, f'ФОТО {int(alpha * 100)} % ПОВЕРХ МОДЕЛИ {tag} · {VIEW_RU[view].lower()} · '
                     f'кадр = кроп фото, пиксель в пиксель', W, hh + 30, fnt)
    # рамки зон на большом кадре
    top_img = top.copy()
    d = ImageDraw.Draw(top_img)
    k = W / photo.width
    for title, (x0, x1, y0, y1) in OVERLAY_ZONES[view]:
        d.rectangle([int(x0 * k), 30 + int(y0 * k), int(x1 * k), 30 + int(y1 * k)],
                    outline=(235, 60, 60), width=2)
    tiles = []
    for title, (x0, x1, y0, y1) in OVERLAY_ZONES[view]:
        zw = ZOOM_W
        zh = int(zw * (y1 - y0) / (x1 - x0))
        def mk(arr):
            c = Image.fromarray(np.asarray(arr).astype(np.uint8)).crop((x0, y0, x1, y1))
            c = c.resize((zw, zh), Image.LANCZOS)
            return _ruler(c, x0, y0, zw / (x1 - x0), fnt_s)
        tiles.append([_cell(mk(photo), f'ФОТО · {title}', zw, zh + 30, fnt_s)])
        if src_rgb is not None:
            tiles[-1].append(_cell(mk(src_rgb), 'ИСХОДНЫЙ СКАН', zw, zh + 30, fnt_s))
        tiles[-1].append(_cell(mk(mrgb), f'МОДЕЛЬ {tag}', zw, zh + 30, fnt_s))
    Wtot = W
    Htot = 46 + top_img.height + sum(t[0].height for t in tiles)
    img = Image.new('RGB', (Wtot, Htot), '#e8e8e8')
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, Wtot, 44], fill='#0d0d0d')
    d.text((10, 8), f'НАЛОЖЕНИЕ В ОДНОМ КАДРЕ · {VIEW_RU[view]}: фото {int(alpha * 100)} % поверх модели {tag}',
           fill='white', font=ImageFont.truetype(FONT_B, 20))
    img.paste(top_img, ((Wtot - W) // 2, 46))
    y = 46 + top_img.height
    for col_tiles in tiles:
        for j, t in enumerate(col_tiles):
            img.paste(t, (j * (ZOOM_W + 4), y))
        y += col_tiles[0].height
    img.save(out, optimize=True)
    print(out, img.size)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['front', 'rear', 'both', 'zones', 'overlay'])
    ap.add_argument('--model', default=ROOT + 'tayron_2023_optimized_v2.stl')
    ap.add_argument('--src', default=None, help='исходный скан для колонки «до»')
    ap.add_argument('--tag', default='v2')
    a = ap.parse_args()
    views = ('front', 'rear') if a.cmd == 'both' else ((a.cmd,) if a.cmd in ('front', 'rear') else ('front', 'rear'))
    if a.cmd == 'overlay':
        for v in ('front', 'rear'):
            sheet_overlay(v, ROOT + f'overlay_{v}.png', a.model, a.src, a.tag)
    elif a.cmd == 'zones':
        for v in views:
            sheet_zones(v, ROOT + f'photo_tex_{v}_zones.png', a.model, a.src, a.tag)
    else:
        for v in views:
            sheet_view(v, ROOT + f'photo_tex_{v}.png', a.model, a.tag)
