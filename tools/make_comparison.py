#!/usr/bin/env python3
"""Build honest visual check sheets from reference photos and STL renders.

The photos and the CPU clay renders are not pixel-aligned surveys: they are
cropped to the visible car and scaled to the same cell so shape differences
can be judged by eye.  No attempt is made to pass a render off as a photo.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def load(path):
    return Image.open(os.path.join(ROOT, path)).convert('RGB')


def bbox_nonwhite(im, thresh=244):
    a = np.asarray(im)
    mask = np.min(a, axis=2) < thresh
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return im
    pad = 8
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, im.width)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, im.height)
    return im.crop((x0, y0, x1, y1))


def crop(im, box):
    return im.crop(box)


def fit(im, cell, bg=(255, 255, 255)):
    im = im.copy()
    im.thumbnail((cell[0] - 20, cell[1] - 20), Image.LANCZOS)
    canvas = Image.new('RGB', cell, bg)
    canvas.paste(im, ((cell[0] - im.width) // 2, (cell[1] - im.height) // 2))
    return canvas


def label(draw, xy, text, size=22, bold=False):
    draw.text(xy, text, fill=(20, 20, 20),
              font=ImageFont.truetype(FONT_B if bold else FONT, size))


def main():
    outdir = os.path.join(ROOT, 'renders')
    os.makedirs(outdir, exist_ok=True)
    rows = [
        ('Боковой вид', '1793820501.png', None, 'side_left'),
        ('Передняя 3/4 проекция', 'gen1200 (2).jpg', (35, 400, 1185, 860), 'front34_left'),
        ('Задняя 3/4 проекция', 'gen1200 (5).jpg', (35, 400, 1185, 850), 'rear34_right'),
        ('Вид спереди', '1793820865.png', None, 'front'),
        ('Вид сзади', '1793820606.png', None, 'rear'),
    ]
    cell = (560, 360)
    W = 3 * cell[0] + 40
    H = 110 + len(rows) * (cell[1] + 46) + 70
    sheet = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    label(d, (20, 18), 'Tayron: визуальная сверка исправленной STL-модели с фотографиями', 28, True)
    label(d, (20, 58), 'Фотографии обрезаны по автомобилю; рендеры — ортографическая глина. Это качественная, не метрологическая сверка.', 18)
    for j, title in enumerate(['Фото-ориентир', 'Исходная модель', 'Исправленная модель']):
        label(d, (20 + j * cell[0] + 10, 86), title, 22, True)
    for i, (name, ref, box, view) in enumerate(rows):
        y = 110 + i * (cell[1] + 46)
        label(d, (20, y + 6), name, 20, True)
        ref_im = load(ref)
        ref_im = crop(ref_im, box) if box else bbox_nonwhite(ref_im)
        before = bbox_nonwhite(load(f'renders/before/{view}.png'), 250)
        after = bbox_nonwhite(load(f'renders/final/{view}.png'), 250)
        for j, im in enumerate([ref_im, before, after]):
            sheet.paste(fit(im, cell), (20 + j * cell[0], y + 34))
        d.rectangle([20, y + 34, W - 20, y + 34 + cell[1]], outline=(220, 220, 220))
    label(d, (20, H - 52), 'Итог: сохранены габариты; переработаны диски и рейлинги, добавлена задняя световая полоса. Цвет/материалы не моделировались.', 18)
    sheet.save(os.path.join(outdir, 'final_comparison.png'))

    # Detail checks: wheel, roof rail, rear light bar.
    details = [
        ('Колесо: пять пар спиц', '1793820501.png', (1620, 980, 1980, 1340),
         'renders/before/side_left.png', (930, 530, 1170, 770),
         'renders/final/side_left.png', (930, 530, 1170, 770)),
        ('Рейлинги: тонкая дуга на крыше', '1793820501.png', (400, 520, 1450, 720),
         'renders/before/side_left.png', (280, 230, 840, 340),
         'renders/final/side_left.png', (280, 230, 840, 340)),
        ('Задняя световая полоса', '1793820606.png', (520, 900, 1830, 1150),
         'renders/before/rear.png', (280, 390, 1120, 560),
         'renders/final/rear.png', (280, 390, 1120, 560)),
    ]
    cell2 = (560, 300)
    W2 = 3 * cell2[0] + 40
    H2 = 110 + len(details) * (cell2[1] + 46) + 40
    sheet2 = Image.new('RGB', (W2, H2), (255, 255, 255))
    d2 = ImageDraw.Draw(sheet2)
    label(d2, (20, 18), 'Контроль деталей по фотографиям', 28, True)
    label(d2, (20, 58), 'Крупные планы: фото-ориентир / исходная модель / исправленная модель.', 18)
    for i, (name, ref, rbox, bp, bb, fp, fb) in enumerate(details):
        y = 110 + i * (cell2[1] + 46)
        label(d2, (20, y + 6), name, 20, True)
        ims = [crop(load(ref), rbox), crop(load(bp), bb), crop(load(fp), fb)]
        for j, im in enumerate(ims):
            sheet2.paste(fit(im, cell2), (20 + j * cell2[0], y + 34))
        d2.rectangle([20, y + 34, W2 - 20, y + 34 + cell2[1]], outline=(220, 220, 220))
    sheet2.save(os.path.join(outdir, 'detail_checks.png'))
    print('saved renders/final_comparison.png and renders/detail_checks.png')


if __name__ == '__main__':
    main()
