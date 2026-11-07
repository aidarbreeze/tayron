#!/usr/bin/env python3
"""Compare: reference photo / earlier corrected model / model built from scratch.

All three columns are cropped to the visible car and fitted to equal cells.
This is a qualitative shape check, not a metrological overlay.
"""
import os

from PIL import Image, ImageDraw

from make_comparison import ROOT, bbox_nonwhite, crop, fit, label, load


def main():
    outdir = os.path.join(ROOT, 'renders')
    rows = [
        ('Боковой вид', '1793820501.png', None, 'side_left'),
        ('Передняя 3/4', 'gen1200 (2).jpg', (35, 400, 1185, 860), 'front34_left'),
        ('Задняя 3/4', 'gen1200 (5).jpg', (35, 400, 1185, 850), 'rear34_right'),
        ('Спереди', '1793820865.png', None, 'front'),
        ('Сзади', '1793820606.png', None, 'rear'),
    ]
    cell = (560, 360)
    W = 3 * cell[0] + 40
    H = 116 + len(rows) * (cell[1] + 46) + 76
    sheet = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    label(d, (20, 16), 'Tayron с нуля: фото / прежняя правка / новая процедурная модель', 27, True)
    label(d, (20, 56), 'Новая модель построена генератором по замерам с фото; исходные STL не читались ни разу.', 18)
    for j, t in enumerate(['Фото-ориентир', 'Ранее исправленная STL', 'Новая модель с нуля']):
        label(d, (20 + j * cell[0] + 10, 88), t, 21, True)
    for i, (name, ref, box, view) in enumerate(rows):
        y = 116 + i * (cell[1] + 46)
        label(d, (20, y + 6), name, 20, True)
        ref_im = load(ref)
        ref_im = crop(ref_im, box) if box else bbox_nonwhite(ref_im)
        prev = bbox_nonwhite(load(f'renders/final/{view}.png'), 250)
        new = bbox_nonwhite(load(f'renders/scratch/{view}.png'), 250)
        for j, im in enumerate([ref_im, prev, new]):
            sheet.paste(fit(im, cell), (20 + j * cell[0], y + 34))
        d.rectangle([20, y + 34, W - 20, y + 34 + cell[1]], outline=(220, 220, 220))
    label(d, (20, H - 56), 'Совпадают габариты и посадка; поверхности новой модели глаже и проще, детали переданы геометрией без цвета.', 18)
    sheet.save(os.path.join(outdir, 'scratch_comparison.png'))

    details = [
        ('Колесо', '1793820501.png', (1620, 980, 1980, 1340),
         'renders/final/side_left.png', (930, 530, 1170, 770),
         'renders/scratch/side_left.png', (940, 520, 1200, 780)),
        ('Решётка и фары', '1793820865.png', (520, 760, 1880, 1180),
         'renders/final/front.png', (300, 430, 1100, 720),
         'renders/scratch/front.png', (280, 490, 1120, 800)),
        ('Задняя полоса', '1793820606.png', (520, 900, 1830, 1150),
         'renders/final/rear.png', (280, 390, 1120, 560),
         'renders/scratch/rear.png', (250, 390, 1150, 570)),
        ('Рейлинги', '1793820501.png', (400, 520, 1450, 720),
         'renders/final/side_left.png', (280, 230, 840, 340),
         'renders/scratch/side_left.png', (280, 230, 900, 350)),
    ]
    cell2 = (560, 300)
    W2 = 3 * cell2[0] + 40
    H2 = 110 + len(details) * (cell2[1] + 46) + 40
    s2 = Image.new('RGB', (W2, H2), (255, 255, 255))
    d2 = ImageDraw.Draw(s2)
    label(d2, (20, 18), 'Детали: фото / прежняя правка / с нуля', 27, True)
    label(d2, (20, 58), 'Крупные планы для качественной оценки формы.', 18)
    for i, (name, ref, rbox, pp, pb, np_, nb) in enumerate(details):
        y = 110 + i * (cell2[1] + 46)
        label(d2, (20, y + 6), name, 20, True)
        ims = [crop(load(ref), rbox), crop(load(pp), pb), crop(load(np_), nb)]
        for j, im in enumerate(ims):
            s2.paste(fit(im, cell2), (20 + j * cell2[0], y + 34))
        d2.rectangle([20, y + 34, W2 - 20, y + 34 + cell2[1]], outline=(220, 220, 220))
    s2.save(os.path.join(outdir, 'scratch_details.png'))
    print('saved renders/scratch_comparison.png and renders/scratch_details.png')


if __name__ == '__main__':
    main()
