"""Раскладка всех деталей на стол принтера.

Читает печатную сборку (по умолчанию Tayron5_flex.3mf), ориентирует каждую
деталь «лицом вниз», раскладывает полками на столе 220 x 220 мм с зазором
4 мм и пишет:

    Tayron6_plate.3mf              — 13 объектов, все на столе, z >= 0;
    suspension/assets/plate.json   — те же позы для анимации на сайте.

    python3 layout.py                          # стол 220x220, зазор 4 мм
    python3 layout.py --bed 180 --gap 3        # стол 180x180
    python3 layout.py --check                  # только проверить раскладку

Ориентация (её же показывает сайт по кнопке «На стол»):
    кузов       — как есть, дном на стол;
    колесо      — лежа, осью вверх (поворот 90° вокруг X, вокруг центра колеса);
    втулка      — фланцем на стол (тот же поворот);
    переходник  — как есть: колодкой на стол, лента лежит плоско.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

import m3mf

HERE = os.path.dirname(os.path.abspath(__file__))
U = 5.8824                      # единиц модели в 1 мм печати (печать 150 мм)
MM = U                          # 1 мм печати в единицах модели


# ------------------------------------------------------------------ ориентация
def orient(name, rig):
    """→ (R 3x3, точка вращения O). Поворот задан в координатах модели."""
    kind = ('wheel' if name.startswith('Колесо') else
            'bush' if name.startswith('Втулка') else
            'link' if name.startswith('S-') else 'body')
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    RxP = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])      # +90°: ось колеса Y → Z
    RxM = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])      # −90°: фланцем вниз
    if kind in ('wheel', 'bush'):
        i = int(''.join(ch for ch in name if ch.isdigit()) or 0) - 1
        # колесо — на бок (все четыре одинаково), втулка — фланцем на стол
        return (RxP if kind == 'wheel' else RxM), np.array(rig[i]['wc'], float) / U
    return np.eye(3), np.zeros(3)


# ------------------------------------------------------------------- упаковка
def layout(parts, bed=(220, 220), gap=4.0, margin=5.0):
    """parts: [(имя, mesh, R, O)] → [(имя, mesh_на_столе, поза_группы, R, габарит)]"""
    W, H = bed
    items = []
    for name, m, R, O in parts:
        V = (np.asarray(m.vertices, float) - O) @ R.T
        ext = V.max(0) - V.min(0)
        items.append(dict(name=name, m=m, R=R, O=O, V=V,
                          size=(float(ext[0]), float(ext[1])), h=float(ext[2]),
                          area=float(ext[0] * ext[1])))
    items.sort(key=lambda it: -it['h'])                    # высокое — вперёд
    x = y = margin
    row_h = 0.0
    out = []
    for it in items:
        w, h = it['size']
        if x + w > W - margin + 1e-9 and x > margin + 1e-9:
            x, y, row_h = margin, y + row_h + gap, 0.0
        if y + h > H - margin + 1e-9:
            raise SystemExit('не влезает на стол %gx%g мм: %s' % (W, H, it['name']))
        t = np.array([x - it['V'][:, 0].min(), y - it['V'][:, 1].min(),
                      -it['V'][:, 2].min()])
        # поза группы для сайта: группа стоит в t (V' уже относительно точки вращения)
        out.append((it['name'], it['V'] + t, t, it['R'], (w, h, it['h']),
                    (x, y), it['m']))
        x += w + gap
        row_h = max(row_h, h)
    return out


def check(placed, bed=(220, 220)):
    """Проверка: всё в пределах стола, низ на нуле, детали не пересекаются."""
    problems = []
    boxes = []
    for name, V, pos, R, ext, (x, y), _m in placed:
        lo, hi = V.min(0), V.max(0)
        if lo[2] < -1e-6 or abs(lo[2]) > 1e-6:
            problems.append('%s: низ на z=%.3f' % (name, lo[2]))
        if lo[0] < -1e-6 or lo[1] < -1e-6 or hi[0] > bed[0] + 1e-6 or hi[1] > bed[1] + 1e-6:
            problems.append('%s: вылезает на стол %gx%g' % (name, bed[0], bed[1]))
        boxes.append((name, lo, hi, ext))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            n1, lo1, hi1, e1 = boxes[i]
            n2, lo2, hi2, e2 = boxes[j]
            if np.all(lo1 < hi2) and np.all(lo2 < hi1):
                problems.append('пересекаются: %s и %s' % (n1, n2))
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description='раскладка деталей на стол для печати')
    ap.add_argument('--src', default=os.path.join(HERE, 'Tayron5_flex.3mf'))
    ap.add_argument('--out', default=os.path.join(HERE, 'Tayron6_plate.3mf'))
    ap.add_argument('--bed', default='220x220', help='размер стола, мм (например 180x180)')
    ap.add_argument('--gap', type=float, default=4.0, help='зазор между деталями, мм')
    ap.add_argument('--check', action='store_true', help='только проверить')
    a = ap.parse_args(argv)

    bed = tuple(float(v) for v in a.bed.lower().split('x'))[:2]
    rig = json.load(open(os.path.join(HERE, 'suspension', 'assets', 'rig.json')))

    parts = []
    for name, m in m3mf.read(a.src):
        m = m.copy()
        m.apply_scale(1.0 / U)                 # единицы модели → мм печати
        R, O = orient(name, rig)
        parts.append((name, m, R, O))

    placed = layout(parts, bed=bed, gap=a.gap)
    probs = check(placed, bed=bed)
    for p in probs:
        print('  ! ' + p)
    print('стол %gx%g мм, зазор %g мм, деталей %d' % (bed[0], bed[1], a.gap, len(placed)))
    used = max(V[:, 0].max() for _, V, *_ in placed)
    usedH = max(V[:, 1].max() for _, V, *_ in placed)
    print('занято %.1f x %.1f мм (не выше %.1f мм)' % (
        used, usedH, max(e[2] for *_, e, _xy, _m in placed)))
    for name, V, pos, R, ext, (x, y), _m in placed:
        print('  %-20s %5.1f x %5.1f x %5.1f мм  в (%5.1f, %5.1f)' % (
            name, ext[0], ext[1], ext[2], x, y))
    if probs:
        return 1
    if a.check:
        return 0

    # --- 3MF: все детали на столе, мировые координаты
    out_parts = [(name, trimesh.Trimesh(vertices=V * U, faces=m.faces, process=False))
                 for name, V, pos, R, ext, xy, m in placed]
    m3mf.write(a.out, out_parts,
               meta='VW Tayron v14: раскладка на стол %gx%g мм (%d деталей)' %
                    (bed[0], bed[1], len(out_parts)))
    print('записано %s (%.2f МБ)' % (os.path.basename(a.out),
                                    os.path.getsize(a.out) / 1e6))

    # --- позы для сайта
    def pose(V, pos, R):
        return dict(rot=[float(np.degrees(np.arctan2(R[2, 1], R[1, 1]))), 0.0, 0.0],
                    pos=[round(float(v) * U, 2) for v in pos])   # обратно в единицы модели

    bx0 = min(V[:, 0].min() for _, V, *_ in placed)
    by0 = min(V[:, 1].min() for _, V, *_ in placed)
    bx1 = max(V[:, 0].max() for _, V, *_ in placed)
    by1 = max(V[:, 1].max() for _, V, *_ in placed)
    site = dict(bed=[bed[0] * MM, bed[1] * MM], gap=a.gap, U=U,
                bbox=[round(b * MM, 1) for b in (bx0, by0, bx1, by1)],
                body=None, link=[None] * 4, wheel=[None] * 4, bush=[None] * 4)
    for name, V, pos, R, ext, xy, _m in placed:
        p = pose(V, pos, R)
        if name.startswith('S-'):
            site['link'][int([c for c in name if c.isdigit()][0]) - 1] = p
        elif name.startswith('Колесо'):
            site['wheel'][int([c for c in name if c.isdigit()][0]) - 1] = p
        elif name.startswith('Втулка'):
            site['bush'][int([c for c in name if c.isdigit()][0]) - 1] = p
        else:
            site['body'] = p
    dst = os.path.join(HERE, 'suspension', 'assets', 'plate.json')
    json.dump(site, open(dst, 'w'), indent=1, ensure_ascii=False)
    print('записано %s' % os.path.relpath(dst, HERE))
    return 0


if __name__ == '__main__':
    sys.exit(main())
