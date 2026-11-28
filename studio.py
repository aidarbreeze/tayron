#!/usr/bin/env python3
"""Tayron Studio — локальный сервер для работы с моделями: загрузил → правишь → сохранил.

Что умеет (всё на стандартной библиотеке, без pip):

* открывает тот же редактор кузова (`suspension/studio.html` → `sculpt.js`), но
  модель берётся **из вашего файла**, а не из `assets/body.glb`;
* принимает **GLB, STL (binary и ascii), 3MF** — формат определяется по сигнатуре;
* держит рабочий каталог с проектами: список, переименование, корзина, размеры,
  дата, число версий, габарит и объём каждой модели;
* пишет **версии**: каждое «Сохранить» в редакторе ложится файлом в
  `projects/<id>/versions/`, старые не затираются;
* автосохранение правки (json состояния) — можно закрыть вкладку и вернуться;
* считает по сетке: вершин, граней, габарит в мм, объём, открытые рёбра,
  вырожденные грани — видно сразу после загрузки;
* умеет отдать готовый файл обратно (3MF/STL) и удалить проект в корзину.

Запуск:
    python3 studio.py                 # http://127.0.0.1:8131
    python3 studio.py --port 9000 --dir ~/модели --open
    python3 studio.py --host 0.0.0.0  # для доступа с других машин / превью
"""
from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import html
import io
import json
import os
import re
import shutil
import socket
import struct
import sys
import threading
import time
import urllib.parse
import webbrowser
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

APP = 'tayron-studio'
VERSION = '1.0'
ROOT = os.path.dirname(os.path.abspath(__file__))          # каталог репозитория
MAX_UPLOAD = 200 * 1024 * 1024                              # 200 МБ на файл
MAX_JSON = 64 * 1024 * 1024                                 # состояние редактора

# ---------------------------------------------------------------------------
# 1. модели: разбор файлов
# ---------------------------------------------------------------------------


class ModelError(Exception):
    """Файл не похож на модель — сообщение уходит пользователю как 400."""


def _weld(pos, idx):
    """Склеить совпадающие вершины (STL и часть GLB приходят разрезанными).

    Без склейки кузов из STL даёт 300 000 вершин вместо 52 000 и вода-тайт
    проверка невозможна: рёбра не сойдутся никогда.
    """
    key = {}
    new_pos = []
    remap = [0] * len(pos)
    for i, p in enumerate(pos):
        k = (round(p[0], 4), round(p[1], 4), round(p[2], 4))
        j = key.get(k)
        if j is None:
            j = len(new_pos)
            key[k] = j
            new_pos.append([p[0], p[1], p[2]])
        remap[i] = j
    new_idx = [remap[a] for a in idx]
    return new_pos, new_idx


def read_glb(data):
    """GLB (binary glTF) → позиции и индексы первой сетки."""
    if len(data) < 20 or data[:4] != b'glTF':
        raise ModelError('это не GLB: нет подписи glTF')
    ver, _length = struct.unpack_from('<II', data, 4)
    if ver != 2:
        raise ModelError(f'GLB версии {ver} не поддерживается, нужен 2')
    off, js, bin_ = 12, None, b''
    while off + 8 <= len(data):
        clen, ctype = struct.unpack_from('<II', data, off)
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode('utf-8'))
        elif ctype == 0x004E4942:
            bin_ = chunk
        off += 8 + clen + (-clen % 4)
    if js is None:
        raise ModelError('в GLB нет JSON-чанка')
    comp = {5126: ('f', 4), 5125: ('I', 4), 5123: ('H', 2), 5121: ('B', 1)}
    ncomp = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}

    def acc(i):
        a = js['accessors'][i]
        bv = js['bufferViews'][a['bufferView']]
        fmt, size = comp[a['componentType']]
        n = ncomp[a['type']]
        stride = bv.get('byteStride') or size * n
        base = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        out = [struct.unpack_from('<' + fmt * n, bin_, base + k * stride) for k in range(a['count'])]
        return out

    best = None
    for m in js.get('meshes', []):
        for p in m.get('primitives', []):
            if 'POSITION' not in p.get('attributes', {}):
                continue
            pos = [list(v) for v in acc(p['attributes']['POSITION'])]
            if 'indices' in p:
                idx = [t[0] for t in acc(p['indices'])]
            else:
                idx = list(range(len(pos)))
            if best is None or len(idx) > best[1]:
                best = (pos, idx)
    if not best:
        raise ModelError('в GLB нет сеток с вершинами')
    return best


def read_stl(data):
    """STL: бинарный (84 + 50·n байт) или ascii."""
    head = data[:512].lstrip()
    binary_ok = False
    if len(data) >= 84:
        n = struct.unpack_from('<I', data, 80)[0]
        binary_ok = 84 + 50 * n == len(data) and n > 0
    if not binary_ok and head[:5].lower() == b'solid' and b'facet' in data[:4096].lower():
        return _read_stl_ascii(data)
    if not binary_ok:
        raise ModelError('это не STL: ни бинарный, ни ascii')
    n = struct.unpack_from('<I', data, 80)[0]
    if n <= 0 or n > 20_000_000:
        raise ModelError(f'STL заявляет {n} граней — не верю')
    pos, idx, key = [], [], {}
    for t in range(n):
        base = 84 + 50 * t
        vals = struct.unpack_from('<12f', data, base)
        for k in range(3):
            v = (vals[3 + k * 3], vals[4 + k * 3], vals[5 + k * 3])
            j = key.get(v)
            if j is None:
                j = len(pos)
                key[v] = j
                pos.append(list(v))
            idx.append(j)
    return pos, idx


def _read_stl_ascii(data):
    text = data.decode('utf-8', 'ignore')
    pos, idx, key = [], [], {}

    def node(v):
        j = key.get(v)
        if j is None:
            j = len(pos)
            key[v] = j
            pos.append(list(v))
        return j

    for m in re.finditer(r'vertex\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)\s+(-?[\d.eE+]+)', text):
        idx.append(node((float(m.group(1)), float(m.group(2)), float(m.group(3)))))
    if not idx or len(idx) % 3:
        raise ModelError('в ascii-STL не нашлось треугольников')
    return pos, idx


def read_3mf(data):
    """3MF — это zip с 3D/3dmodel.model: вершины и треугольники в XML."""
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ModelError('это не 3MF: файл не zip')
    names = [n for n in z.namelist() if n.endswith('.model')]
    if not names:
        raise ModelError('в 3MF нет .model-файла')
    xml = z.read(names[0]).decode('utf-8', 'ignore')
    pos, idx = [], []
    for m in re.finditer(r'<vertex\s+x="([-\d.eE+]+)"\s+y="([-\d.eE+]+)"\s+z="([-\d.eE+]+)"', xml):
        pos.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
    for m in re.finditer(r'<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"', xml):
        idx.extend((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if not pos or not idx:
        raise ModelError('в 3MF нет ни вершин, ни треугольников')
    return pos, idx


def looks_like_stl_binary(data):
    """Бинарный STL не имеет подписи: 84 байта заголовка + 50 байт на грань."""
    if len(data) < 84:
        return False
    n = struct.unpack_from('<I', data, 80)[0]
    return 0 < n < 20_000_000 and 84 + 50 * n == len(data)


def guess_kind(name, data):
    """Формат по сигнатуре, а не только по расширению."""
    if data[:4] == b'glTF':
        return 'glb'
    if data[:2] == b'PK':
        return '3mf'
    if data[:5].lower() == b'solid':
        return 'stl'
    if looks_like_stl_binary(data):
        return 'stl'
    ext = os.path.splitext(name or '')[1].lower().lstrip('.')
    if ext in ('glb', 'gltf'):
        return 'glb'
    if ext in ('3mf',):
        return '3mf'
    if ext in ('stl',):
        return 'stl'
    return None


def read_model(name, data):
    kind = guess_kind(name, data)
    if kind is None:
        raise ModelError('формат не распознан: нужен GLB, STL или 3MF')
    if kind == 'glb':
        pos, idx = read_glb(data)
    elif kind == 'stl':
        pos, idx = read_stl(data)
    else:
        pos, idx = read_3mf(data)
    if not pos or not idx:
        raise ModelError('в файле нет треугольников')
    if len(idx) % 3:
        raise ModelError('число индексов не кратно трём — битая сетка')
    pos, idx = _weld(pos, idx)
    if len(idx) < 3:
        raise ModelError('после склейки вершин не осталось граней')
    return kind, pos, idx


# ---------------------------------------------------------------------------
# 2. модели: замеры и проверки
# ---------------------------------------------------------------------------


def mesh_stats(pos, idx):
    """Вершины, грани, габарит, объём и целостность — то, что видно в карточке."""
    n = len(pos)
    if n == 0 or not idx:
        return {'verts': 0, 'faces': 0, 'volume_mm3': 0.0, 'open_edges': 0, 'degenerate': 0}
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    zs = [p[2] for p in pos]
    vol = 0.0
    deg = 0
    edges = {}
    for t in range(0, len(idx) - 2, 3):
        a, b, c = idx[t], idx[t + 1], idx[t + 2]
        pa, pb, pc = pos[a], pos[b], pos[c]
        vol += (pa[0] * (pb[1] * pc[2] - pb[2] * pc[1])
                - pa[1] * (pb[0] * pc[2] - pb[2] * pc[0])
                + pa[2] * (pb[0] * pc[1] - pb[1] * pc[0])) / 6.0
        if a == b or b == c or a == c:
            deg += 1
        for p, q in ((a, b), (b, c), (c, a)):
            k = (p, q) if p < q else (q, p)
            edges[k] = edges.get(k, 0) + 1
    open_edges = sum(1 for v in edges.values() if v == 1)
    bad_edges = sum(1 for v in edges.values() if v > 2)
    return {
        'verts': n,
        'faces': len(idx) // 3,
        'bbox_mm': None,        # заполняется выше по масштабу
        'size': [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
        'min': [min(xs), min(ys), min(zs)],
        'max': [max(xs), max(ys), max(zs)],
        'volume': abs(vol),
        'open_edges': open_edges,
        'bad_edges': bad_edges,
        'degenerate': deg,
    }


def write_stl(pos, idx, name='Tayron'):
    """Бинарный STL (в нём легче всего открыть модель обратно)."""
    out = bytearray(80)
    out[:len(name[:79])] = name[:79].encode('ascii', 'ignore')
    out.extend(struct.pack('<I', len(idx) // 3))
    for t in range(0, len(idx) - 2, 3):
        a, b, c = pos[idx[t]], pos[idx[t + 1]], pos[idx[t + 2]]
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        nx, ny, nz = u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]
        ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
        out.extend(struct.pack('<12fH', nx / ln, ny / ln, nz / ln,
                               a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], 0))
    return bytes(out)


def write_3mf(pos, idx, name='Кузов'):
    """3MF (zip + 3dmodel.model) — то, что ест слайсер."""
    verts = ''.join(
        f'<vertex x="{p[0]:.5f}" y="{p[1]:.5f}" z="{p[2]:.5f}"/>' for p in pos)
    tris = ''.join(
        f'<triangle v1="{idx[t]}" v2="{idx[t + 1]}" v3="{idx[t + 2]}"/>'
        for t in range(0, len(idx) - 2, 3))
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<model unit="millimeter" xml:lang="ru-RU" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources><object id="1" type="model" name="{html.escape(name)}">'
             f'<mesh><vertices>{verts}</vertices><triangles>{tris}</triangles></mesh>'
             '</object></resources>'
             '<build><item objectid="1"/></build></model>')
    types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
             '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr('[Content_Types].xml', types)
        z.writestr('_rels/.rels', rels)
        z.writestr('3D/3dmodel.model', model)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 3. хранилище проектов
# ---------------------------------------------------------------------------

SAFE = re.compile(r'[^\w\-. ]+', re.UNICODE)


def safe_name(name, fallback='model'):
    """Имя без путей и мусора: ../../etc/passwd → passwd."""
    base = os.path.basename((name or '').replace('\\', '/')).strip()
    base = SAFE.sub('_', base)
    base = base.strip('. ') or fallback
    return base[:120]


def slug(name):
    s = SAFE.sub('-', os.path.splitext(name or '')[0]).strip('-. ').lower()
    s = re.sub(r'-+', '-', s)
    return (s or 'model')[:40]


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def human_size(n):
    if n < 1024:
        return f'{n} Б'
    if n < 1024 * 1024:
        return f'{n / 1024:.0f} КБ'
    return f'{n / 1048576:.1f} МБ'


def human_when(iso):
    try:
        d = datetime.fromisoformat(iso)
    except Exception:
        return iso or '—'
    return d.strftime('%d.%m.%Y %H:%M')


class Store:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.projects = os.path.join(self.root, 'projects')
        self.trash = os.path.join(self.root, 'trash')
        os.makedirs(self.projects, exist_ok=True)
        os.makedirs(self.trash, exist_ok=True)
        self.lock = threading.Lock()

    # --- пути -----------------------------------------------------------
    def dir_of(self, pid):
        if not re.fullmatch(r'[0-9a-f]{8,20}', pid or ''):
            return None
        d = os.path.join(self.projects, pid)
        return d if os.path.isdir(d) else None

    def meta_path(self, pid):
        return os.path.join(self.projects, pid, 'meta.json')

    # --- чтение ---------------------------------------------------------
    def meta(self, pid):
        d = self.dir_of(pid)
        if not d:
            return None
        try:
            with open(os.path.join(d, 'meta.json'), encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def body(self, pid):
        d = self.dir_of(pid)
        if not d:
            return None
        p = os.path.join(d, 'body.json')
        if not os.path.exists(p):
            return None
        with open(p, encoding='utf-8') as f:
            return json.load(f)

    def state(self, pid):
        d = self.dir_of(pid)
        p = os.path.join(d or '', 'state.json') if d else None
        if not p or not os.path.exists(p):
            return None
        with open(p, encoding='utf-8') as f:
            return f.read()

    def list(self, query=''):
        out = []
        q = (query or '').lower().strip()
        for pid in os.listdir(self.projects):
            m = self.meta(pid)
            if not m:
                continue
            if q and q not in (m.get('name') or '').lower() and q not in (m.get('source') or '').lower():
                continue
            out.append({**m, 'id': pid})
        out.sort(key=lambda m: m.get('updated') or '', reverse=True)
        return out

    # --- запись ---------------------------------------------------------
    def create(self, name, source_name, kind, pos, idx, scale_mm=None):
        pid = binascii.hexlify(os.urandom(6)).decode()
        d = os.path.join(self.projects, pid)
        os.makedirs(os.path.join(d, 'versions'), exist_ok=True)
        os.makedirs(os.path.join(d, 'source'), exist_ok=True)
        st = mesh_stats(pos, idx)
        size = st['size']
        # масштаб печати: сколько миллиметров должно быть в самой длинной оси
        longest = max(size) or 1.0
        if not scale_mm or scale_mm <= 0:
            scale_mm = 150.0 if longest > 200 else round(longest, 2)
        u = longest / float(scale_mm)      # единиц модели на 1 мм печати
        body = {'pos': pos, 'idx': idx, 'U': u, 'units': 'model'}
        with open(os.path.join(d, 'body.json'), 'w', encoding='utf-8') as f:
            json.dump(body, f, separators=(',', ':'))
        meta = {
            'app': APP, 'v': VERSION, 'id': pid,
            'name': name or os.path.splitext(source_name)[0],
            'source': source_name, 'kind': kind,
            'created': now_iso(), 'updated': now_iso(),
            'scale_mm': round(float(scale_mm), 2), 'U': round(u, 6),
            'stats': {
                'verts': st['verts'], 'faces': st['faces'],
                'volume_mm3': round(st['volume'] / (u ** 3), 2) if u else 0,
                'bbox_mm': [round(v / u, 2) for v in size],
                'open_edges': st['open_edges'], 'bad_edges': st['bad_edges'],
                'degenerate': st['degenerate'],
            },
            'versions': [],
            'state_saved': None,
        }
        self._write_meta(pid, meta)
        return pid, meta

    def _write_meta(self, pid, meta):
        tmp = self.meta_path(pid) + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.meta_path(pid))

    def save_version(self, pid, kind, data, note=''):
        m = self.meta(pid)
        if not m:
            return None
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        ext = '3mf' if kind == '3mf' else 'stl'
        fname = f'{stamp}_{slug(m["name"])}.{ext}'
        vdir = os.path.join(self.projects, pid, 'versions')
        with open(os.path.join(vdir, fname), 'wb') as f:
            f.write(data)
        ver = {'file': fname, 'kind': ext, 'size': len(data),
               'when': now_iso(), 'note': note[:200]}
        m['versions'].append(ver)
        m['versions'] = m['versions'][-50:]
        m['updated'] = now_iso()
        self._write_meta(pid, m)
        return ver

    def save_state(self, pid, text):
        d = self.dir_of(pid)
        if not d:
            return False
        with open(os.path.join(d, 'state.json'), 'w', encoding='utf-8') as f:
            f.write(text)
        m = self.meta(pid) or {}
        m['state_saved'] = now_iso()
        m['updated'] = now_iso()
        self._write_meta(pid, m)
        return True

    def rename(self, pid, name):
        m = self.meta(pid)
        if not m:
            return None
        m['name'] = (name or m['name'])[:120]
        m['updated'] = now_iso()
        self._write_meta(pid, m)
        return m

    def delete(self, pid):
        d = self.dir_of(pid)
        if not d:
            return False
        with self.lock:
            dst = os.path.join(self.trash, f'{pid}-{int(time.time())}')
            shutil.move(d, dst)
        return True

    def version_path(self, pid, fname):
        d = self.dir_of(pid)
        if not d:
            return None
        p = os.path.abspath(os.path.join(d, 'versions', fname))
        if not p.startswith(os.path.abspath(os.path.join(d, 'versions')) + os.sep):
            return None
        return p if os.path.exists(p) else None


# ---------------------------------------------------------------------------
# 4. страница-лаунчер
# ---------------------------------------------------------------------------

LAUNCH_HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tayron Studio · модели</title>
<style>
:root{--bg:#0b0e13;--panel:#111720;--line:rgba(255,255,255,.10);--txt:#e7ecf3;
  --dim:#8b97a8;--acc:#38e1c4;--warn:#ffc94d;--bad:#f2645c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif}
a{color:var(--acc);text-decoration:none}
a:hover{text-decoration:underline}
header{padding:22px 26px 10px;display:flex;flex-wrap:wrap;gap:14px;align-items:baseline}
h1{margin:0;font-size:19px;font-weight:600;letter-spacing:.2px}
h1 span{color:var(--dim);font-weight:400}
.badge{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:20px;padding:2px 9px}
.wrap{padding:0 26px 40px;max-width:1240px}
.drop{margin:12px 0 18px;border:1.5px dashed rgba(56,225,196,.45);border-radius:14px;
  padding:26px;text-align:center;background:rgba(56,225,196,.04);transition:.15s}
.drop.hot{background:rgba(56,225,196,.12);border-color:var(--acc)}
.drop b{color:var(--acc)}
.drop p{margin:6px 0;color:var(--dim);font-size:12.5px}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0}
input[type=text],input[type=number]{background:#0e141c;border:1px solid var(--line);color:var(--txt);
  border-radius:9px;padding:8px 11px;font:inherit;min-width:180px}
input[type=number]{min-width:110px}
button,.btn{background:rgba(255,255,255,.06);border:1px solid var(--line);color:var(--txt);
  border-radius:9px;padding:8px 13px;font:inherit;cursor:pointer}
button:hover,.btn:hover{background:rgba(56,225,196,.14);border-color:rgba(56,225,196,.5)}
button.acc{background:rgba(56,225,196,.18);border-color:var(--acc);color:#b9fff2}
button.mini{padding:5px 9px;font-size:12px}
button:disabled{opacity:.45;cursor:default}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:14px 15px}
.card h3{margin:0 0 4px;font-size:15px;font-weight:600;word-break:break-word}
.card .sub{color:var(--dim);font-size:11.5px;margin-bottom:9px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px;color:var(--dim);
  font-variant-numeric:tabular-nums;margin-bottom:10px}
.kv b{color:var(--txt);font-weight:500}
.acts{display:flex;flex-wrap:wrap;gap:6px}
.ok{color:var(--acc)}.warn{color:var(--warn)}.bad{color:var(--bad)}
#log{margin-top:14px;font:12px/1.5 ui-monospace,Menlo,monospace;color:var(--dim);white-space:pre-wrap}
.hint{color:var(--dim);font-size:12px}
.empty{color:var(--dim);padding:26px;border:1px dashed var(--line);border-radius:13px;text-align:center}
</style></head><body>
<header>
  <h1>Tayron Studio <span>· модели, правка, версии</span></h1>
  <span class="badge" id="stat"></span>
  <span class="badge">каталог: __DIR__</span>
</header>
<div class="wrap">
  <div class="drop" id="drop">
    <b>Перетащите файл сюда</b> или <button class="acc" id="pick">выбрать файл</button>
    <p>GLB, STL (двоичный и ascii), 3MF · можно выбрать несколько сразу</p>
    <div class="row" style="justify-content:center">
      <label class="hint">масштаб печати, мм по длинной стороне:</label>
      <input type="number" id="scale" min="1" max="2000" step="1" placeholder="авто">
    </div>
    <input type="file" id="file" multiple accept=".glb,.gltf,.stl,.3mf" style="display:none">
  </div>
  <div class="row">
    <input type="text" id="q" placeholder="поиск по названию…">
    <button id="refresh">Обновить</button>
    <span class="hint" id="hint">Правка открывается в том же редакторе, что и раньше; модель берётся из вашего файла.</span>
  </div>
  <div class="cards" id="cards"></div>
  <div id="log"></div>
</div>
<script>
const PROJECTS = __PROJECTS__;
const fmt = (n) => n >= 1024*1024 ? (n/1048576).toFixed(1)+' МБ' : n >= 1024 ? Math.round(n/1024)+' КБ' : n+' Б';
function card(p) {
  const st = p.stats || {};
  const bbox = (st.bbox_mm || []).map(v => v.toFixed(1)).join(' × ');
  const bad = st.open_edges > 0;
  const vol = st.volume_mm3 ? (st.volume_mm3/1000).toFixed(2)+' см³' : '—';
  const vcount = (p.versions || []).length;
  return `<div class="card">
    <h3>${esc(p.name)}</h3>
    <div class="sub">${esc(p.source)} · ${p.kind || ''} · обновлён ${p.updated_h || ''}</div>
    <div class="kv">
      <span>сетка</span><b>${st.verts || 0} вершин · ${st.faces || 0} граней</b>
      <span>габарит</span><b>${bbox || '—'} мм</b>
      <span>объём</span><b>${vol}</b>
      <span>сетка цела</span><b class="${bad ? 'warn' : 'ok'}">${bad ? 'открытых рёбер ' + st.open_edges : 'замкнута'}</b>
      <span>версий</span><b>${vcount}${vcount ? ' · ' + (p.versions[p.versions.length-1].size/1024).toFixed(0) + ' КБ' : ''}</b>
      ${p.state_saved ? `<span>автосохранение</span><b>${p.state_saved_h || ''}</b>` : ''}
    </div>
    <div class="acts">
      <a class="btn" href="/edit?id=${p.id}">Открыть в редакторе</a>
      <a class="btn" href="/api/project/${p.id}/export?kind=3mf">3MF</a>
      <a class="btn" href="/api/project/${p.id}/export?kind=stl">STL</a>
      <button class="mini" data-rename="${p.id}">Переименовать</button>
      <button class="mini" data-del="${p.id}">В корзину</button>
    </div>
  </div>`;
}
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function draw(list) {
  document.getElementById('cards').innerHTML = list.length ? list.map(card).join('')
    : '<div class="empty">Пока пусто — перетащите модель в рамку выше.</div>';
  document.getElementById('stat').textContent = list.length + ' проект(ов)';
}
function log(s, cls) {
  const el = document.getElementById('log');
  el.innerHTML += `<div class="${cls || ''}">${esc(s)}</div>`;
}
async function upload(files) {
  const scale = document.getElementById('scale').value;
  let bad = 0, i = 0;
  for (const f of files) {
    i++;
    log(`[${i}/${files.length}] ${f.name} · ${fmt(f.size)} — загружаю…`);
    const fd = new FormData();
    fd.append('file', f);
    if (scale) fd.append('scale_mm', scale);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const j = await r.json();
      if (!r.ok) { bad++; log('  отказ: ' + (j.error || r.status), 'bad'); continue; }
      log('  готово: ' + j.name + ' · ' + j.stats.verts + ' вершин, ' + j.stats.faces +
          ' граней, ' + j.stats.bbox_mm.map(v => v.toFixed(1)).join(' × ') + ' мм', 'ok');
    } catch (e) { bad++; log('  сеть: ' + e, 'bad'); }
  }
  if (bad) log(`не вышло: ${bad} из ${files.length}. Остальные в списке — обновите страницу, когда разберётесь.`, 'bad');
  else setTimeout(() => location.reload(), 500);   // даём прочитать итог
}
const drop = document.getElementById('drop');
['dragenter','dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('hot'); }));
['dragleave','drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('hot'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files.length) upload([...e.dataTransfer.files]); });
document.getElementById('pick').onclick = () => document.getElementById('file').click();
document.getElementById('file').onchange = (e) => { if (e.target.files.length) upload([...e.target.files]); };
document.getElementById('refresh').onclick = () => location.reload();
document.getElementById('q').oninput = (e) => {
  const q = e.target.value.toLowerCase();
  draw(PROJECTS.filter(p => (p.name + ' ' + p.source).toLowerCase().includes(q)));
};
document.addEventListener('click', async (e) => {
  const rn = e.target.getAttribute('data-rename');
  const del = e.target.getAttribute('data-del');
  if (rn) {
    const name = prompt('Новое название модели:', (PROJECTS.find(p => p.id === rn) || {}).name || '');
    if (!name) return;
    await fetch('/api/project/' + rn + '/rename', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ name }) });
    location.reload();
  }
  if (del) {
    if (!confirm('Убрать проект в корзину? Файлы останутся в папке trash.')) return;
    await fetch('/api/project/' + del, { method: 'DELETE' });
    location.reload();
  }
});
draw(PROJECTS);
</script>
</body></html>
"""


def launcher_html(store, query=''):
    items = store.list(query)
    cards = []
    for m in items:
        m = dict(m)
        m['updated_h'] = human_when(m.get('updated'))
        m['state_saved_h'] = human_when(m.get('state_saved')) if m.get('state_saved') else None
        cards.append(m)
    body = json.dumps(cards, ensure_ascii=False)
    return (LAUNCH_HTML
            .replace('__PROJECTS__', body)
            .replace('__DIR__', html.escape(store.root)))


# ---------------------------------------------------------------------------
# 5. HTTP: API и раздача файлов
# ---------------------------------------------------------------------------

MIME = {
    '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
    '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
    '.json': 'application/json; charset=utf-8', '.png': 'image/png', '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.woff2': 'font/woff2',
    '.glb': 'model/gltf-binary', '.3mf': 'model/3mf', '.stl': 'model/stl',
}
STATIC_DIRS = ('suspension', 'docs', 'reference')


def parse_multipart(body, ctype):
    """Минимальный разбор multipart/form-data: текстовые поля и файлы."""
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', ctype or '')
    if not m:
        raise ModelError('нет boundary в multipart/form-data')
    boundary = (m.group(1) or m.group(2)).strip().encode()
    out = {}
    sep = b'--' + boundary
    for part in body.split(sep):
        if not part or part in (b'--\r\n', b'--', b'\r\n'):
            continue
        head, _, data = part.partition(b'\r\n\r\n')
        if not _:
            continue
        data = data[:-2] if data.endswith(b'\r\n') else data
        head_s = head.decode('utf-8', 'ignore')
        nm = re.search(r'name="([^"]*)"', head_s)
        fn = re.search(r'filename="([^"]*)"', head_s)
        if not nm:
            continue
        if fn:
            out[nm.group(1)] = {'filename': fn.group(1) or 'model', 'data': data}
        else:
            out[nm.group(1)] = data.decode('utf-8', 'ignore').strip()
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = f'TayronStudio/{VERSION}'
    store: Store = None            # заполняется в main()
    quiet = False

    # ---------- служебное ----------
    def log_message(self, fmt, *args):
        if not self.quiet:
            sys.stderr.write(f'{datetime.now().strftime("%H:%M:%S")} {self.address_string()} '
                             f'{fmt % args}\n')

    def _send(self, code, body=b'', ctype='text/plain; charset=utf-8', extra=None):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode('utf-8'),
                   'application/json; charset=utf-8')

    def _err(self, code, msg):
        self._json(code, {'error': msg})

    def _body(self, limit):
        try:
            n = int(self.headers.get('Content-Length') or 0)
        except ValueError:
            n = 0
        if n > limit:
            self._err(413, f'файл больше {human_size(limit)}')
            return None
        return self.rfile.read(n) if n else b''

    def _json_body(self, limit=MAX_JSON):
        raw = self._body(limit)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode('utf-8') or '{}')
        except Exception as e:
            self._err(400, f'плохой JSON: {e}')
            return None

    # ---------- маршруты ----------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(u.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if path == '/':
                return self._send(200, launcher_html(self.store, q.get('q', [''])[0]).encode('utf-8'),
                                  'text/html; charset=utf-8')
            if path == '/api/health':
                return self._json(200, {'ok': True, 'app': APP, 'v': VERSION,
                                        'root': self.store.root, 'python': sys.version.split()[0]})
            if path == '/api/projects':
                items = []
                for m in self.store.list(q.get('q', [''])[0]):
                    m = dict(m)
                    m['updated_h'] = human_when(m.get('updated'))
                    items.append(m)
                return self._json(200, {'projects': items})
            m = re.fullmatch(r'/api/project/([0-9a-f]+)(?:/(\w[\w.]*))?', path)
            if m:
                return self._api_project(m.group(1), m.group(2), q)
            m = re.fullmatch(r'/download/([0-9a-f]+)/(.+)', path)
            if m:
                return self._download(m.group(1), m.group(2))
            if path == '/edit':
                pid = (q.get('id', [''])[0] or '')
                if not pid:
                    return self._send(302, b'', extra={'Location': '/'})
                if not self.store.meta(pid):
                    return self._send(404, 'проект не найден: <a href="/">к списку</a>'.encode(),
                                      'text/html; charset=utf-8')
                return self._send_file(os.path.join(ROOT, 'suspension', 'studio.html'))
            if path in ('/favicon.ico',):
                return self._send(204)
            return self._static(path)
        except BrokenPipeError:
            pass
        except Exception as e:                       # pragma: no cover
            import traceback
            traceback.print_exc()
            return self._err(500, f'внутренняя ошибка: {e}')

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(u.path)
        try:
            if path == '/api/upload':
                return self._upload()
            m = re.fullmatch(r'/api/project/([0-9a-f]+)/(rename|state|save)', path)
            if m:
                pid, what = m.group(1), m.group(2)
                if not self.store.meta(pid):
                    return self._err(404, 'проект не найден')
                if what == 'rename':
                    d = self._json_body(1024 * 1024)
                    if d is None:
                        return
                    meta = self.store.rename(pid, d.get('name'))
                    return self._json(200, {'ok': True, 'meta': meta})
                if what == 'state':
                    raw = self._body(MAX_JSON)
                    if raw is None:
                        return
                    try:
                        json.loads(raw.decode('utf-8'))
                    except Exception:
                        return self._err(400, 'состояние редактора — это JSON')
                    self.store.save_state(pid, raw.decode('utf-8'))
                    return self._json(200, {'ok': True, 'when': now_iso()})
                return self._save(pid)
            return self._err(404, 'нет такого адреса')
        except BrokenPipeError:
            pass
        except Exception as e:                       # pragma: no cover
            import traceback
            traceback.print_exc()
            return self._err(500, f'внутренняя ошибка: {e}')

    def do_DELETE(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        m = re.fullmatch(r'/api/project/([0-9a-f]+)', path)
        if not m:
            return self._err(404, 'нет такого адреса')
        pid = m.group(1)
        if not self.store.meta(pid):
            return self._err(404, 'проект не найден')
        self.store.delete(pid)
        return self._json(200, {'ok': True, 'trashed': pid})

    # ---------- конкретные ручки ----------
    def _api_project(self, pid, sub, q):
        meta = self.store.meta(pid)
        if not meta:
            return self._err(404, 'проект не найден')
        if sub is None:
            return self._json(200, meta)
        if sub == 'body.json':
            body = self.store.body(pid)
            if not body:
                return self._err(404, 'нет геометрии')
            raw = json.dumps(body, separators=(',', ':')).encode()
            if 'gzip' in (self.headers.get('Accept-Encoding') or ''):
                raw = gzip.compress(raw, 5)
                return self._send(200, raw, 'application/json; charset=utf-8',
                                  {'Content-Encoding': 'gzip'})
            return self._send(200, raw, 'application/json; charset=utf-8')
        if sub == 'state':
            st = self.store.state(pid)
            if st is None:
                return self._err(404, 'автосохранения нет')
            return self._send(200, st.encode('utf-8'), 'application/json; charset=utf-8')
        if sub == 'versions':
            return self._json(200, {'versions': meta.get('versions', [])})
        if sub == 'export':
            kind = (q.get('kind', ['3mf'])[0] or '3mf').lower()
            body = self.store.body(pid)
            if not body:
                return self._err(404, 'нет геометрии')
            pos, idx = body['pos'], body['idx']
            data = write_3mf(pos, idx, meta.get('name') or 'Кузов') if kind == '3mf' \
                else write_stl(pos, idx, meta.get('name') or 'Tayron')
            fname = f"{(meta.get('name') or 'model')}.{kind}"
            return self._send(200, data, MIME['.' + kind],
                              {'Content-Disposition': f'attachment; filename="{fname}"'})
        return self._err(404, f'нет ручки {sub}')

    def _upload(self):
        ctype = self.headers.get('Content-Type') or ''
        if 'multipart/form-data' not in ctype:
            return self._err(400, 'нужен multipart/form-data с полем file')
        raw = self._body(MAX_UPLOAD)
        if raw is None:
            return
        try:
            form = parse_multipart(raw, ctype)
        except ModelError as e:
            return self._err(400, str(e))
        f = form.get('file')
        if not isinstance(f, dict) or not f['data']:
            return self._err(400, 'в запросе нет файла (поле file)')
        try:
            kind, pos, idx = read_model(f['filename'], f['data'])
        except ModelError as e:
            return self._err(400, f'{safe_name(f["filename"])}: {e}')
        scale = None
        try:
            if form.get('scale_mm'):
                scale = float(form['scale_mm'])
                if not (1 <= scale <= 5000):
                    return self._err(400, 'масштаб печати — от 1 до 5000 мм')
        except ValueError:
            return self._err(400, 'масштаб печати должен быть числом')
        src = safe_name(f['filename'], 'model.' + kind)
        name = os.path.splitext(src)[0]
        pid, meta = self.store.create(name, src, kind, pos, idx, scale)
        with open(os.path.join(self.store.projects, pid, 'source', src), 'wb') as fh:
            fh.write(f['data'])
        meta['source_size'] = len(f['data'])
        self.store._write_meta(pid, meta)
        return self._json(201, {
            'ok': True, 'id': pid, 'name': meta['name'],
            'url': f'/edit?id={pid}', 'stats': meta['stats'],
            'scale_mm': meta['scale_mm'], 'U': meta['U'],
        })

    def _save(self, pid):
        d = self._json_body(MAX_JSON)
        if d is None:
            return
        kind = (d.get('kind') or '3mf').lower()
        if kind not in ('3mf', 'stl'):
            return self._err(400, 'kind — только 3mf или stl')
        b64 = d.get('data') or ''
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            return self._err(400, 'data — это base64')
        if not data:
            return self._err(400, 'пустой файл')
        if kind == '3mf' and data[:2] != b'PK':
            return self._err(400, 'это не 3MF (нет zip-подписи)')
        if len(data) > MAX_UPLOAD:
            return self._err(413, 'файл слишком большой')
        ver = self.store.save_version(pid, kind, data, d.get('note') or '')
        return self._json(200, {'ok': True, 'version': ver,
                                'url': f'/download/{pid}/{ver["file"]}' if ver else None})

    def _download(self, pid, fname):
        fname = safe_name(fname, '')
        p = self.store.version_path(pid, fname)
        if not p:
            return self._err(404, 'файл не найден')
        with open(p, 'rb') as f:
            data = f.read()
        ext = os.path.splitext(p)[1].lower()
        return self._send(200, data, MIME.get(ext, 'application/octet-stream'),
                          {'Content-Disposition': f'attachment; filename="{fname}"'})

    def _send_file(self, full):
        if not os.path.isfile(full):
            return self._send(404, b'not found')
        ext = os.path.splitext(full)[1].lower()
        with open(full, 'rb') as f:
            data = f.read()
        return self._send(200, data, MIME.get(ext, 'application/octet-stream'))

    def _static(self, path):
        rel = path.lstrip('/')
        if not rel:
            return self._send(404, b'not found')
        first = rel.split('/', 1)[0]
        if first not in STATIC_DIRS:
            return self._send(404, b'not found')
        full = os.path.abspath(os.path.join(ROOT, rel))
        if not full.startswith(os.path.abspath(ROOT) + os.sep):
            return self._send(403, b'no')
        if not os.path.isfile(full) and rel.endswith('/'):
            full = os.path.join(full, 'index.html')
        return self._send_file(full)


# ---------------------------------------------------------------------------
# 6. запуск
# ---------------------------------------------------------------------------


def free_port(host, start):
    for port in range(start, start + 40):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host if host != '0.0.0.0' else '0.0.0.0', port))
                return port
            except OSError:
                continue
    raise SystemExit(f'нет свободного порта начиная с {start}')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Tayron Studio — загрузка, правка и сохранение моделей')
    ap.add_argument('--port', type=int, default=8131, help='порт (по умолчанию 8131)')
    ap.add_argument('--host', default='127.0.0.1', help='адрес; 0.0.0.0 — доступ с других машин')
    ap.add_argument('--dir', default=os.path.join(os.path.expanduser('~'), 'TayronStudio'),
                    help='рабочий каталог с проектами')
    ap.add_argument('--open', action='store_true', help='открыть браузер после запуска')
    ap.add_argument('--quiet', action='store_true', help='не писать каждый запрос в консоль')
    ap.add_argument('--self-test', action='store_true',
                    help='проверить разборщики, запись 3MF/STL и хранилище, затем выйти')
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    store = Store(args.dir)
    Handler.store = store
    Handler.quiet = args.quiet
    port = free_port(args.host, args.port)
    httpd = ThreadingHTTPServer((args.host, port), Handler)
    show_host = '127.0.0.1' if args.host in ('0.0.0.0', '', '::') else args.host
    url = f'http://{show_host}:{port}/'
    print(f'Tayron Studio {VERSION} · {url}')
    print(f'  рабочий каталог: {store.root}')
    print(f'  проектов: {len(store.list())} · Ctrl+C — остановить')
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nостановлено')
    finally:
        httpd.server_close()
    return 0


def self_test(tmpdir=None):
    """Быстрая самопроверка: разбор, запись, хранилище, без сети."""
    import tempfile
    tmp = tmpdir or tempfile.mkdtemp(prefix='studio_selftest_')
    ok = 0

    def check(name, cond, note=''):
        nonlocal ok
        print(('  ok   ' if cond else '  FAIL ') + name + (f' — {note}' if note else ''))
        if not cond:
            raise SystemExit(1)
        ok += 1

    cube_p = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
              [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    cube_i = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
              1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    stl = write_stl(cube_p, cube_i)
    kind, pos, idx = read_model('c.stl', stl)
    check('STL двоичный разобран', kind == 'stl' and len(pos) == 8 and len(idx) == 36)
    tf = write_3mf(cube_p, cube_i, 'куб')
    p2, i2 = read_3mf(tf)
    check('3MF записан и прочитан', len(p2) == 8 and len(i2) == 36)
    st = mesh_stats(cube_p, cube_i)
    check('объём куба 1.0', abs(st['volume'] - 1.0) < 1e-9 and st['open_edges'] == 0)
    store = Store(tmp)
    pid, meta = store.create('куб', 'cube.stl', 'stl', cube_p, cube_i)
    check('проект создан', bool(store.meta(pid)), f'U={meta["U"]}')
    ver = store.save_version(pid, '3mf', tf)
    check('версия записана', ver and store.version_path(pid, ver['file']))
    check('автосохранение', store.save_state(pid, '{"app":"tayron-sculpt"}') and store.state(pid))
    check('переименование', store.rename(pid, 'новое')['name'] == 'новое')
    check('корзина', store.delete(pid) and store.meta(pid) is None)
    print(f'самопроверка: {ok} проверок, 0 провалов')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
