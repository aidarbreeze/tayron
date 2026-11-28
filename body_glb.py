#!/usr/bin/env python3
"""3MF кузова → suspension/assets/body.glb (то, что грузит сайт).

    python3 body_glb.py "Tayron_body_smooth (4).3mf"
    python3 body_glb.py --dump body.json "Tayron_body_smooth (4).3mf"   # ещё и массивы для стендов

Координаты в 3MF — те же единицы, что у сайта (1 единица = 0.17 мм печати),
поэтому масштаб не меняется: сайт берёт вершины как есть. GLB собирается так же,
как прежний body.glb от trimesh: только POSITION и индексы, без материалов и
нормалей — сайт включает flatShading и красит вершины сам.
"""
import json
import re
import struct
import sys
import zipfile
from pathlib import Path


def read_3mf(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('.model')]
        if not names:
            raise SystemExit('в файле нет 3D/*.model')
        xml = z.read(names[0]).decode('utf8')
    objects = re.findall(r'<object\b[^>]*>', xml)
    verts = [(float(a), float(b), float(c)) for a, b, c in
             re.findall(r'<vertex x="([-\d.eE+]+)" y="([-\d.eE+]+)" z="([-\d.eE+]+)"', xml)]
    tris = [(int(a), int(b), int(c)) for a, b, c in
            re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', xml)]
    if not verts or not tris:
        raise SystemExit('не нашёл вершины/треугольники')
    if len(objects) > 1:
        print(f'внимание: объектов {len(objects)} — индексы взяты сквозные, проверьте результат')
    top = max(max(t) for t in tris)
    if top >= len(verts):
        raise SystemExit(f'индекс {top} больше числа вершин {len(verts)}')
    return verts, tris


def build_glb(verts, tris, generator='body_glb.py'):
    pos = b''.join(struct.pack('<3f', *v) for v in verts)
    idx = b''.join(struct.pack('<3I', *t) for t in tris)
    while len(idx) % 4:                      # bufferView на 4 байта
        idx += b'\0'
    bin_chunk = idx + pos
    nv, nf = len(verts), len(tris)
    gltf = {
        'asset': {'version': '2.0', 'generator': generator},
        'scene': 0,
        'scenes': [{'nodes': [0]}],
        'nodes': [{'name': 'geometry_0', 'mesh': 0}],
        'meshes': [{'name': 'geometry_0', 'extras': {},
                    'primitives': [{'attributes': {'POSITION': 1}, 'indices': 0, 'mode': 4}]}],
        'accessors': [
            {'componentType': 5125, 'type': 'SCALAR', 'bufferView': 0, 'count': nf * 3,
             'max': [nv - 1], 'min': [0]},
            {'componentType': 5126, 'type': 'VEC3', 'bufferView': 1, 'count': nv,
             'max': [max(v[0] for v in verts), max(v[1] for v in verts), max(v[2] for v in verts)],
             'min': [min(v[0] for v in verts), min(v[1] for v in verts), min(v[2] for v in verts)]},
        ],
        'bufferViews': [
            {'buffer': 0, 'byteOffset': 0, 'byteLength': len(idx)},
            {'buffer': 0, 'byteOffset': len(idx), 'byteLength': len(pos)},
        ],
        'buffers': [{'byteLength': len(bin_chunk)}],
    }
    js = json.dumps(gltf, separators=(',', ':')).encode('utf8')
    while len(js) % 4:
        js += b' '
    out = struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(bin_chunk))
    out += struct.pack('<II', len(js), 0x4E4F534A) + js
    out += struct.pack('<II', len(bin_chunk), 0x004E4942) + bin_chunk
    return out


def main(argv):
    dump = None
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == '--dump':
            dump = argv[i + 1]
            i += 2
        else:
            args.append(argv[i])
            i += 1
    src = Path(args[0] if args else 'Tayron_body_smooth (4).3mf')
    dst = Path(args[1] if len(args) > 1 else 'suspension/assets/body.glb')
    verts, tris = read_3mf(src)
    data = build_glb(verts, tris)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)
    print(f'{src} → {dst}: вершин {len(verts)}, треугольников {len(tris)}, {len(data) / 1048576:.2f} МБ')
    if dump:
        flat = [i for t in tris for i in t]      # стенды ждут плоский список индексов
        Path(dump).write_text(json.dumps({'pos': [list(v) for v in verts], 'idx': flat}))
        print(f'массивы для стендов → {dump}')


if __name__ == '__main__':
    main(sys.argv[1:])
