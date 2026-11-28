#!/usr/bin/env python3
"""Кузов приложения -> tests/body.json (позиции + индексы) для тестов лепки.

Приложение грузит suspension/assets/body.glb; тестам нужен простой JSON.
Запуск:  python3 tests/mkbody.py [--glb путь] [--out путь]
"""
import json
import struct
import sys

ROOT = __file__.rsplit('/tests/', 1)[0]


def read_glb(path):
    data = open(path, 'rb').read()
    magic, version, _ = struct.unpack_from('<III', data, 0)
    assert magic == 0x46546C67, 'не GLB'
    off, js, bin_ = 12, None, None
    while off < len(data):
        clen, ctype = struct.unpack_from('<II', data, off)
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode('utf-8'))
        elif ctype == 0x004E4942:
            bin_ = chunk
        off += 8 + clen + (-clen % 4)
    return js, bin_


COMP = {5126: ('f', 4), 5125: ('I', 4), 5123: ('H', 2), 5121: ('B', 1)}
NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def accessor(js, bin_, i):
    a = js['accessors'][i]
    bv = js['bufferViews'][a['bufferView']]
    fmt, size = COMP[a['componentType']]
    n = NCOMP[a['type']]
    stride = bv.get('byteStride') or size * n
    base = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    out = []
    for k in range(a['count']):
        out.append(struct.unpack_from('<' + fmt * n, bin_, base + k * stride))
    return out


def main():
    glb = ROOT + '/suspension/assets/body.glb'
    out = ROOT + '/tests/body.json'
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--glb':
            glb = args[i + 1]
        if a == '--out':
            out = args[i + 1]
    js, bin_ = read_glb(glb)
    prims = [p for m in js['meshes'] for p in m['primitives']]
    assert len(prims) == 1, 'ждали одну часть кузова, получили %d' % len(prims)
    p = prims[0]
    pos = [list(v) for v in accessor(js, bin_, p['attributes']['POSITION'])]
    idx = [t[0] for t in accessor(js, bin_, p['indices'])]
    json.dump({'pos': pos, 'idx': idx}, open(out, 'w'), separators=(',', ':'))
    print('body.json: %d вершин, %d граней -> %s' % (len(pos), len(idx) // 3, out))


if __name__ == '__main__':
    main()
