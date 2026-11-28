#!/usr/bin/env python3
"""Сквозные сценарии Tayron Studio (`studio.py`) — через настоящий HTTP.

Поднимает сервер на свободном порту, кладёт ему файлы как браузер
(multipart), правит, сохраняет версии, скачивает их и проверяет, что
всё прочитанное совпадает с тем, что лежит на диске.

Запуск:  python3 tests/studio_api.py        (нужен только python3)
"""
import base64
import io
import json
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import studio  # noqa: E402

ok = 0
bad = 0


def check(name, cond, note=''):
    global ok, bad
    if cond:
        ok += 1
    else:
        bad += 1
    print(('  ok   ' if cond else '  FAIL ') + name + (f' — {note}' if note else ''))


def free_port():
    s = socket.socket()
    s.bind(('127.0.0.1', 0))
    p = s.getsockname()[1]
    s.close()
    return p


def start_server(root):
    from http.server import ThreadingHTTPServer
    studio.Handler.store = studio.Store(root)
    studio.Handler.quiet = True
    for _ in range(20):
        port = free_port()
        try:
            httpd = ThreadingHTTPServer(('127.0.0.1', port), studio.Handler)
        except OSError:
            continue
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        return httpd, f'http://127.0.0.1:{port}'
    raise RuntimeError('не нашли свободный порт')


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


def http(url, method='GET', data=None, headers=None, timeout=120, follow=True):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(req, timeout=timeout) as r:
            body = r.read()
            return r.status, body, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def upload(base, path, scale=None):
    data = open(path, 'rb').read()
    fields = {'scale_mm': str(scale)} if scale else {}
    boundary = '----studio' + uuid.uuid4().hex
    chunks = []
    for k, v in fields.items():
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                  f'filename="{os.path.basename(path)}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
    chunks.append(data + b'\r\n')
    chunks.append(f'--{boundary}--\r\n'.encode())
    payload = b''.join(chunks)
    return http(base + '/api/upload', 'POST', payload,
                {'Content-Type': f'multipart/form-data; boundary={boundary}'})


def post_json(url, obj):
    return http(url, 'POST', json.dumps(obj).encode(), {'Content-Type': 'application/json'})


def cube_3mf():
    p = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    i = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
         1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    return studio.write_3mf(p, i, 'куб')


def main():
    root = tempfile.mkdtemp(prefix='studio_api_')
    httpd, base = start_server(root)
    print(f'Tayron Studio API · {base} · каталог {root}\n')
    try:
        # 1. лаунчер и здоровье
        st, body, _ = http(base + '/')
        check('лаунчер отдаётся и умеет перетаскивание файлов',
              st == 200 and 'Перетащите файл' in body.decode('utf-8'), f'{len(body)} байт')
        st, body, _ = http(base + '/api/health')
        h = json.loads(body)
        check('health: имя, версия, рабочий каталог', st == 200 and h['ok'] and h['root'] == root,
              f"{h['app']} {h['v']}, python {h['python']}")

        # 2. загрузка всех трёх форматов
        glb = os.path.join(REPO, 'suspension/assets/body.glb')
        st, body, _ = upload(base, glb, scale=150)
        d = json.loads(body)
        gid = d['id']
        check('GLB загружен и размечен по масштабу печати',
              st == 201 and d['stats']['verts'] == 52596 and d['stats']['faces'] == 105822,
              f"{d['stats']['verts']} вершин, {d['stats']['faces']} граней, "
              f"габарит {[round(x, 1) for x in d['stats']['bbox_mm']]} мм, "
              f"объём {d['stats']['volume_mm3'] / 1000:.1f} см³")
        check('масштаб: U = длина модели / 150 мм', abs(d['U'] - 5.8827) < 1e-3, f"U = {d['U']:.4f}")

        stl_bin = os.path.join(root, 'probe_bin.stl')
        p = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        i = [0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3]
        open(stl_bin, 'wb').write(studio.write_stl(p, i, 'tetra'))
        st, body, _ = upload(base, stl_bin)
        d2 = json.loads(body)
        check('двоичный STL загружен (сигнатура, а не расширение)',
              st == 201 and d2['stats']['faces'] == 4 and d2['stats']['open_edges'] == 0,
              f"{d2['stats']['verts']} вершин, {d2['stats']['faces']} граней, дыр {d2['stats']['open_edges']}")

        tmf = os.path.join(REPO, 'Tayron_body_smooth_v3.3mf')
        st, body, _ = upload(base, tmf)
        d3 = json.loads(body)
        zid = d3['id']
        check('3MF из репозитория загружен без конвертации',
              st == 201 and d3['stats']['faces'] == 120376,
              f"{d3['stats']['verts']} вершин, {d3['stats']['faces']} граней, "
              f"объём {d3['stats']['volume_mm3'] / 1000:.1f} см³")

        # 3. список проектов
        st, body, _ = http(base + '/api/projects')
        lst = json.loads(body)['projects']
        check('список проектов: три штуки, с сеткой и габаритом',
              st == 200 and len(lst) == 3 and all('stats' in p and 'bbox_mm' in p['stats'] for p in lst),
              ', '.join(f"{p['name']} ({p['stats']['faces']} граней)" for p in lst))

        # 4. тело модели для редактора
        t0 = time.time()
        st, body, hdr = http(base + '/api/project/%s/body.json' % gid)
        ms = (time.time() - t0) * 1000
        b = json.loads(body)
        check('body.json: вершины и индексы для редактора',
              st == 200 and len(b['pos']) == 52596 and len(b['idx']) == 105822 * 3,
              f'{len(body) / 1048576:.1f} МБ за {ms:.0f} мс, U = {b["U"]:.4f}')
        st, body, _ = http(base + '/api/project/%s/body.json' % gid,
                           headers={'Accept-Encoding': 'gzip'})
        gz = body[:2] == b'\x1f\x8b'
        check('body.json жмётся gzip, когда браузер просит',
              gz and len(body) < 2000000, f'{len(body) / 1048576:.1f} МБ сжато')

        # 5. автосохранение правки
        state = json.dumps({'app': 'tayron-sculpt', 'v': 1, 'when': 'now', 'edits': [1, 2, 3]})
        st, body, _ = http(base + '/api/project/%s/state' % gid, 'POST', state.encode())
        check('автосохранение правки принято сервером', st == 200 and json.loads(body)['ok'], body[:80].decode())
        st, body, _ = http(base + '/api/project/%s/state' % gid)
        check('автосохранение читается обратно байт в байт', body.decode() == state,
              f'{len(body)} байт')

        # 6. версии: 3MF и STL
        b64 = base64.b64encode(cube_3mf()).decode()
        st, body, _ = post_json(base + '/api/project/%s/save' % gid, {'kind': '3mf', 'data': b64, 'note': 'из редактора'})
        v = json.loads(body)['version']
        check('«Сохранить 3MF» создаёт версию на диске',
              st == 200 and v['kind'] == '3mf' and v['size'] > 500 and v['file'].endswith('.3mf'),
              f"{v['file']}, {v['size']} байт, заметка «{v['note']}»")

        b64s = base64.b64encode(studio.write_stl(p, i, 'tetra')).decode()
        st, body, _ = post_json(base + '/api/project/%s/save' % gid, {'kind': 'stl', 'data': b64s})
        v2 = json.loads(body)['version']
        check('«Сохранить STL» создаёт вторую версию, не затирая первую',
              st == 200 and v2['file'].endswith('.stl') and v2['file'] != v['file'],
              f"{v2['file']}, {v2['size']} байт")

        st, body, _ = http(base + '/api/project/%s/versions' % gid)
        vers = json.loads(body)['versions']
        check('история версий отдаётся списком с размерами и временем',
              len(vers) == 2 and all('size' in x and 'when' in x for x in vers),
              '; '.join(f"{x['file']} ({x['size']} б)" for x in vers))

        # 7. скачивание версии обратно
        st, body, hdr = http(base + '/download/%s/%s' % (gid, vers[0]['file']))
        zf = zipfile.ZipFile(io.BytesIO(body))
        names = zf.namelist()
        st2, body_stl, _ = http(base + '/download/%s/%s' % (gid, vers[1]['file']))
        faces_stl = struct.unpack('<I', body_stl[80:84])[0] if len(body_stl) > 84 else -1
        check('скачивание версий: 3MF — настоящий zip, STL — ровно 4 грани',
              st == 200 and '3D/3dmodel.model' in names and st2 == 200 and faces_stl == 4,
              f"3MF внутри {len(names)} файлов, STL {len(body_stl)} байт, {faces_stl} граней")

        # 8. экспорт прямо из геометрии проекта
        st, body, hdr = http(base + '/api/project/%s/export?kind=stl' % gid)
        faces = struct.unpack('<I', body[80:84])[0]
        cd = hdr.get('Content-Disposition', '')
        check('экспорт STL из геометрии проекта — по числу граней проекта',
              st == 200 and faces == 105822 and '.stl' in cd,
              f'{faces} граней, {len(body) / 1048576:.1f} МБ, «{cd}»')
        st, body, hdr = http(base + '/api/project/%s/export?kind=3mf' % gid)
        check('экспорт 3MF из геометрии проекта — zip с моделью',
              st == 200 and body[:2] == b'PK' and '3D/3dmodel.model' in zipfile.ZipFile(io.BytesIO(body)).namelist(),
              f'{len(body) / 1048576:.2f} МБ')

        # 9. переименование и редактор
        st, body, _ = post_json(base + '/api/project/%s/rename' % gid, {'name': 'Кузов Tayron (правка)'})
        meta = json.loads(body)['meta']
        check('переименование проекта сохраняется в meta.json',
              st == 200 and meta['name'] == 'Кузов Tayron (правка)' and meta['source'] == 'body.glb',
              f"«{meta['name']}», источник {meta['source']}")

        st, body, _ = http(base + '/edit?id=%s' % gid)
        html = body.decode('utf-8')
        check('страница редактора открывается по /edit?id=',
              st == 200 and 'studio-bar' in html and 'studio.js' in html and 'importmap' in html,
              f'{len(body)} байт')
        check('importmap стоит до модуля studio.js (иначе three не найдётся)',
              html.index('importmap') < html.index("await import('./studio.js')"), 'порядок верный')
        for f in ('/suspension/studio.js', '/suspension/sculpt.js', '/suspension/vendor/three.module.js'):
            st, body, hdr = http(base + f)
            if st != 200:
                break
        check('редактору отдаются все три скрипта', st == 200,
              f'{f.split("/")[-1]}: {len(body) / 1024:.0f} КБ, {hdr.get("Content-Type")}')
        st, body, hdr = http(base + '/edit', follow=False)
        check('/edit без проекта уводит на список моделей', st in (301, 302, 303, 307, 308),
              f'код {st} → {hdr.get("Location")}')

        # 10. ошибки и защита
        junk = os.path.join(root, 'junk.stl')
        open(junk, 'w').write('это не модель')
        st, body, _ = upload(base, junk)
        check('мусор вместо модели объясняется по-русски',
              st == 400 and 'не STL' in json.loads(body)['error'], json.loads(body)['error'])

        st, body, _ = http(base + '/download/%s/../../../../etc/passwd' % gid)
        check('выход из каталога через /download закрыт', st == 404, f'код {st}')
        st, body, _ = http(base + '/suspension/../../etc/passwd')
        check('выход из каталога через статику закрыт', st == 403, f'код {st}')
        st, body, _ = http(base + '/download/%s/%s' % (gid, vers[0]['file'] + '/../../meta.json'))
        check('подкаталог в имени файла версии не открывается', st in (403, 404), f'код {st}')

        st, body, _ = http(base + '/api/project/deadbeef0000')
        check('чужой проект — 404, а не падение', st == 404, body[:60].decode(errors='replace'))

        # 11. корзина
        st, body, _ = http(base + '/api/project/%s' % zid, 'DELETE')
        st2, _, _ = http(base + '/api/project/%s' % zid)
        trash = os.listdir(os.path.join(root, 'trash'))
        check('удаление — в корзину, проект пропадает из работы',
              st == 200 and st2 == 404 and len(trash) == 1, f'в корзине {trash[0][:13]}…')
        st, body, _ = http(base + '/api/projects')
        check('список после удаления короче', len(json.loads(body)['projects']) == 2,
              f"{len(json.loads(body)['projects'])} проекта")

        # 12. перезапуск: проекты и версии на месте
        httpd.shutdown()
        httpd.server_close()
        httpd, base2 = start_server(root)
        st, body, _ = http(base2 + '/api/projects')
        again = json.loads(body)['projects']
        gid2 = [x['id'] for x in again if x['id'] == gid]
        st2, body2, _ = http(base2 + '/api/project/%s/versions' % gid)
        check('после перезапуска сервера проекты и версии на месте',
              gid2 and len(json.loads(body2)['versions']) == 2,
              f'{len(again)} проекта, {len(json.loads(body2)["versions"])} версии у {gid}')
    finally:
        try:
            httpd.shutdown()
            httpd.server_close()
        except Exception:
            pass

    print(f'\nИТОГ API студии: {ok} из {ok + bad} проверок зелёные, провалов {bad}')
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
