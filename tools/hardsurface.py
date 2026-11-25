#!/usr/bin/env python3
"""Чёткий дизайн: жёсткие панели и острые рёбра вместо «мыла» скана.

Задача: скан (или STL из редактора) — гладкий, детали читаются «на ощупь».
В доноре Tiguan кузов собран плоскими панелями с острыми рёбрами, и такой
дизайн читается сразу. Инструмент переводит модель в тот же язык:

1. `facet` — децимация до заданного числа граней (квадричная, форма держится);
2. `planarize` — грани собираются в плоские регионы по нормали И смещению
   плоскости (параллельные, но разнесённые панели не слипаются), на регион
   подгоняется плоскость, вершины садятся в пересечение плоскостей своих
   регионов → панель становится плоскостью, стык — ребром, угол — точкой;
3. `smooth_creases` (по желанию) — лёгкое сглаживание внутри панелей, рёбра
   остаются острыми;
4. повторить 1–2 раза: панели укрупняются, рёбра уточняются.

`report` — целостность: объекты, watertight, замкнутость, объём, компоненты.
`join` — сварить отдельные объекты (колёса + кузов) в один солид.

    python3 tools/hardsurface.py --in model.3mf --out hard.stl --faces 12000
"""
import argparse
import os
import sys

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- целостность
def parts(mesh_or_scene):
    """Разложить вход на отдельные геометрии (у сцены 3mf — с учётом трансформаций)."""
    if isinstance(mesh_or_scene, trimesh.Scene):
        out = []
        for node in mesh_or_scene.graph.nodes_geometry:
            tr, geom_name = mesh_or_scene.graph[node]
            g = mesh_or_scene.geometry[geom_name].copy()
            g.apply_transform(tr)
            out.append((geom_name, g))
        return out
    return [('mesh', mesh_or_scene)]


def report(mesh_or_scene, verbose=True):
    """Целостность модели: по объектам и итого."""
    rows, whole = [], []
    for name, g in parts(mesh_or_scene):
        v = len(g.vertices)
        f = len(g.faces)
        vol = float(g.volume) if g.is_watertight else float('nan')
        rows.append(dict(name=name, vertices=v, faces=f, watertight=bool(g.is_watertight),
                         winding=bool(g.is_winding_consistent), volume=vol,
                         components=int(len(g.split(only_watertight=False))),
                         bounds=g.bounds.copy()))
        whole.append(g)
    allm = trimesh.util.concatenate(whole)
    # самопересечения: сколько пар граней пересекается (дорого — только если попросят)
    info = dict(objects=rows, total_faces=len(allm.faces), total_vertices=len(allm.vertices),
                components=int(len(allm.split(only_watertight=False))),
                watertight=bool(allm.is_watertight), volume=float(allm.volume),
                bbox=allm.bounds.copy(), extent=(allm.bounds[1] - allm.bounds[0]))
    if verbose:
        print(f'объектов: {len(rows)}, граней всего {info["total_faces"]}, '
              f'вершин {info["total_vertices"]}')
        for r in rows:
            print(f'  {r["name"][:34]:34s} граней {r["faces"]:7d}  watertight '
                  f'{"да " if r["watertight"] else "НЕТ"}  компонент {r["components"]:3d}  '
                  f'объём {r["volume"]:12.1f}')
        e = info['extent']
        print(f'габарит: {np.round(e, 1).tolist()}  (Д×Ш×В), components {info["components"]}, '
              f'watertight {"да" if info["watertight"] else "НЕТ"}')
    return info


def self_intersections(mesh, limit=200000):
    """Сколько граней участвует в самопересечениях (грубая оценка по bbox-парам)."""
    if len(mesh.faces) > limit:
        return None
    try:
        inter = mesh.face_adjacency_convex  # noqa: F841  (прогрев)
        chk = trimesh.proximity.ProximityQuery(mesh)
        hits = chk.signed_distance(mesh.vertices[:0])
        return int(len(hits))
    except Exception:
        return None


# ---------------------------------------------------------------- жёсткие панели
def _faces_adjacency_ok(mesh, ang_deg, dist):
    e = mesh.face_adjacency
    a = mesh.face_adjacency_angles
    n = mesh.face_normals
    c = mesh.vertices[mesh.faces].mean(1)
    off = np.einsum('ij,ij->i', n, c)
    keep = (a < np.radians(ang_deg)) & (np.abs(off[e[:, 0]] - off[e[:, 1]]) < dist)
    return e[keep]


def planarize(mesh, ang_deg=12.0, dist=0.6, lam=0.08, rounds=1, max_move=2.0):
    """Собрать грани в плоские регионы и посадить вершины в пересечение плоскостей."""
    out = mesh
    for _ in range(rounds):
        n = out.face_normals
        c = out.vertices[out.faces].mean(1)
        off = np.einsum('ij,ij->i', n, c)
        e = _faces_adjacency_ok(out, ang_deg, dist)
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                       shape=(len(out.faces),) * 2)
        ncomp, lab = connected_components(g, directed=False)
        # средняя нормаль и смещение региона (по площади)
        w = out.area_faces
        wsum = np.bincount(lab, w, ncomp)
        wn = np.stack([np.bincount(lab, w * n[:, k], ncomp) for k in range(3)], 1)
        wo = np.bincount(lab, w * off, ncomp)
        good = wsum > 1e-12
        wn[good] /= wsum[good][:, None]
        wo[good] /= wsum[good]
        wn[~good] = np.array([0.0, 0.0, 1.0])
        wn /= np.linalg.norm(wn, axis=1, keepdims=True) + 1e-12
        # вершина ← пересечение плоскостей своих граней (+ тяга к исходной точке)
        F = out.faces
        flab = lab
        N = out.vertices
        A = np.zeros((len(N), 3, 3))
        b = np.zeros((len(N), 3))
        for k in range(3):
            v = F[:, k]
            nr = wn[flab]                       # (F,3)
            orr = wo[flab]
            for i in range(3):
                for j in range(3):
                    A[:, i, j] += np.bincount(v, nr[:, i] * nr[:, j], len(N))
                b[:, i] += np.bincount(v, nr[:, i] * orr, len(N))
        A += lam * np.eye(3)[None]
        b += lam * N
        try:
            new = np.linalg.solve(A, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            new = np.stack([np.linalg.lstsq(A[i], b[i], rcond=None)[0] for i in range(len(N))])
        # далеко не двигаем: большие сдвиги рвут топологию (дырки, немогообразные рёбра)
        far = ~np.isfinite(new).all(1)
        far |= np.linalg.norm(new - N, axis=1) > max_move
        new[far] = N[far]
        out = trimesh.Trimesh(vertices=new, faces=F, process=False)
    return out


def clean(mesh, merge=True):
    """Убрать вырожденные и дублирующиеся грани, сварить вершины, выправить нормали."""
    m = mesh.copy()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    if merge:
        m.merge_vertices()
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    return m


def fix_nonmanifold(mesh, rounds=4):
    """Убрать «зажимы»: на ребре, где сходятся >2 граней, оставить две крупнейшие,
    остальные удалить и заделать дырку. Так модель снова становится watertight."""
    m = mesh
    for _ in range(rounds):
        e = np.sort(m.edges, axis=1)
        u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        bad = np.where(cnt > 2)[0]
        if not len(bad):
            break
        eid = inv.reshape(-1, 3)
        area = m.area_faces
        drop = set()
        for b in bad:
            faces = np.where((eid == b).any(1))[0]
            if len(faces) <= 2:
                continue
            keep = faces[np.argsort(-area[faces])[:2]]
            drop.update(int(f) for f in faces if f not in keep)
        if not drop:
            break
        m = m.copy()
        m.update_faces(~np.isin(np.arange(len(m.faces)), list(drop)))
        m = clean(m)
        try:
            trimesh.repair.fill_holes(m)
        except Exception:                       # noqa: BLE001
            pass
        m = clean(m)
    return m


def split_pinched(mesh):
    """Расщепить «зажимы»: там, где по одному ребру сходятся 4+ граней (две
    поверхности касаются ребром), развести их, продублировав вершины ребра для
    второй пары граней. Так две склеенные оболочки снова становятся двумя
    замкнутыми.

    Важно: правки копятся в одном массиве граней (newF), иначе второе ребро
    затирает замены первого — на этом прошлый вариант и ломал кузов.
    """
    m = mesh.copy()
    V = [v for v in m.vertices]
    newF = m.faces.copy()
    e = np.sort(m.edges, axis=1)
    u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    eid = inv.reshape(-1, 3)
    for b in np.where(cnt > 2)[0]:
        faces = np.where((eid == b).any(1))[0]
        a, c = int(u[b][0]), int(u[b][1])
        pa, pc = m.vertices[a], m.vertices[c]
        ax = pc - pa
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        ref = np.cross(ax, [0.0, 0.0, 1.0])
        if np.linalg.norm(ref) < 1e-6:
            ref = np.cross(ax, [1.0, 0.0, 0.0])
        ref /= np.linalg.norm(ref) + 1e-12
        ref2 = np.cross(ax, ref)
        ang = []
        for f in faces:
            third = [i for i in m.faces[f] if i not in (a, c)]
            d = m.vertices[third[0]] - pa
            d = d - ax * np.dot(d, ax)
            ang.append(np.arctan2(float(np.dot(d, ref2)), float(np.dot(d, ref))))
        order = faces[np.argsort(ang)]
        for k in range(2, len(order), 2):
            na, nc = len(V), len(V) + 1
            V.append(pa.tolist())
            V.append(pc.tolist())
            for f in order[k:k + 2]:
                newF[f] = [na if i == a else (nc if i == c else i) for i in newF[f]]
    out = trimesh.Trimesh(vertices=np.array(V), faces=newF, process=False)
    out = clean(out, merge=False)               # сварка снова склеила бы зажим
    try:
        trimesh.repair.fill_holes(out)
    except Exception:                           # noqa: BLE001
        pass
    return clean(out, merge=False)              # без сварки: иначе зажим вернётся


def fix_nonmanifold(mesh, rounds=4):
    """Убрать «зажимы»: на ребре, где сходятся >2 граней, оставить две крупнейшие,
    остальные удалить и заделать дырку. Так модель снова становится watertight."""
    m = mesh
    for _ in range(rounds):
        e = np.sort(m.edges, axis=1)
        u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        bad = np.where(cnt > 2)[0]
        if not len(bad):
            break
        eid = inv.reshape(-1, 3)
        area = m.area_faces
        drop = set()
        for b in bad:
            faces = np.where((eid == b).any(1))[0]
            if len(faces) <= 2:
                continue
            keep = faces[np.argsort(-area[faces])[:2]]
            drop.update(int(f) for f in faces if f not in keep)
        if not drop:
            break
        m = m.copy()
        m.update_faces(~np.isin(np.arange(len(m.faces)), list(drop)))
        m = clean(m)
        try:
            trimesh.repair.fill_holes(m)
        except Exception:                       # noqa: BLE001
            pass
        m = clean(m)
    return m


def split_pinched(mesh):
    """Расщепить «зажимы»: там, где по одному ребру сходятся 4+ граней (две
    поверхности касаются ребром), развести их, продублировав вершины ребра.
    Так две склеенные оболочки снова становятся двумя замкнутыми."""
    m = mesh.copy()
    V = [v for v in m.vertices]
    F = m.faces.copy()
    e = np.sort(m.edges, axis=1)
    u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    eid = inv.reshape(-1, 3)
    for b in np.where(cnt > 2)[0]:
        faces = np.where((eid == b).any(1))[0]
        a, c = int(u[b][0]), int(u[b][1])
        pa, pc = m.vertices[a], m.vertices[c]
        ax = pc - pa
        ax = ax / (np.linalg.norm(ax) + 1e-12)
        ref = np.cross(ax, [0.0, 0.0, 1.0])
        if np.linalg.norm(ref) < 1e-6:
            ref = np.cross(ax, [1.0, 0.0, 0.0])
        ref /= np.linalg.norm(ref) + 1e-12
        ref2 = np.cross(ax, ref)
        ang = []
        for f in faces:
            third = [i for i in m.faces[f] if i not in (a, c)]
            d = m.vertices[third[0]] - pa
            d = d - ax * np.dot(d, ax)
            ang.append(np.arctan2(float(np.dot(d, ref2)), float(np.dot(d, ref))))
        order = faces[np.argsort(ang)]
        for k in range(2, len(order), 2):
            grp = order[k:k + 2]
            na, nc = len(V), len(V) + 1
            V.append(pa.tolist())
            V.append(pc.tolist())
            for f in grp:
                F[f] = [na if i == a else (nc if i == c else i) for i in m.faces[f]]
    out = trimesh.Trimesh(vertices=np.array(V), faces=F, process=False)
    out = clean(out, merge=False)           # сварка снова склеила бы зажим
    try:
        trimesh.repair.fill_holes(out)      # заделать контуры, оставшиеся от расщепа
    except Exception:                       # noqa: BLE001
        pass
    return clean(out)


def drop_slivers(mesh, keep=0.005):
    """Убрать крошечные оболочки, которые появляются при децимации (зажимы, 0 объёма)."""
    comps = mesh.split(only_watertight=False)
    if len(comps) < 2:
        return mesh
    vols = np.array([abs(float(c.volume)) for c in comps])
    big = vols >= max(vols.max() * keep, 1e-9)
    if big.all():
        return mesh
    out = trimesh.util.concatenate([c for c, b in zip(comps, big) if b])
    return clean(out)


def decimate(mesh, target):
    import fast_simplification as fs
    if target >= len(mesh.faces):
        return mesh.copy()
    v, f = fs.simplify(mesh.vertices, mesh.faces, target_count=int(target), agg=6)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.merge_vertices()
    return m


def facet(mesh, target=12000, ang=12.0, dist=0.6, rounds=2, verbose=True, max_move=None):
    """Децимация + плоские панели: гладкий скан → чёткая низкополигональная модель."""
    mm = max_move if max_move is not None else 2.5 * dist
    m = clean(mesh)
    # 1-й проход: децимация + плоские панели; дальше панели только укрупняются
    m = clean(planarize(drop_slivers(decimate(m, target)), ang, dist, max_move=mm))
    m = split_pinched(fix_nonmanifold(m))
    for r in range(max(rounds - 1, 0)):
        m = clean(planarize(m, ang * 1.6, dist * 1.6, max_move=mm))
        m = split_pinched(fix_nonmanifold(m))
        if verbose:
            a = np.degrees(m.face_adjacency_angles)
            print(f'  проход {r + 2}: граней {len(m.faces)}, рёбер >20° '
                  f'{float((a > 20).mean()):.3f}, watertight {m.is_watertight}')
    return m


def solidify(meshes, pitch=3.0, verbose=True):
    """Сварить всё в один солид через воксели: заливка + marching cubes.

    Надёжнее булевых операций: заливка не боится зажимов, мелких дырок и
    касаний — на выходе всегда одна замкнутая оболочка (если тела реально
    соединены в пространстве).
    """
    from trimesh.voxel.ops import matrix_to_marching_cubes  # noqa: F401
    allm = trimesh.util.concatenate(list(meshes))
    vg = allm.voxelized(pitch=pitch)
    grid = vg.fill()
    out = grid.marching_cubes
    # marching cubes отдаёт индексы вокселей — вернуть в координаты модели
    out.apply_transform(grid.transform)          # индексы вокселей → координаты модели
    out = clean(out)
    if verbose:
        print(f'  воксели: pitch {pitch}, сетка {tuple(int(v) for v in grid.shape)}, '
              f'граней {len(out.faces)}, watertight {"да" if out.is_watertight else "НЕТ"}, '
              f'компонент {len(out.split(only_watertight=False))}')
    return out


def shrink_wrap(mesh, target):
    """Протянуть поверхность к исходной модели: вершины — к ближайшей точке.

    Воксельная сварка даёт «лестницу» с шагом вокселя; проекция возвращает
    вершины на настоящую поверхность, топология (и watertight) не меняются.
    """
    q = trimesh.proximity.ProximityQuery(target)
    pts, dist, _ = q.on_surface(mesh.vertices)
    return trimesh.Trimesh(vertices=pts, faces=mesh.faces, process=False), dist


def _axis_of(part, centre):
    """Ось колеса: направление наименьшего размаха, знаком — к центру кузова."""
    e, v = np.linalg.eigh(np.cov(part.vertices.T))
    ax = v[:, 0]
    ext = part.bounds[1] - part.bounds[0]
    if np.dot(ax, centre - part.bounds.mean(0)) < 0:
        ax = -ax
    return ax / np.linalg.norm(ax), float(np.ptp(np.abs(part.vertices @ ax)))


def split_parts(mesh, min_faces=20, min_area_frac=1e-4):
    """Отдельные оболочки: кузов (самая крупная) и остальное (колёса, мелочь).

    Мелочь отбрасывается сразу: в моделях из редактора рядом с телом висят
    десятки «пылинок» по 2–6 граней с нулевой площадью.
    """
    shells = [c for c in mesh.split(only_watertight=False) if len(c.faces) >= min_faces]
    if not shells:
        return mesh, []
    top = max(float(c.area) for c in shells)
    shells = [c for c in shells if float(c.area) >= top * min_area_frac]
    body = max(shells, key=lambda c: float(c.area))
    return body, [c for c in shells if c is not body]


def repair(part, verbose=False):
    """Довести оболочку до watertight: расщепить зажимы, заделать дырки, сварить."""
    m = part
    if m.is_watertight:
        return clean(m)
    m = clean(m)
    m = split_pinched(m)
    try:
        trimesh.repair.fill_holes(m)
    except Exception:                           # noqa: BLE001
        pass
    m = clean(m)
    if not m.is_watertight:
        m2 = split_pinched(m)
        try:
            trimesh.repair.fill_holes(m2)
        except Exception:                       # noqa: BLE001
            pass
        m = clean(m2)
    if verbose:
        print(f'  ремонт: граней {len(m.faces)}, watertight '
              f'{"да" if m.is_watertight else "НЕТ"}')
    return m


def join(mesh, bridge=True, verbose=True, voxel_pitch=2.5):
    """Соединить колёса с кузовом в один солид.

    Колёса в таких моделях — отдельные замкнутые оболочки, стоящие в арке с
    зазором, поэтому простое объединение оставляет их отдельными телами.
    Поэтому на каждое колесо строится ступица: цилиндр вдоль оси колеса,
    который входит внутрь колеса и уходит внутрь кузова — после булева
    объединения всё становится одним watertight-телом.
    """
    body, rest = split_parts(mesh)
    body = repair(body, verbose)
    rest = [repair(c) for c in rest]
    vol_body = abs(float(body.volume)) if body.is_watertight else float(np.prod(body.extents))
    vol_body = vol_body or 1.0
    centre = body.bounds.mean(0)
    wheels, junk = [], []
    for c in rest:
        vc = abs(float(c.volume)) if c.is_watertight else float(np.prod(c.extents))
        if 0.001 * vol_body < vc < 0.25 * vol_body:
            wheels.append(c)
        else:
            junk.append(c)
    bridges = []
    for w in wheels:
        ax, thick = _axis_of(w, centre)
        wc = w.bounds.mean(0)
        inner = wc + ax * (thick / 2.0)
        loc, idx, _ = body.ray.intersects_location(
            [inner], [ax], multiple_hits=False)
        reach = float(np.linalg.norm(loc[0] - inner)) if len(loc) else 30.0
        r = 0.30 * float(np.linalg.norm((w.bounds[1] - w.bounds[0]) - ax * thick) / 2.0)
        # ступица идёт насквозь через колесо и входит в кузов — иначе объединение
        # оставляет колесо отдельным телом (проверено: колесо∩короткая ступица = 0)
        start = wc - ax * (thick / 2.0 + 6.0)
        end = inner + ax * (reach + 15.0)
        h = float(np.linalg.norm(end - start))
        cyl = trimesh.creation.cylinder(radius=max(r, 1.0), height=h, sections=40)
        T = trimesh.geometry.align_vectors([0, 0, 1], ax)
        cyl.apply_transform(T)
        cyl.apply_translation((start + end) / 2.0)
        bridges.append(cyl)
        if verbose:
            print(f'  колесо: Ø {2 * r:.1f}, ось {np.round(ax, 2).tolist()}, '
                  f'до кузова {reach:.1f} → ступица {h:.1f}')
    solids = [body] + wheels + bridges
    out = None
    if all(x.is_watertight for x in solids):
        try:
            out = trimesh.boolean.union(solids, engine='manifold')
        except Exception as e:                  # noqa: BLE001
            if verbose:
                print(f'  булево объединение не прошло ({e}) — свариваю вокселями')
    voxel = out is None
    if voxel:
        out = solidify(solids, pitch=voxel_pitch, verbose=verbose)
        out, dist = shrink_wrap(out, trimesh.util.concatenate(solids))
        if verbose:
            print(f'  протяжка к исходной поверхности: медиана {float(np.median(dist)):.2f}, '
                  f'95 % {float(np.percentile(dist, 95)):.2f} юнита; '
                  f'watertight {"да" if out.is_watertight else "НЕТ"}')
    if voxel:
        # после вокселей ни сварка, ни выкидывание «вырожденных» граней нельзя:
        # вырожденные появляются при протяжке, но их удаление рвёт оболочку
        out = out.copy()
        trimesh.repair.fix_normals(out)
    else:
        out = clean(out)
    if verbose:
        print(f'  соединение: тел {len(solids)} → граней {len(out.faces)}, '
              f'компонент {len(out.split(only_watertight=False))}, '
              f'watertight {"да" if out.is_watertight else "НЕТ"}')
        if junk:
            print(f'  отброшено мелочи: {len(junk)} оболочек '
                  f'({sum(len(j.faces) for j in junk)} граней)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', dest='dst', default='')
    ap.add_argument('--faces', type=int, default=0, help='целевое число граней (0 — только отчёт)')
    ap.add_argument('--angle', type=float, default=12.0, help='порог «одна плоскость», градусы')
    ap.add_argument('--dist', type=float, default=0.6, help='порог смещения плоскости, юниты')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--join', dest='join', action='store_true', default=True,
                    help='соединить колёса с кузовом (по умолчанию да)')
    ap.add_argument('--no-join', dest='join', action='store_false')
    ap.add_argument('--voxel-pitch', dest='pitch', type=float, default=3.0,
                    help='шаг вокселей при сварке в один солид, юниты')
    a = ap.parse_args()
    src = trimesh.load(a.src, force='mesh' if not a.src.lower().endswith('.3mf') else None)
    print(f'вход: {a.src}')
    report(src)
    if a.faces and a.dst:
        flat = (trimesh.util.concatenate([g for _, g in parts(src)])
                if isinstance(src, trimesh.Scene) else src)
        if a.join:
            print('соединение колёс с кузовом:')
            flat = join(flat, voxel_pitch=a.pitch)
        out = facet(flat, a.faces, a.angle, a.dist, a.rounds)
        out.export(a.dst)
        print(f'выход: {a.dst}: граней {len(out.faces)}, watertight {out.is_watertight}, '
              f'{os.path.getsize(a.dst) / 1e6:.2f} МБ')
        # сравнение «сколько острых рёбер», как в доноре
        for tag, m in (('вход', flat), ('выход', out)):
            ang = np.degrees(m.face_adjacency_angles)
            print(f'  {tag}: рёбер >20° {float((ang > 20).mean()):.3f}, '
                  f'медиана {float(np.median(ang)):.1f}°')


if __name__ == '__main__':
    main()
