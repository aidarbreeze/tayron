#!/usr/bin/env python3
"""Build a Tayron-like SUV mesh FROM SCRATCH, using only the reference photos.

Nothing is read from `tayron lite 2.stl` or from the corrected model: every
vertex here comes from the parametric definition below.  The proportions in
PROPORTIONS were measured off the reference photographs (see
reports/photo_proportions.json) and are expressed in the same nominal units
as the earlier work (overall length 900) so the two models can be rendered
side by side at the same scale.

Axes: +X = front, +Y = left, +Z = up, ground at Z = 0.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator
from shapely.geometry import Polygon

# --------------------------------------------------------------------------
# Proportions measured from the photographs
# --------------------------------------------------------------------------
PROPORTIONS = {
    'length': 900.0,
    'wheelbase': 566.8,
    'axle_front_x': 283.4,
    'axle_rear_x': -283.4,
    'tyre_radius': 71.4,
    'tyre_width': 45.0,
    'track_half': 152.0,
    'rim_radius': 48.0,
    'body_half_width': 184.0,
    'mirror_half_span': 199.0,
    'roof_height': 322.0,
    'rail_height': 332.4,
    'beltline_height': 218.0,
    'glass_top_height': 315.4,
    'hood_cowl_height': 216.0,
    'hood_front_height': 198.0,
    'sill_height': 40.0,
    'arch_radius': 84.0,
    'rear_light_bar_height': 211.0,
    'roof_half_width': 120.0,
    'side_glass_rear_x': -303.0,
    'side_glass_front_x': 171.0,
    'cowl_x': 245.0,
    'a_pillar_top_x': 58.0,
    'roof_rear_x': -330.0,
    'rear_glass_base_x': -437.0,
}

ENGINE = 'manifold'


def boolean(kind, meshes):
    meshes = [m for m in meshes if m is not None and len(m.faces)]
    if kind == 'union':
        return trimesh.boolean.union(meshes, engine=ENGINE)
    if kind == 'inter':
        return trimesh.boolean.intersection(meshes, engine=ENGINE)
    return trimesh.boolean.difference([meshes[0]] + meshes[1:], engine=ENGINE)


# --------------------------------------------------------------------------
# Small primitive helpers (all procedural)
# --------------------------------------------------------------------------
def rounded_box(extents, radius, subdivisions=1):
    """Axis-aligned box with rounded edges, as the hull of corner spheres."""
    ex, ey, ez = np.asarray(extents, dtype=float) / 2.0
    r = float(min(radius, ex * 0.98, ey * 0.98, ez * 0.98))
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=r).vertices
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(sphere + np.array([sx * (ex - r), sy * (ey - r), sz * (ez - r)]))
    return trimesh.convex.convex_hull(np.vstack(pts))


def place(mesh, center=(0, 0, 0), rot=None):
    m = mesh.copy()
    if rot is not None:
        m.apply_transform(rot)
    m.apply_translation(center)
    return m


def rot_y(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 1, 0])


def rot_x(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [1, 0, 0])


def rot_z(deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])


def prism(points_xz, y_from, y_to):
    """Extrude a 2-D outline given in the XZ plane along Y."""
    poly = Polygon(points_xz)
    if not poly.is_valid:
        poly = poly.buffer(0)
    m = trimesh.creation.extrude_polygon(poly, y_to - y_from, engine=ENGINE)
    # extrude_polygon works in XY + Z; map (x, y2d, zext) -> (x, yext, y2d)
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 0], v[:, 2] + y_from, v[:, 1]])
    m.fix_normals()
    return m


def prism_xy(points_xy, z_from, z_to):
    poly = Polygon(points_xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    m = trimesh.creation.extrude_polygon(poly, z_to - z_from, engine=ENGINE)
    m.apply_translation([0, 0, z_from])
    return m


def loft(rings, cap_start=True, cap_end=True):
    """Skin a list of equal-length closed rings (each (n,3)) into a solid."""
    rings = np.asarray(rings, dtype=float)
    k, n, _ = rings.shape
    faces = []
    for i in range(k - 1):
        a = rings[i]
        b = rings[i + 1]
        for j in range(n):
            j2 = (j + 1) % n
            ia, ia2 = i * n + j, i * n + j2
            ib, ib2 = (i + 1) * n + j, (i + 1) * n + j2
            faces.append([ia, ib, ib2])
            faces.append([ia, ib2, ia2])
    verts = list(rings.reshape(-1, 3))
    if cap_start:
        c = len(verts)
        verts.append(rings[0].mean(axis=0))
        for j in range(n):
            faces.append([c, (j + 1) % n, j])
    if cap_end:
        c = len(verts)
        verts.append(rings[-1].mean(axis=0))
        base = (k - 1) * n
        for j in range(n):
            faces.append([c, base + j, base + (j + 1) % n])
    m = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=True)
    m.merge_vertices()
    m.fix_normals()
    if m.volume < 0:
        m.invert()
    return m


def revolve(profile, sections=96):
    """Revolve a closed (r, z) profile around Z."""
    p = np.asarray(profile, dtype=float)
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    return trimesh.creation.revolve(p, sections=sections)


def sweep(path, section, up=(0, 0, 1)):
    """Sweep a closed 2-D section along a 3-D path (parallel transport frame)."""
    path = np.asarray(path, dtype=float)
    sec = np.asarray(section, dtype=float)
    per_station = sec.ndim == 3
    tangents = np.gradient(path, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    up = np.asarray(up, dtype=float)
    rings = []
    for i, t in enumerate(tangents):
        a = np.cross(t, up)
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(t, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(t, a)
        s_i = sec[i] if per_station else sec
        rings.append(path[i] + s_i[:, 0][:, None] * a + s_i[:, 1][:, None] * b)
    return loft(rings)


def ellipse_section(w, h, n=24):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([w / 2 * np.cos(a), h / 2 * np.sin(a)])


def rounded_section(w, h, r=1.0, n=6):
    w, h = w / 2.0, h / 2.0
    r = min(r, w * 0.9, h * 0.9)
    pts = []
    for cx, cy, a0 in ((w - r, h - r, 0), (-w + r, h - r, 90),
                       (-w + r, -h + r, 180), (w - r, -h + r, 270)):
        a = np.radians(np.linspace(a0, a0 + 90, n))
        pts.append(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))
    return np.vstack(pts)


# --------------------------------------------------------------------------
# Body shell: cross-sections lofted along the length
# --------------------------------------------------------------------------
# station:  x      z0    w_sill w_max z_hip w_belt z_belt w_roof z_roof
STATIONS = np.array([
    [-450.0,  46.0, 148.0, 170.0, 118.0, 166.0, 226.0, 150.0, 227.5],
    [-444.0,  41.0, 160.0, 178.0, 120.0, 173.0, 227.0, 152.0, 231.0],
    [-437.0,  38.0, 168.0, 182.0, 124.0, 177.0, 227.0, 152.0, 234.0],
    [-420.0,  37.0, 171.0, 184.0, 128.0, 179.0, 226.0, 147.0, 250.0],
    [-395.0,  37.0, 172.0, 184.0, 130.0, 179.0, 224.0, 138.0, 272.0],
    [-365.0,  38.0, 170.0, 184.0, 130.0, 178.0, 222.0, 128.0, 294.0],
    [-330.0,  40.0, 168.0, 184.0, 130.0, 177.0, 221.0, 122.0, 308.0],
    [-280.0,  40.0, 166.0, 184.0, 130.0, 176.0, 220.0, 120.0, 315.0],
    [-200.0,  40.0, 166.0, 184.0, 130.0, 175.0, 219.0, 120.0, 319.0],
    [-100.0,  40.0, 166.0, 183.5, 130.0, 174.5, 218.5, 120.0, 321.5],
    [0.0,     40.0, 166.0, 183.0, 130.0, 174.0, 218.0, 120.0, 322.0],
    [58.0,    40.0, 166.0, 183.0, 130.0, 174.0, 218.0, 121.0, 320.0],
    [110.0,   40.0, 166.0, 183.0, 130.0, 173.5, 217.5, 132.0, 296.0],
    [160.0,   40.0, 166.0, 182.5, 130.0, 173.0, 217.0, 146.0, 264.0],
    [205.0,   40.0, 165.0, 181.5, 129.0, 172.0, 216.5, 158.0, 234.0],
    [245.0,   41.0, 164.0, 180.0, 128.0, 171.0, 210.0, 163.0, 216.0],
    [300.0,   41.0, 163.0, 179.0, 126.0, 170.0, 205.0, 162.0, 211.0],
    [360.0,   40.0, 161.0, 177.5, 123.0, 168.0, 200.0, 160.0, 206.0],
    [410.0,   40.0, 159.0, 176.0, 120.0, 166.0, 195.0, 158.0, 201.0],
    [437.0,   42.0, 155.0, 173.0, 118.0, 163.0, 191.0, 155.0, 197.0],
    [446.0,   46.0, 148.0, 167.0, 118.0, 158.0, 189.0, 150.0, 195.0],
    [450.0,   52.0, 138.0, 156.0, 118.0, 148.0, 187.0, 140.0, 193.0],
])


def _interp_stations():
    x = STATIONS[:, 0]
    return [PchipInterpolator(x, STATIONS[:, i]) for i in range(1, STATIONS.shape[1])]


def body_section(params, x, n_half=64):
    z0, w_sill, w_max, z_hip, w_belt, z_belt, w_roof, z_roof = [float(f(x)) for f in params]
    z_roof = max(z_roof, z_belt + 1.2)
    w_roof = min(w_roof, w_belt - 1.0)
    gh = z_roof - z_belt          # greenhouse height at this station
    ctrl = [
        (0.0, z0),
        (w_sill * 0.55, z0),
        (w_sill, z0 + 7.0),
        (w_max * 0.985, z0 + (z_hip - z0) * 0.55),
        (w_max, z_hip),
        (w_belt + (w_max - w_belt) * 0.35, z_belt - 26.0),
        (w_belt + 0.8, z_belt - 3.0),
        (w_belt - 2.0, z_belt + gh * 0.22),
        (w_belt - (w_belt - w_roof) * 0.42, z_belt + gh * 0.52),
        (w_roof + 2.0, z_belt + gh * 0.86),
        (w_roof - 4.0, z_belt + gh * 0.985),
        (w_roof * 0.72, z_roof + 1.2),
        (0.0, z_roof + 2.0),
    ]
    ctrl = np.asarray(ctrl, dtype=float)
    # keep the polyline strictly increasing in arc length
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))]
    keep = np.r_[True, np.diff(d) > 1e-6]
    ctrl, d = ctrl[keep], d[keep]
    t = np.linspace(0, d[-1], n_half)
    fy = PchipInterpolator(d, ctrl[:, 0])
    fz = PchipInterpolator(d, ctrl[:, 1])
    half = np.column_stack([fy(t), fz(t)])
    full = np.vstack([half, np.column_stack([-half[-2:0:-1, 0], half[-2:0:-1, 1]])])
    return np.column_stack([np.full(len(full), x), full[:, 0], full[:, 1]])


def build_body_shell():
    params = _interp_stations()
    xs = np.unique(np.concatenate([
        np.linspace(-450, -420, 9),
        np.linspace(-420, -330, 19)[1:],
        np.linspace(-330, 58, 30)[1:],
        np.linspace(58, 245, 34)[1:],
        np.linspace(245, 410, 18)[1:],
        np.linspace(410, 450, 11)[1:],
    ]))
    rings = [body_section(params, x) for x in xs]
    return loft(rings)


def _mono_y_of_z(pts):
    """PCHIP y(z) from a monotone greenhouse part of a section."""
    z, y = pts[:, 2], pts[:, 1]
    order = np.argsort(z)
    z, y = z[order], y[order]
    keep = np.r_[True, np.diff(z) > 1e-4]
    return PchipInterpolator(z[keep], y[keep]), z[keep].min(), z[keep].max()


def _mono_z_of_y(pts):
    y, z = pts[:, 1], pts[:, 2]
    order = np.argsort(y)
    y, z = y[order], z[order]
    keep = np.r_[True, np.diff(y) > 1e-4]
    return PchipInterpolator(y[keep], z[keep]), y[keep].min(), y[keep].max()


def side_pocket(params, xa, xb, f_lo, f_hi, depth=3.0, taper_front=None,
                taper_rear=None, nx=26, nz=9, n_half=84):
    """Pocket that follows the greenhouse side surface.

    f_lo/f_hi are heights as a fraction of the local greenhouse, either
    constants or callables of x.  taper_front/rear are (x_tip, slope) pairs
    describing a slanted pillar edge: the glass top is limited to
    z_tip + slope * (x_tip - x).
    """
    rings = []
    for x in np.linspace(xa, xb, nx):
        sec = body_section(params, x, n_half=n_half)[:n_half]
        z0, w_sill, w_max, z_hip, w_belt, z_belt, w_roof, z_roof = [float(f(x)) for f in params]
        z_roof = max(z_roof, z_belt + 1.2)
        gh = z_roof - z_belt
        if gh < 12.0:
            continue
        yz, zmin, zmax = _mono_y_of_z(sec[sec[:, 2] >= z_belt - 2.0])
        lo = z_belt + gh * (f_lo(x) if callable(f_lo) else f_lo)
        hi = z_belt + gh * (f_hi(x) if callable(f_hi) else f_hi)
        if taper_front is not None:
            xt, sl, zt = taper_front
            hi = min(hi, zt + sl * (xt - x))
        if taper_rear is not None:
            xt, sl, zt = taper_rear
            hi = min(hi, zt + sl * (x - xt))
        lo = max(lo, zmin + 0.4)
        hi = min(hi, zmax - 0.4)
        if hi - lo < 2.0:
            continue
        zz = np.linspace(lo, hi, nz)
        yy = yz(zz)
        inner = np.column_stack([np.full(nz, x), yy - depth, zz])
        outer = np.column_stack([np.full(nz, x), yy + 70.0, zz])
        rings.append(np.vstack([inner, outer[::-1]]))
    if len(rings) < 2:
        return None
    cut = loft(rings)
    mirrored = cut.copy()
    mirrored.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
    if mirrored.volume < 0:
        mirrored.invert()
    return [cut, mirrored]


def top_pocket(params, xa, xb, half_width, depth=3.0, nx=30, ny=15, n_half=84,
               end_taper=0.0):
    """Pocket that follows the upper surface (windshield / rear glass / roof)."""
    rings = []
    for i, x in enumerate(np.linspace(xa, xb, nx)):
        sec = body_section(params, x, n_half=n_half)[:n_half]
        z0, w_sill, w_max, z_hip, w_belt, z_belt, w_roof, z_roof = [float(f(x)) for f in params]
        z_roof = max(z_roof, z_belt + 1.2)
        gh = z_roof - z_belt
        zy, ymin, ymax = _mono_z_of_y(sec[sec[:, 2] >= z_belt + 0.55 * gh])
        t = i / (nx - 1)
        w = half_width(x) if callable(half_width) else half_width
        if end_taper > 0:
            e = min(t, 1 - t) / end_taper
            if e < 1:
                w *= 0.35 + 0.65 * (e * e * (3 - 2 * e))
        w = float(min(w, ymax - 0.5))
        if w < 4.0:
            continue
        yy = np.linspace(-w, w, ny)
        zz = zy(np.abs(yy))
        inner = np.column_stack([np.full(ny, x), yy, zz - depth])
        outer = np.column_stack([np.full(ny, x), yy, zz + 70.0])
        rings.append(np.vstack([inner, outer[::-1]]))
    if len(rings) < 2:
        return None
    return loft(rings)


def glazing_cutters():
    params = _interp_stations()
    cuts = []
    # side glass: front door, rear door, quarter light
    for xa, xb, tf, tr in (
        (10.0, 171.0, (190.0, 1.15, 218.0), None),
        (-168.0, -12.0, None, None),
        (-300.0, -190.0, None, (-309.0, 0.95, 222.0)),
    ):
        got = side_pocket(params, xa, xb, 0.10, 0.90, depth=3.2,
                          taper_front=tf, taper_rear=tr, nx=34, nz=12)
        if got:
            cuts += got
    # windshield and rear glass
    ws = top_pocket(params, 72.0, 236.0, lambda x: 104.0 + (x - 72.0) / 164.0 * 44.0,
                    depth=3.0, nx=52, ny=25, end_taper=0.08)
    if ws is not None:
        cuts.append(ws)
    rg = top_pocket(params, -428.0, -344.0, lambda x: 108.0 + (-344.0 - x) / 84.0 * 30.0,
                    depth=3.0, nx=34, ny=25, end_taper=0.08)
    if rg is not None:
        cuts.append(rg)
    # panoramic roof panel
    roof = top_pocket(params, -258.0, 34.0, 92.0, depth=2.2, end_taper=0.06)
    if roof is not None:
        cuts.append(roof)
    return cuts


def wheel_arch_cutters(radius=None, inner_y=96.0):
    r = PROPORTIONS['arch_radius'] if radius is None else radius
    out = []
    for ax in (PROPORTIONS['axle_front_x'], PROPORTIONS['axle_rear_x']):
        for side in (-1, 1):
            cyl = trimesh.creation.cylinder(radius=r, height=160.0, sections=72)
            cyl.apply_transform(rot_x(90))
            cyl.apply_translation([ax, side * (inner_y + 80.0), PROPORTIONS['tyre_radius']])
            out.append(cyl)
    return out


def arch_cladding():
    """Black-look arch lip: a thin band following each wheel opening."""
    parts = []
    r_in = PROPORTIONS['arch_radius']
    r_out = r_in + 7.0
    mid_r = (r_in + r_out) / 2.0
    for ax in (PROPORTIONS['axle_front_x'], PROPORTIONS['axle_rear_x']):
        for side in (-1, 1):
            a = np.radians(np.linspace(8.0, 172.0, 48))
            path = np.column_stack([
                ax + mid_r * np.cos(a),
                np.full_like(a, side * 171.0),
                PROPORTIONS['tyre_radius'] + mid_r * np.sin(a),
            ])
            band = sweep(path, rounded_section(28.0, r_out - r_in, r=2.0, n=5))
            parts.append(band)
    return parts


# --------------------------------------------------------------------------
# Wheels (built from scratch: revolved tyre + five paired spokes)
# --------------------------------------------------------------------------
def build_wheel(center, flip=False):
    R = PROPORTIONS['tyre_radius']
    W = PROPORTIONS['tyre_width']
    rim = PROPORTIONS['rim_radius']
    h = W / 2.0
    profile = [
        (rim - 1.0, -h + 2.0), (rim + 3.0, -h + 0.5), (R - 9.0, -h),
        (R - 2.5, -h + 3.0), (R, -h + 9.0), (R, h - 9.0),
        (R - 2.5, h - 3.0), (R - 9.0, h), (rim + 3.0, h - 0.5),
        (rim - 1.0, h - 2.0),
    ]
    tyre = revolve(profile, sections=96)
    # circumferential tread grooves
    grooves = []
    for off in (-14.0, -4.7, 4.7, 14.0):
        g = trimesh.creation.cylinder(radius=R + 3.0, height=3.0, sections=96)
        gi = trimesh.creation.cylinder(radius=R - 3.2, height=6.0, sections=96)
        ring = boolean('diff', [g, gi])
        ring.apply_translation([0, 0, off])
        grooves.append(ring)
    tyre = boolean('diff', [tyre] + grooves)

    barrel = revolve([(rim - 6.0, -h + 3.0), (rim, -h + 3.0), (rim, h - 1.0),
                      (rim - 3.0, h - 1.0), (rim - 3.0, -h + 6.0), (rim - 6.0, -h + 6.0)],
                     sections=72)
    lip = revolve([(rim - 4.0, h - 5.0), (rim + 0.5, h - 5.0), (rim + 0.5, h - 1.0),
                   (rim - 4.0, h - 1.0)], sections=72)
    disc = trimesh.creation.cylinder(radius=rim - 4.5, height=6.0, sections=48)
    disc.apply_translation([0, 0, h - 20.0])
    hat = trimesh.creation.cylinder(radius=15.0, height=22.0, sections=32)
    hat.apply_translation([0, 0, h - 12.0])
    hub = trimesh.creation.cylinder(radius=13.0, height=12.0, sections=36)
    hub.apply_translation([0, 0, h - 9.0])
    cap = trimesh.creation.cylinder(radius=8.5, height=4.0, sections=32)
    cap.apply_translation([0, 0, h - 2.0])

    spokes = []
    face_z = h - 5.0
    for k in range(5):
        base = np.radians(k * 72.0 + 12.0)
        for splay in (-1, 1):
            a0 = base + splay * np.radians(5.0)
            a1 = base + splay * np.radians(21.0)
            p0 = np.array([13.0 * np.cos(a0), 13.0 * np.sin(a0)])
            p1 = np.array([(rim - 2.0) * np.cos(a1), (rim - 2.0) * np.sin(a1)])
            d = p1 - p0
            L = np.linalg.norm(d)
            ang = np.degrees(np.arctan2(d[1], d[0]))
            sp = rounded_box([L + 6.0, 9.0, 9.0], 2.2)
            sp.apply_transform(rot_z(ang))
            mid = (p0 + p1) / 2.0
            sp.apply_translation([mid[0], mid[1], face_z - 3.0])
            spokes.append(sp)
    wheel = boolean("union", [tyre, barrel, lip, disc, hat, hub, cap] + spokes)
    wheel.apply_transform(rot_x(90 if not flip else -90))
    wheel.apply_translation(center)
    return wheel


def build_wheels():
    out = []
    ty = PROPORTIONS['track_half']
    tz = PROPORTIONS['tyre_radius']
    for ax in (PROPORTIONS['axle_front_x'], PROPORTIONS['axle_rear_x']):
        for side in (-1, 1):
            out.append(build_wheel([ax, side * ty, tz], flip=side > 0))
    return out


# --------------------------------------------------------------------------
# Surface queries used to sit details on the skin
# --------------------------------------------------------------------------
def side_y(params, x, z, n_half=96):
    sec = body_section(params, x, n_half=n_half)[:n_half]
    f, zmin, zmax = _mono_y_of_z(sec)
    return float(f(np.clip(z, zmin + 0.2, zmax - 0.2)))


def roof_z(params, x, y, n_half=96):
    sec = body_section(params, x, n_half=n_half)[:n_half]
    z0, w_sill, w_max, z_hip, w_belt, z_belt, w_roof, z_roof = [float(f(x)) for f in params]
    gh = max(z_roof, z_belt + 1.2) - z_belt
    f, ymin, ymax = _mono_z_of_y(sec[sec[:, 2] >= z_belt + 0.5 * gh])
    return float(f(np.clip(abs(y), ymin, ymax - 0.2)))


def groove_along_side(params, xa, xb, z_of_x, depth=1.8, width=3.0, nx=60):
    """Shallow channel following the flank (character line / shut lines)."""
    out = []
    xs = np.linspace(xa, xb, nx)
    for side in (-1, 1):
        path = np.array([[x, side * (side_y(params, x, z_of_x(x)) + depth * 0.55), z_of_x(x)]
                         for x in xs])
        out.append(sweep(path, rounded_section(depth * 3.0, width, r=width * 0.35, n=4)))
    return out


def groove_vertical(params, x_of_z, za, zb, depth=1.6, width=2.6, nz=30):
    out = []
    zs = np.linspace(za, zb, nz)
    for side in (-1, 1):
        path = np.array([[x_of_z(z), side * (side_y(params, x_of_z(z), z) + depth * 0.55), z]
                         for z in zs])
        out.append(sweep(path, rounded_section(depth * 3.0, width, r=width * 0.35, n=4),
                         up=(1, 0, 0)))
    return out


def lattice_bars(outline_yz, spacing, thickness, x_center, depth, span=260.0):
    """Diamond mesh trimmed in 2-D, then extruded along X.

    Trimming with shapely instead of a 3-D clip avoids the coplanar slivers a
    boolean clip leaves where many bars end on the same plane.
    """
    from shapely.geometry import LineString
    poly = Polygon(outline_yz)
    if not poly.is_valid:
        poly = poly.buffer(0)
    ylo, zlo, yhi, zhi = poly.bounds
    bars = []
    n = int((yhi - ylo + 2 * (zhi - zlo)) / spacing) + 2
    for fam, sign in ((0, 1.0), (1, -1.0)):
        xc = x_center + fam * 0.45
        for k in range(-n, n + 1):
            c = (ylo + yhi) / 2.0 + k * spacing
            line = LineString([(c - span, zlo - span * sign), (c + span, zhi + (zhi - zlo) + span * sign)])
            piece = line.buffer(thickness / 2.0, cap_style=2).intersection(poly)
            if piece.is_empty:
                continue
            geoms = getattr(piece, 'geoms', [piece])
            for g in geoms:
                if g.area < 0.4:
                    continue
                pts = np.asarray(g.exterior.coords)[:-1]
                bars.append(x_cut(pts, xc - depth / 2.0, xc + depth / 2.0))
    return bars


def x_cut(points_yz, x_floor, x_out=520.0):
    """Prism along X from a YZ outline (used on the flat nose / tailgate)."""
    poly = Polygon(points_yz)
    if not poly.is_valid:
        poly = poly.buffer(0)
    m = trimesh.creation.extrude_polygon(poly, abs(x_out - x_floor), engine=ENGINE)
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 2] + min(x_floor, x_out), v[:, 0], v[:, 1]])
    m.fix_normals()
    if m.volume < 0:
        m.invert()
    return m


# --------------------------------------------------------------------------
# Front fascia
# --------------------------------------------------------------------------
def front_fasia_cutters():
    cuts = [
        x_cut([(-166, 152), (166, 152), (168, 191), (-168, 191)], 444.5),
        x_cut([(-104, 70), (104, 70), (120, 126), (-120, 126)], 444.0),
        x_cut([(-52, 96), (52, 96), (52, 130), (-52, 130)], 447.0),
    ]
    for s in (-1, 1):
        cuts.append(x_cut([(s * 126, 72), (s * 160, 78), (s * 158, 112), (s * 128, 116)], 444.0))
    return cuts


def front_fasia_addons():
    adds = []
    adds += lattice_bars([(-111, 154.5), (111, 154.5), (115, 188.5), (-115, 188.5)],
                         spacing=13.0, thickness=2.6, x_center=446.2, depth=6.0, span=60.0)
    adds.append(place(rounded_box([4.0, 300.0, 4.0], 1.4), [449.0, 0, 190.0]))
    for s in (-1, 1):
        for k in range(2):
            lens = trimesh.creation.cylinder(radius=7.5, height=9.0, sections=28)
            lens.apply_transform(rot_y(90))
            adds.append(place(lens, [444.5, s * (124.0 + k * 20.0), 170.0]))
        adds.append(place(rounded_box([8.0, 54.0, 4.0], 1.4), [445.0, s * 136.0, 184.0]))
        adds.append(place(rounded_box([8.0, 46.0, 3.0], 1.2), [445.0, s * 134.0, 158.0]))
    badge = trimesh.creation.cylinder(radius=15.0, height=8.0, sections=48)
    badge.apply_transform(rot_y(90))
    adds.append(place(badge, [448.0, 0, 168.0]))
    ring = boolean('diff', [trimesh.creation.cylinder(radius=15.5, height=3.0, sections=48),
                            trimesh.creation.cylinder(radius=12.5, height=6.0, sections=48)])
    ring.apply_transform(rot_y(90))
    adds.append(place(ring, [450.0, 0, 168.0]))
    adds.append(place(rounded_box([6.0, 210.0, 16.0], 4.0), [448.0, 0, 56.0]))
    adds.append(place(rounded_box([12.0, 286.0, 14.0], 4.0), [444.0, 0, 46.0]))
    adds += lattice_bars([(-101, 72.5), (101, 72.5), (116, 123.5), (-116, 123.5)],
                         spacing=15.0, thickness=2.4, x_center=445.8, depth=6.0, span=70.0)
    return adds


# --------------------------------------------------------------------------
# Rear fascia
# --------------------------------------------------------------------------
def rear_cutters():
    cuts = [
        x_cut([(-58, 108), (58, 108), (58, 148), (-58, 148)], -452.0, -445.0),
        x_cut([(-108, 46), (108, 46), (98, 74), (-98, 74)], -452.0, -443.0),
    ]
    for s in (-1, 1):
        cuts.append(x_cut([(s * 17.0, 202), (s * 162.0, 199), (s * 162.0, 219), (s * 17.0, 216)],
                          -452.0, -444.0))
        cuts.append(x_cut([(s * 104.0, 186), (s * 164.0, 184), (s * 164.0, 222), (s * 104.0, 224)],
                          -452.0, -443.0))
        cuts.append(x_cut([(s * 112, 50), (s * 150, 54), (s * 148, 72), (s * 112, 70)],
                          -452.0, -442.0))
    return cuts


def rear_addons():
    adds = []
    for s in (-1, 1):
        adds.append(place(rounded_box([8.0, 143.0, 6.0], 2.0), [-447.0, s * 89.0, 209.0]))
        for k in range(3):
            adds.append(place(rounded_box([8.0, 14.0, 22.0], 3.0),
                              [-446.0, s * (116.0 + k * 17.0), 203.0]))
        adds.append(place(rounded_box([10.0, 34.0, 14.0], 4.0), [-445.0, s * 130.0, 61.0]))
    badge = trimesh.creation.cylinder(radius=14.0, height=5.0, sections=48)
    badge.apply_transform(rot_y(90))
    adds.append(place(badge, [-449.0, 0, 208.0]))
    for k in range(-3, 4):
        adds.append(place(rounded_box([9.0, 5.0, 22.0], 1.5), [-445.0, k * 26.0, 60.0]))
    adds.append(place(rounded_box([7.0, 250.0, 10.0], 3.0), [-447.0, 0, 92.0]))
    params = _interp_stations()
    path = np.array([[-348.0 - 0.05 * abs(y), y, roof_z(params, -348.0, y) + 2.0]
                     for y in np.linspace(-132, 132, 29)])
    adds.append(sweep(path, rounded_section(26.0, 7.0, r=2.5, n=5)))
    adds.append(place(rounded_box([3.0, 96.0, 4.0], 1.5), [-424.0, -18.0, 246.0], rot_y(-38)))
    adds.append(trimesh.convex.convex_hull(np.array([
        [-268, -7, 312], [-236, -7, 312], [-262, -7, 334],
        [-268, 7, 312], [-236, 7, 312], [-262, 7, 334],
    ], dtype=float)))
    return adds


# --------------------------------------------------------------------------
# Flanks and roof rails
# --------------------------------------------------------------------------
def flank_cutters():
    params = _interp_stations()
    cuts = []
    cuts += groove_along_side(params, -412.0, 414.0, lambda x: 152.0 + 0.008 * x,
                              depth=2.0, width=3.2, nx=70)
    cuts += groove_along_side(params, -300.0, 300.0, lambda x: 74.0,
                              depth=1.6, width=2.6, nx=50)
    for xz in (lambda z: 178.0 - (z - 60.0) * 0.06,
               lambda z: 2.0,
               lambda z: -178.0 + (z - 60.0) * 0.03):
        cuts += groove_vertical(params, xz, 48.0, 214.0)
    flap = rounded_box([30.0, 8.0, 26.0], 6.0)
    flap.apply_translation([-320.0, -(side_y(params, -320.0, 172.0) + 2.4), 172.0])
    cuts.append(flap)
    for s in (-1, 1):
        xs = np.linspace(258.0, 432.0, 30)
        path = np.array([[x, s * 58.0, roof_z(params, x, 58.0) + 1.0] for x in xs])
        cuts.append(sweep(path, rounded_section(4.0, 2.6, r=0.9, n=4)))
        xs = np.linspace(252.0, 436.0, 30)
        path = np.array([[x, s * (float(params[6](x)) - 6.0),
                          roof_z(params, x, float(params[6](x)) - 6.0) + 1.0] for x in xs])
        cuts.append(sweep(path, rounded_section(3.4, 2.4, r=0.8, n=4)))
    return cuts


def flank_addons():
    params = _interp_stations()
    adds = []
    for s in (-1, 1):
        for hx in (96.0, -86.0):
            y = side_y(params, hx, 190.0)
            adds.append(place(rounded_box([34.0, 7.0, 9.0], 3.0), [hx, s * (y + 1.5), 190.0]))
        base_y = side_y(params, 176.0, 222.0)
        adds.append(place(rounded_box([16.0, 20.0, 9.0], 3.0),
                          [176.0, s * (base_y + 7.0), 226.0], rot_z(s * 12)))
        adds.append(trimesh.convex.convex_hull(np.array([
            [164, s * (base_y + 8), 224], [198, s * (base_y + 8), 225],
            [164, s * (base_y + 8), 240], [198, s * (base_y + 8), 241],
            [158, s * (base_y + 28), 226], [192, s * (base_y + 30), 228],
            [160, s * (base_y + 28), 245], [190, s * (base_y + 30), 246],
            [166, s * (base_y + 32), 230], [188, s * (base_y + 33), 232],
            [167, s * (base_y + 32), 242], [187, s * (base_y + 33), 243],
        ], dtype=float)))
        adds.append(place(rounded_box([26.0, 4.0, 13.0], 2.0),
                          [176.0, s * (base_y + 30.0), 236.0]))
        # straight sill strip between the arches; a prism crosses the flank
        # transversally, where a surface-hugging sweep would graze it and
        # leave sliver triangles in the union
        adds.append(place(rounded_box([392.0, 16.0, 26.0], 4.0), [0.0, s * 168.0, 56.0]))
    return adds


def roof_rails():
    params = _interp_stations()
    adds = []
    xs = np.linspace(-282.0, 36.0, 55)
    t = (xs + 282.0) / 318.0
    end = np.clip(np.minimum(t, 1 - t) / 0.10, 0, 1)
    end = end * end * (3 - 2 * end)
    for s in (-1, 1):
        y = s * 112.0
        base = np.array([roof_z(params, x, y) for x in xs])
        ped_h = 2.2 + 6.0 * end
        ped = np.column_stack([xs, np.full_like(xs, y), base + ped_h / 2.0 - 0.8])
        ped_secs = np.array([rounded_section(7.0, h + 1.6, r=min(1.6, h * 0.4), n=5)
                             for h in ped_h])
        adds.append(sweep(ped, ped_secs))
        cap = np.column_stack([xs, np.full_like(xs, y), base + ped_h + 1.4])
        cap_secs = np.array([rounded_section(6.0 + 3.0 * e, 4.6, r=2.0, n=6) for e in end])
        adds.append(sweep(cap, cap_secs))
    return adds


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build(parts='body'):
    body = build_body_shell()
    body = boolean('diff', [body] + wheel_arch_cutters())
    body = boolean('union', [body] + arch_cladding())
    body = boolean('diff', [body] + glazing_cutters())
    body = boolean('diff', [body] + front_fasia_cutters() + rear_cutters() + flank_cutters())
    body = boolean('union', [body] + front_fasia_addons() + rear_addons()
                   + flank_addons() + roof_rails())
    wheels = build_wheels()
    return body, wheels


def finalize(mesh):
    """Make the mesh survive the float32 round-trip through binary STL.

    Booleans leave slivers whose vertices collapse when written as float32;
    weld at 1e-4, drop the collapsed triangles and close the pinholes.
    """
    # Round-trip through binary STL first: float32 storage is what the user
    # loads, so cleanup must happen on exactly those coordinates.
    m = trimesh.load(trimesh.util.wrap_as_stream(mesh.export(file_type='stl')),
                     file_type='stl', process=True)
    # Weld and drop any remaining slivers.
    m.vertices = np.asarray(m.vertices, dtype=np.float32).astype(np.float64)
    m.merge_vertices(digits_vertex=4)
    m.update_faces(m.nondegenerate_faces(height=1e-4))
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(m)
    groups = trimesh.graph.connected_components(
        m.face_adjacency, min_len=12, nodes=np.arange(len(m.faces)))
    keep = np.zeros(len(m.faces), dtype=bool)
    for g in groups:
        keep[g] = True
    m.update_faces(keep)
    m.remove_unreferenced_vertices()
    m.fix_normals()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tayron scratch.stl')
    ap.add_argument('--report', default='reports/scratch_build.json')
    args = ap.parse_args()
    body, wheels = build()
    mesh = finalize(trimesh.util.concatenate([body] + wheels))
    mesh.export(args.out)
    info = {
        'source': 'procedural, no source STL read',
        'proportions': PROPORTIONS,
        'faces': len(mesh.faces),
        'watertight': bool(mesh.is_watertight),
        'components': len(mesh.split(only_watertight=False)),
        'extents': mesh.extents.tolist(),
    }
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w') as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: v for k, v in info.items() if k != 'proportions'}, indent=2))


if __name__ == '__main__':
    main()
