/* «🔥 Прожечь»: канал в кузове выжигается ровно по сечению каната, самого
   каната в сцене не остаётся. Настоящий three, настоящий sculpt.js, кузов из
   body.glb. Проверяем: глубина = радиус, ширина = 2R, стенка = полукруг,
   кожа вне канала не сдвинута ни на единицу, сетка замкнута, отмена цела.

   Запуск: sh tests/run_all.sh   (или node tests/burn_cut.mjs после mkharness.sh) */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from '../suspension/vendor/three.module.js';
import { initSculpt } from './.harness/sculpt_real.mjs';
import { buildTopology, checkMesh } from '../suspension/meshops.js';
import { cleanNormals } from '../suspension/ropes.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const U = 5.8824;
const HTML = fs.readFileSync(path.join(HERE, '../suspension/index.html'), 'utf8');
const raw = JSON.parse(fs.readFileSync(path.join(HERE, 'body.json'), 'utf8'));

/* ---------- стенд ---------- */
const els = {};
const mk = (id) => {
  const e = { id, value: '', style: {}, hidden: false, files: [], checked: false, _html: '', _text: '',
    classList: { s: new Set(), add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
      toggle(c, v) { v ? this.s.add(c) : this.s.delete(c); }, contains(c) { return this.s.has(c); } },
    addEventListener(t, f) { ((this._ev ||= {})[t] ||= []).push(f); },
    fire(t, e = {}) { ((this._ev || {})[t] || []).forEach(f => f(e)); },
    click() { this.fire('click'); } };
  Object.defineProperty(e, 'innerHTML', { get() { return this._html; }, set(v) { this._html = v; this._text = ''; } });
  Object.defineProperty(e, 'textContent', { get() { return this._text || this._html.replace(/<[^>]+>/g, ''); }, set(v) { this._text = v || ''; this._html = ''; } });
  return (els[id] = e);
};
[...HTML.matchAll(/id="([a-z0-9-]+)"/g)].map(m => m[1]).forEach(mk);
els['sc-brush'].value = '1.2'; els['sc-depth'].value = '0.6'; els['sc-amt'].value = '25';
const scRows = Object.keys(els).filter(id => id.endsWith('row')).map(id => els[id]);
const canvas = { style: {}, _ev: {}, addEventListener(t, f) { (this._ev[t] ||= []).push(f); },
  setPointerCapture() {}, releasePointerCapture() {}, hasPointerCapture() { return true; } };
globalThis.document = { getElementById: (id) => els[id] || null,
  querySelectorAll: (s) => (s === '.sc-row' ? scRows : []), body: { appendChild() {} },
  createElement: () => ({ style: {}, click() {}, remove() {}, setAttribute() {} }) };
globalThis.innerWidth = 1200; globalThis.innerHeight = 800;
globalThis.URL = { createObjectURL: () => 'blob:x', revokeObjectURL() {} };
globalThis.Blob = class {};
const keys = [];
globalThis.addEventListener = (t, f) => { if (t === 'keydown') keys.push(f); };
globalThis.removeEventListener = () => {};

const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(raw.pos.flat()), 3));
geo.setIndex(new THREE.BufferAttribute(Uint32Array.from(raw.idx), 1));
geo.computeVertexNormals();
const material = new THREE.MeshStandardMaterial({ color: 0xb9c6d4 });
const mesh = new THREE.Mesh(geo, material);
const scene = new THREE.Scene(); scene.add(mesh); scene.updateMatrixWorld(true);
const camera = new THREE.PerspectiveCamera(36, innerWidth / innerHeight, 5, 9000);
camera.position.set(760, 250, 640); camera.lookAt(220, 40, 120); camera.updateMatrixWorld(true);
const sculpt = initSculpt({ mesh, material, camera, renderer: { domElement: canvas, localClippingEnabled: false },
  controls: { target: new THREE.Vector3(0, 125, 0) }, U });

/* ---------- помощники ---------- */
let n = 0, bad = 0;
const it = (name, cond, note = '') => { if (!cond) bad++; n++;
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${note ? ' — ' + note : ''}`); };
const info = () => (els['sc-info'].innerHTML || els['sc-info'].textContent || '').replace(/<[^>]+>/g, '');
const P = () => geo.attributes.position.array;
const I = () => geo.index.array;
const f0 = I().length / 3;
const chk = () => checkMesh(buildTopology(Float32Array.from(P()), Uint32Array.from(I())),
  Uint32Array.from(I()), Float32Array.from(P()), U);
const snap = () => Float32Array.from(P());
const maxDiff = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > m) m = d; } return m; };
const ev = (x, y, shift = false) => ({ button: 0, clientX: x, clientY: y, pointerId: 1, shiftKey: shift, preventDefault() {} });
const scr = (x, y, z) => { const v = new THREE.Vector3(x, y, z); mesh.localToWorld(v); v.project(camera);
  return [(v.x * 0.5 + 0.5) * innerWidth, (-v.y * 0.5 + 0.5) * innerHeight]; };
const down = (x, z) => { const r = new THREE.Raycaster(new THREE.Vector3(x, 600, z), new THREE.Vector3(0, -1, 0), 0, 4000);
  const h = r.intersectObject(mesh, false); if (!h.length) throw new Error('мимо кузова: ' + x + ',' + z); return h[0].point.toArray(); };
const strokeSurface = (x0, z0, x1, z1, cnt = 20, stopAtEdge = false) => {
  const pts = [];
  for (let i = 0; i <= cnt; i++) {
    const t = i / cnt;
    let p;
    try { p = down(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t); } catch (e) { if (stopAtEdge) break; throw e; }
    pts.push(p);
  }
  if (pts.length < 3) throw new Error('штрих не лёг на кузов');
  const sp = pts.map(p => scr(...p));
  (canvas._ev.pointerdown || []).forEach(f => f(ev(sp[0][0], sp[0][1])));
  for (let i = 1; i < sp.length - 1; i++) (canvas._ev.pointermove || []).forEach(f => f(ev(sp[i][0], sp[i][1])));
  (canvas._ev.pointerup || []).forEach(f => f(ev(sp[sp.length - 1][0], sp[sp.length - 1][1])));
};
const tool = (t) => { sculpt.setTool(null); if (t) sculpt.setTool(t); };
const reset = () => {
  els['sc-clear'].click(); sculpt.setTool(null);
  while (sculpt.ropeCount) sculpt.removeRope();      // канаты прошлых блоков стенда не мешают
};
const key = (k, mods = {}) => keys.forEach(f => f({ key: k, target: { tagName: 'BODY' }, ctrlKey: false, metaKey: false,
  altKey: false, shiftKey: false, preventDefault() {}, ...mods }));

/* объём замкнутой сетки (теорема о дивергенции) и «закрытость» */
const volume = (pos, idx) => {
  let v = 0;
  for (let i = 0; i < idx.length; i += 3) {
    const a = idx[i] * 3, b = idx[i + 1] * 3, c = idx[i + 2] * 3;
    v += (pos[a] * (pos[b + 1] * pos[c + 2] - pos[b + 2] * pos[c + 1])
        - pos[a + 1] * (pos[b] * pos[c + 2] - pos[b + 2] * pos[c])
        + pos[a + 2] * (pos[b] * pos[c + 1] - pos[b + 1] * pos[c])) / 6;
  }
  return v;
};
const badEdges = (idx) => {
  const m = new Map();
  for (let i = 0; i < idx.length; i += 3) {
    const tr = [idx[i], idx[i + 1], idx[i + 2]];
    for (let k = 0; k < 3; k++) {
      const a = tr[k], b = tr[(k + 1) % 3];
      const key = a < b ? a + ':' + b : b + ':' + a;
      m.set(key, (m.get(key) || 0) + 1);
    }
  }
  let b = 0; for (const v of m.values()) if (v !== 2) b++;
  return b;
};


/* ---------- профиль пути: (t, h) узла относительно каната ---------- */
const pathProj = (pts, p) => {
  let best = Infinity, bi = 0, bt = 0;
  for (let i = 0; i + 1 < pts.length; i++) {
    const a = pts[i], b = pts[i + 1];
    const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const L2 = dx * dx + dy * dy + dz * dz;
    let t = L2 > 1e-12 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy + (p[2] - a[2]) * dz) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const qx = p[0] - (a[0] + dx * t), qy = p[1] - (a[1] + dy * t), qz = p[2] - (a[2] + dz * t);
    const d2 = qx * qx + qy * qy + qz * qz;
    if (d2 < best) { best = d2; bi = i; bt = t; }
  }
  /* за торцом каната? */
  let beyond = 0;
  const A0 = pts[0], B0 = pts[1], A1 = pts[pts.length - 2], B1 = pts[pts.length - 1];
  if (bi === 0 && bt <= 1e-9) {
    const q = [p[0] - A0[0], p[1] - A0[1], p[2] - A0[2]];
    if (q[0] * (B0[0] - A0[0]) + q[1] * (B0[1] - A0[1]) + q[2] * (B0[2] - A0[2]) < 0) beyond = -1;
  } else if (bi === pts.length - 2 && bt >= 1 - 1e-9) {
    const q = [p[0] - B1[0], p[1] - B1[1], p[2] - B1[2]];
    if (q[0] * (B1[0] - A1[0]) + q[1] * (B1[1] - A1[1]) + q[2] * (B1[2] - A1[2]) > 0) beyond = 1;
  }
  return { seg: bi, t: bt, lat: Math.sqrt(best), beyond };
};
/* расстояние узла от оси каната В ПЛОСКОСТИ КОЖИ (без высоты над ней) */
const inPlane = (p) => {
  const pr = pathProj(rpath, p);
  const A = rpath[pr.seg], B = rpath[pr.seg + 1], u = pr.t;
  const n0 = nrms[pr.seg] || [0, 1, 0], n1 = nrms[pr.seg + 1] || n0;
  let nx = n0[0] + (n1[0] - n0[0]) * u, ny = n0[1] + (n1[1] - n0[1]) * u, nz = n0[2] + (n1[2] - n0[2]) * u;
  const nl = Math.hypot(nx, ny, nz) || 1; nx /= nl; ny /= nl; nz /= nl;
  const ax = A[0] + (B[0] - A[0]) * u, ay = A[1] + (B[1] - A[1]) * u, az = A[2] + (B[2] - A[2]) * u;
  const rx = p[0] - ax, ry = p[1] - ay, rz = p[2] - az;
  const dn = rx * nx + ry * ny + rz * nz;
  return { t: Math.hypot(rx - nx * dn, ry - ny * dn, rz - nz * dn), dn, beyond: pr.beyond };
};

/* ---------- 1. канат по кузову, «прожечь» ---------- */
reset();
tool('rope');
strokeSurface(150, 160, 400, 160, 26);
const rp = sculpt.ropeList()[0];
const R = rp.radius, L = rp.len;
const rpath = sculpt.ropePath(0).slice();
const nrms = cleanNormals(sculpt.ropePath(0), sculpt.ropeNormals(0), 3, 35);
const tubeObj = sculpt.ropeMesh(0);
const before = snap();
const volBefore = volume(P(), I());
const fBefore = I().length / 3;
const c0 = chk();
it('кузов до прожига замкнут', c0.openEdges === 0 && c0.nonManifold === 0 && c0.degenerate === 0,
  `${c0.openEdges}/${c0.nonManifold}/${c0.degenerate}, граней ${fBefore}`);

const t0 = Date.now();
const res = sculpt.burnRope(rp.i);
const ms = Date.now() - t0;
it('прожиг выполнен', res && res.mode === 'cut', JSON.stringify(res));
it('быстро (< 15 с)', ms < 15000, `${ms} мс`);

/* ---------- 2. глубина и ширина = сечение каната ---------- */
const moved = [];
for (let v = 0; v < before.length / 3; v++) {
  const o = v * 3;
  const d = Math.hypot(P()[o] - before[o], P()[o + 1] - before[o + 1], P()[o + 2] - before[o + 2]);
  if (d > 1e-9) moved.push({ v, d, p: [before[o], before[o + 1], before[o + 2]] });
}
let maxD = 0, worst = 0, outside = 0, maxOutside = 0;
for (const m of moved) {
  const q = inPlane(m.p);
  maxD = Math.max(maxD, m.d);
  if (q.beyond || q.t >= R) { outside++; maxOutside = Math.max(maxOutside, m.d); continue; }
  const want = R * Math.sqrt(1 - (q.t / R) ** 2);         // полуцилиндр каната
  worst = Math.max(worst, Math.abs(m.d - want));
}
it('глубина канала = радиус каната', Math.abs(maxD - R) / U < 0.02,
  `${(maxD / U).toFixed(2)} мм при R = ${(R / U).toFixed(2)} мм`);
it('стенка канала = полукруг каната', worst / U < 0.05,
  `макс. отклонение ${(worst / U).toFixed(3)} мм по ${moved.length} узлам`);
it('вне сечения каната узлы не сдвинуты', outside === 0,
  `сдвинуто вне полосы ${outside}, макс ${(maxOutside / U).toFixed(2)} мм`);
it('ширина канала = 2 радиуса', Math.abs(res.width - (R * 2) / U) < 0.01,
  `${res.width.toFixed(2)} мм`);

/* кожа дальше 1.4 R от каната не двигается вовсе */
let farMoved = 0;
for (const m of moved) { const q = inPlane(m.p); if (q.lat > 1.4 * R) farMoved++; }
it('кожа вокруг канала не тронута', farMoved === 0, `сдвинуто дальше 1.4 R: ${farMoved}`);

/* ---------- 3. каната больше нет ---------- */
const inScene = (mesh.parent && mesh.parent.children || []).includes(tubeObj);
it('каната в сцене нет', sculpt.ropeCount === 0 && !inScene,
  `канатов ${sculpt.ropeCount}, труба в сцене ${inScene}`);
it('канат помечен впаянным', sculpt.ropeAbsorbed === 1, `впаяно ${sculpt.ropeAbsorbed}`);

/* ---------- 4. сетка цела, узлы добавились ---------- */
const c1 = chk();
it('после прожига сетка замкнута', c1.openEdges === 0 && c1.nonManifold === 0 && c1.degenerate === 0,
  `${c1.openEdges}/${c1.nonManifold}/${c1.degenerate} дыр/неманифолд/вырожденных, watertight ${c1.watertight}`);
const f1 = I().length / 3, v1 = P().length / 3;
it('в полосе каната сетка сгущена', f1 > fBefore + 2000 && res.added > 0,
  `граней ${fBefore} → ${f1} (+${f1 - fBefore}), вершин ${v1}, добавлено ${res.added}`);
let nan = 0;
for (let i = 0; i < P().length; i++) if (!Number.isFinite(P()[i])) nan++;
it('координаты без NaN', nan === 0, `NaN ${nan}`);

/* ---------- 5. объём: минус полуцилиндр каната ---------- */
const volAfter = volume(P(), I());
const lost = (volBefore - volAfter) / U ** 3;
const wantMm3 = Math.PI * (R / U) ** 2 / 2 * (L / U);
it('объём уменьшился на полуцилиндр каната', Math.abs(lost - wantMm3) / wantMm3 < 0.25,
  `−${lost.toFixed(1)} мм³ при ожидаемых ${wantMm3.toFixed(1)} мм³ (${((lost / wantMm3) * 100).toFixed(0)} %)`);

/* ---------- 6. далеко от каната сетка та же ---------- */
let farDiff = 0;
for (let v = 0; v < before.length / 3; v++) {
  const o = v * 3;
  if (pathProj(rpath, [before[o], before[o + 1], before[o + 2]]).lat > 3 * R) {
    farDiff = Math.max(farDiff, Math.abs(P()[o] - before[o]), Math.abs(P()[o + 1] - before[o + 1]),
      Math.abs(P()[o + 2] - before[o + 2]));
  }
}
it('дальше 3 R от каната кузов не изменился', farDiff === 0, `макс ${farDiff}`);

/* ---------- 7. повторный прожиг нечего делать ---------- */
const again = sculpt.burnRope(rp.i);
it('повторный прожиг отклонён', again === null && /впаян|прожечь/i.test(info()), info().slice(0, 90));

/* ---------- 8. отмена возвращает и кузов, и канат ---------- */
key('z', { ctrlKey: true });
const restored = maxDiff(before, P());
it('отмена вернула кузов', restored === 0 && I().length / 3 === fBefore,
  `макс сдвиг ${restored}, граней ${I().length / 3}`);
it('отмена вернула канат', sculpt.ropeCount === 1 && sculpt.ropeAbsorbed === 0,
  `канатов ${sculpt.ropeCount}, впаяно ${sculpt.ropeAbsorbed}`);

/* ---------- 9. прожиг без каната ---------- */
reset();
const none = sculpt.burnRope();
it('без каната прожиг отклонён', none === null, info().slice(0, 90));

console.log(`\n${bad ? 'FAIL' : 'ok'}: ${n - bad}/${n} проверок`);
if (bad) process.exit(1);
