/* «🔥 Прожечь»: канат ВСТРАИВАЕТСЯ в кузов. Грани кожи внутри следа удаляются,
   на их место встают грани самого каната (полуцилиндр R + торцы), между кромкой
   кожи и каналом зашивается лента. Координаты кожи не меняются ни на единицу.
   Настоящий three, настоящий sculpt.js, кузов из body.glb.

   Профиль канала проверяется по индексам рядов (станция × шаг дуги), а не
   «на глаз»: каждая точка ряда обязана лежать на дуге радиуса R между точками
   кромки, а самый глубокий ряд — ровно на R ниже хорды.

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




/* ---------- 1. канат по кузову, «прожечь» ---------- */
reset();
tool('rope');
strokeSurface(280 - 110, 160, 280 + 110, 160, 30);
const rp = sculpt.ropeList()[0];
const R = rp.radius, L = rp.len;
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
it('прожиг выполнен', !!(res && res.mode === 'cut'), res ? `${res.stations} станций, кожа ${res.removed} граней` : 'отказ');
if (!res) { console.log(`\nИТОГ: прожиг не выполнен — ${n - bad} из ${n}`); process.exit(1); }
it('быстро (< 25 с)', ms < 25000, `${ms} мс`);

/* ---------- 2. кожа не сдвинута ---------- */
let moved = 0, maxMove = 0;
for (let v = 0; v < before.length / 3; v++) {
  const o = v * 3;
  const d = Math.hypot(P()[o] - before[o], P()[o + 1] - before[o + 1], P()[o + 2] - before[o + 2]);
  if (d > 1e-9) { moved++; if (d > maxMove) maxMove = d; }
}
it('ни один узел кожи не сдвинут', moved === 0, `сдвинуто ${moved}, макс ${(maxMove / U).toFixed(4)} мм`);

/* ---------- 3. профиль канала = полукруг каната ----------
   Узлы ряда: кромка слева vertsFrom+i, кромка справа vertsFrom+M+i,
   точки дуги wallFrom + i*(K−1) + (j−1) для j = 1..K−1. */
const Pp = P();
const { M, K, vertsFrom, wallFrom } = res;
let profErr = 0, depth = 0, widthErr = 0, rows = 0;
for (let i = 0; i < M; i++) {
  const a = (vertsFrom + i) * 3, b = (vertsFrom + M + i) * 3;
  const ax = Pp[a], ay = Pp[a + 1], az = Pp[a + 2];
  const bx = Pp[b], by = Pp[b + 1], bz = Pp[b + 2];
  const w = Math.hypot(bx - ax, by - ay, bz - az);
  if (Math.abs(w - 2 * R) / (2 * R) > widthErr) widthErr = Math.abs(w - 2 * R) / (2 * R);
  for (let j = 1; j < K; j++) {
    const th = Math.PI * j / K, lam = (1 - Math.cos(th)) / 2, want = R * Math.sin(th);
    const o = (wallFrom + i * (K - 1) + (j - 1)) * 3;
    const cx = ax + (bx - ax) * lam, cy = ay + (by - ay) * lam, cz = az + (bz - az) * lam;
    const d = Math.hypot(Pp[o] - cx, Pp[o + 1] - cy, Pp[o + 2] - cz);
    if (Math.abs(d - want) > profErr) profErr = Math.abs(d - want);
    if (d > depth) depth = d;
    rows++;
  }
}
it('глубина канала = радиус каната', Math.abs(depth - R) / R < 0.02,
  `${(depth / U).toFixed(2)} мм при R = ${(R / U).toFixed(2)} мм`);
it('стенка канала = полукруг каната', profErr / U < 0.02,
  `макс отклонение ${(profErr / U).toFixed(3)} мм по ${rows} узлам дуги`);
it('ширина канала = 2 радиуса', widthErr < 0.03,
  `хорда кромки ${(2 * R * (1 + widthErr) / U).toFixed(2)} мм при 2R = ${(2 * R / U).toFixed(2)} мм`);

/* ---------- 4. каната больше нет, сетка цела ---------- */
const inScene = (mesh.parent && mesh.parent.children || []).includes(tubeObj);
it('каната в сцене нет', sculpt.ropeCount === 0 && !inScene,
  `канатов ${sculpt.ropeCount}, труба в сцене ${inScene}, впаяно ${sculpt.ropeAbsorbed}`);
const c1 = chk();
it('после прожига сетка замкнута', c1.openEdges === 0 && c1.nonManifold === 0 && c1.degenerate === 0
  && c1.watertight, `${c1.openEdges}/${c1.nonManifold}/${c1.degenerate}, watertight ${c1.watertight}`);
const f1 = I().length / 3;
it('грани канала встали на место кожи', f1 > fBefore && res.removed > 100,
  `граней ${fBefore} → ${f1} (−${res.removed} кожи, +${f1 - fBefore + res.removed})`);
let nan = 0;
for (let i = 0; i < P().length; i++) if (!Number.isFinite(P()[i])) nan++;
it('координаты без NaN', nan === 0, `NaN ${nan}`);

/* ---------- 5. объём: минус полуцилиндр каната ---------- */
const lost = (volBefore - volume(P(), I())) / U ** 3;
const want3 = Math.PI * (R / U) ** 2 / 2 * (L / U);
it('объём уменьшился на полуцилиндр каната', Math.abs(lost - want3) / want3 < 0.4,
  `−${lost.toFixed(1)} мм³ при ожидаемых ${want3.toFixed(1)} мм³ (${((lost / want3) * 100).toFixed(0)} %)`);

/* ---------- 6. повторный прожиг ---------- */
const again = sculpt.burnRope(rp.i);
it('повторный прожиг отклонён', again === null, info().slice(0, 70));

/* ---------- 7. отмена возвращает и кузов, и канат ---------- */
key('z', { ctrlKey: true });
const restored = maxDiff(before, P());
it('отмена вернула кузов', restored === 0 && I().length / 3 === fBefore,
  `макс сдвиг ${restored}, граней ${I().length / 3}`);
it('отмена вернула канат', sculpt.ropeCount === 1 && sculpt.ropeAbsorbed === 0,
  `канатов ${sculpt.ropeCount}, впаяно ${sculpt.ropeAbsorbed}`);

/* ---------- 8. прожиг там, где кожа под следом скачет (уступ панели):
   операция обязана отказаться и не трогать кузов ---------- */
reset();
tool('rope');
strokeSurface(150, 160, 400, 160, 26);
const rp2 = sculpt.ropeList()[0];
const before2 = snap();
const f2 = I().length / 3;
const refused = sculpt.burnRope(rp2.i);
it('на уступе панели — честный отказ', refused === null, info().slice(0, 90));
it('после отказа кузов не тронут', maxDiff(before2, P()) === 0 && I().length / 3 === f2
  && sculpt.ropeCount === 1, `граней ${I().length / 3}, канатов ${sculpt.ropeCount}`);

/* ---------- 9. прожиг без каната ---------- */
reset();
const none = sculpt.burnRope();
it('без каната прожиг отклонён', none === null, info().slice(0, 90));

console.log(`\nИТОГ «прожечь»: ${n - bad} из ${n} проверок зелёные, провалов ${bad}`);
if (bad) process.exit(1);
