/* Идея «полуканат»: канат ставится вполовину в кузов, всё, что оказалось за
   поверхностью, не существует; «◍ Впаять» делает наружную половину частью
   модели. Настоящий three, настоящий sculpt.js, настоящий кузов из body.glb.

   Запуск: sh tests/run_rope.sh   (или node tests/rope_half.mjs после mkharness.sh) */
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

/* ---------- 1. канат ставится вполовину ---------- */
reset();
const body0 = snap();
tool('rope');
strokeSurface(150, 160, 400, 160, 26);        // 42 мм по кожуху вдоль борта
const rp = sculpt.ropeList()[0];
const R = rp.radius, L = rp.len;
const tubeObj = sculpt.ropeMesh(0);
const tp = tubeObj.geometry.attributes.position.array;
const ti = tubeObj.geometry.index.array;
const pathPts = sculpt.ropePath(0), pathNrms = sculpt.ropeNormals(0);
console.log(`канат: длина ${(L / U).toFixed(1)} мм, радиус ${(R / U).toFixed(2)} мм, ` +
  `точек пути ${pathPts.length}, трубка ${tp.length / 3} вершин / ${ti.length / 3} граней, срез ${tubeObj.userData.tris ? '' : ''}off ${rp.off}\n`);

it('1. канат ставится вполовину в кузов (off = 0), а не лежит сверху',
  rp.off === 0, `off = ${rp.off} при радиусе ${(R / U).toFixed(2)} мм`);

/* точный замер: у каждой вершины кольца — расстояние вдоль нормали кожи */
let below = 0, minD = 1e9, rings = 0;
{
  const K = Math.round(tp.length / 3 / pathPts.length);      // точек в кольце
  rings = pathPts.length;
  for (let i = 0; i < rings; i++) {
    const c = pathPts[i], nr = pathNrms[i];
    const nl = Math.hypot(nr[0], nr[1], nr[2]) || 1;
    for (let j = 0; j < K; j++) {
      const k = (i * K + j) * 3;
      const d = ((tp[k] - c[0]) * nr[0] + (tp[k + 1] - c[1]) * nr[1] + (tp[k + 2] - c[2]) * nr[2]) / nl;
      if (d < -1e-4) below++;
      if (d < minD) minD = d;
    }
  }
}
it('2. за поверхность не свисает ни одной вершины — лишняя часть просто не существует',
  below === 0, `${below} вершин ниже кожи из ${tp.length / 3}, минимум ${(minD / U).toFixed(4)} мм, колец ${rings}`);

const vHalf = volume(tp, ti), vIdeal = Math.PI * R * R * L / 2;
it('3. в сцене ровно половина каната: объём = половина цилиндра',
  vHalf > 0 && Math.abs(vHalf / vIdeal - 1) < 0.05,
  `${(vHalf / 1000).toFixed(1)} тыс. ед³ против ${(vIdeal / 1000).toFixed(1)} тыс. (${(100 * (vHalf / vIdeal - 1)).toFixed(1)} %)`);

it('4. полуканат — замкнутое тело: дно-хорда и оба торца на месте',
  badEdges(ti) === 0, `рёбер с неверным числом граней ${badEdges(ti)}, граней ${ti.length / 3}`);

it('5. кузов от постановки каната не изменился',
  I().length / 3 === f0 && maxDiff(body0, P()) === 0,
  `граней ${I().length / 3}, сдвиг ${maxDiff(body0, P()).toExponential(1)}`);

/* ---------- 2. впаивание ---------- */
let raise0 = 0;
{
  /* ровный путь по панели двери: точки и нормали задаём явно, без экранной
     проекции — так канат точно там, где мы его мерим */
  reset();
  const R2 = 2.5 * U, z0 = 100;
  const pts = [], nrs = [];
  const posArr = P(), nrmArr = geo.attributes.normal.array;
  for (let k = 0; k <= 30; k++) {
    const x = 200 + (180 * k) / 30;
    const y = down(x, z0)[1];
    pts.push([x, y, z0]);
    let bi = 0, bd = Infinity;
    for (let v = 0; v < posArr.length; v += 3) {
      const d = (posArr[v] - x) ** 2 + (posArr[v + 1] - y) ** 2 + (posArr[v + 2] - z0) ** 2;
      if (d < bd) { bd = d; bi = v; }
    }
    nrs.push([nrmArr[bi], nrmArr[bi + 1], nrmArr[bi + 2]]);
  }
  const rope2 = sculpt.addRope(pts, nrs, 2.5);
  let len2 = 0;
  if (rope2) for (let i = 1; i < rope2.path.length; i++) {
    const a = rope2.path[i - 1], b = rope2.path[i];
    len2 += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
  }
  it('6. ровный канат по панели ставится вполовину (off = 0)',
    !!rope2 && rope2.off === 0,
    `«${rope2 && rope2.name}», длина ${(len2 / U).toFixed(1)} мм, точек ${rope2 ? rope2.path.length : 0}`);

  /* ось каната и поперечное направление — по самому пути */
  const pth = rope2.path, pnr = rope2.nrms;
  const B = [];
  for (let i = 0; i < pth.length; i++) {
    const a = pth[Math.max(0, i - 1)], b = pth[Math.min(pth.length - 1, i + 1)];
    const tx = b[0] - a[0], ty = b[1] - a[1], tz = b[2] - a[2];
    const tl = Math.hypot(tx, ty, tz) || 1;
    const tt = [tx / tl, ty / tl, tz / tl];
    const nn = pnr[Math.min(pnr.length - 1, i)] || [0, 1, 0];
    const bx = tt[1] * nn[2] - tt[2] * nn[1], by = tt[2] * nn[0] - tt[0] * nn[2], bz = tt[0] * nn[1] - tt[1] * nn[0];
    const bl = Math.hypot(bx, by, bz) || 1;
    B.push([bx / bl, by / bl, bz / bl]);
  }
  const near = (o, arr) => {                    // ближайшая точка пути
    let bk = 0, bd = Infinity;
    for (let k = 0; k < pth.length; k++) {
      const d = (pth[k][0] - arr[o]) ** 2 + (pth[k][1] - arr[o + 1]) ** 2 + (pth[k][2] - arr[o + 2]) ** 2;
      if (d < bd) { bd = d; bk = k; }
    }
    return bk;
  };

  const bodyBeforeFuse = snap();
  const t0 = Date.now();
  const fused = sculpt.fuseRope();               // по умолчанию — именно канавка
  const ms = Date.now() - t0;
  let maxDown = 0, up = 0, moved = 0;
  for (let v = 0; v < bodyBeforeFuse.length / 3; v++) {
    const o = v * 3;
    const dx = P()[o] - bodyBeforeFuse[o], dy = P()[o + 1] - bodyBeforeFuse[o + 1], dz = P()[o + 2] - bodyBeforeFuse[o + 2];
    if (Math.hypot(dx, dy, dz) < 1e-6) continue;
    moved++;
    const bk = near(o, bodyBeforeFuse), nr = pnr[Math.min(pnr.length - 1, bk)];
    const h = (dx * nr[0] + dy * nr[1] + dz * nr[2]) / U;   // мм: минус — вдавлено
    if (h > 0.3) up++;
    if (-h > maxDown) maxDown = -h;
  }
  raise0 = maxDown * U;
  const c = chk();
  console.log(`впаивание канавкой: ${ms} мс, вдавлено ${moved} вершин, глубина ${maxDown.toFixed(2)} мм, ` +
    `+${fused && fused.added} точек, граней ${I().length / 3}\n`);
  it('7. канавка ровно в радиус: кожа вдавлена на R, вспучивания нет',
    Math.abs(maxDown - R2 / U) < 0.1 && up === 0 && moved > 0,
    `глубина ${maxDown.toFixed(2)} мм при радиусе ${(R2 / U).toFixed(2)} мм, вдавлено ${moved} вершин, ` +
    `вспучивание больше 0.3 мм ${up}, ${ms} мс`);
  it('8. после впаивания кузов цел: дыр, неманифолдных и вырожденных нет',
    c.openEdges === 0 && c.nonManifold === 0 && c.degenerate === 0,
    `дыр ${c.openEdges}, неманиф ${c.nonManifold}, вырожд ${c.degenerate}, граней ${I().length / 3} (+${(I().length / 3) - f0})`);
  it('9. каната в сцене нет — остался только его след в кузове',
    sculpt.ropeCount === 0 && sculpt.ropeAbsorbed === 1 && !(mesh.parent.children || []).includes(tubeObj),
    `канатов в сцене ${sculpt.ropeCount}, впаяно ${sculpt.ropeAbsorbed}`);

  /* Профиль канавки — по самому полю деформации: поперечное смещение ↔ глубина */
  const nb = 17, tLo = -(R2 / U) * 1.8, tHi = (R2 / U) * 1.8;
  const prof = new Array(nb).fill(null);
  for (let v = 0; v < bodyBeforeFuse.length / 3; v++) {
    const o = v * 3;
    const dx = P()[o] - bodyBeforeFuse[o], dy = P()[o + 1] - bodyBeforeFuse[o + 1], dz = P()[o + 2] - bodyBeforeFuse[o + 2];
    if (Math.hypot(dx, dy, dz) < 1e-6) continue;
    const bk = near(o, bodyBeforeFuse);
    const qx = bodyBeforeFuse[o] - pth[bk][0], qy = bodyBeforeFuse[o + 1] - pth[bk][1], qz = bodyBeforeFuse[o + 2] - pth[bk][2];
    const t = (qx * B[bk][0] + qy * B[bk][1] + qz * B[bk][2]) / U;
    const h = (dx * pnr[bk][0] + dy * pnr[bk][1] + dz * pnr[bk][2]) / U;
    const bi = Math.round(((t - tLo) / (tHi - tLo)) * (nb - 1));
    if (bi < 0 || bi >= nb) continue;
    if (prof[bi] === null || h < prof[bi]) prof[bi] = h;    // глубже = отрицательнее
  }
  const ideal = Array.from({ length: nb }, (_, i) => {
    const t = tLo + ((tHi - tLo) * i) / (nb - 1);
    const u = Math.abs(t) / (R2 / U);
    return u >= 1 ? 0 : -(R2 / U) * Math.sqrt(1 - u * u);
  });
  const peak = -Math.min(...prof.filter(v => v !== null));
  let worst = 0, seen = 0;
  prof.forEach((v, i) => { if (v === null) return; seen++; worst = Math.max(worst, Math.abs(v - ideal[i])); });
  prof.forEach((v, i) => { if (v === null) worst = Math.max(worst, Math.abs(ideal[i])); });
  it('10. профиль поперёк — полукруг вниз на R, к краям сходит в ноль',
    peak > 0.9 * R2 / U && worst < 0.25 * R2 / U && seen >= nb - 6,
    `поперёк ${tLo.toFixed(1)}…${tHi.toFixed(1)} мм, глубина ${peak.toFixed(2)} мм, ` +
    `отклонение от полукруга до ${worst.toFixed(2)} мм · ` +
    `замер ${prof.map(v => (v === null ? '—' : v.toFixed(2))).join(' ')} · ` +
    `полукруг ${ideal.map(v => v.toFixed(2)).join(' ')}`);

  /* коридор: деформация только вокруг каната, только вниз и не глубже R */
  {
    let outside = 0, dirt = 0, deeper = 0;
    for (let v = 0; v < bodyBeforeFuse.length / 3; v++) {
      const o = v * 3;
      const dx = P()[o] - bodyBeforeFuse[o], dy = P()[o + 1] - bodyBeforeFuse[o + 1], dz = P()[o + 2] - bodyBeforeFuse[o + 2];
      if (Math.hypot(dx, dy, dz) < 1e-6) continue;
      const bk = near(o, bodyBeforeFuse);
      const qx = bodyBeforeFuse[o] - pth[bk][0], qy = bodyBeforeFuse[o + 1] - pth[bk][1], qz = bodyBeforeFuse[o + 2] - pth[bk][2];
      const t = (qx * B[bk][0] + qy * B[bk][1] + qz * B[bk][2]) / U;
      if (Math.abs(t) > 1.5 * (R2 / U)) outside++;
      const h = (dx * pnr[bk][0] + dy * pnr[bk][1] + dz * pnr[bk][2]) / U;
      if (h > 0.3) dirt++;                       // вспучивание не больше 12 % радиуса
      if (-h > 1.02 * (R2 / U)) deeper++;
    }
    it('11. деформация только вокруг каната: коридор 1.5R, не глубже R, без вспучивания',
      outside === 0 && dirt === 0 && deeper === 0,
      `вне коридора ${outside}, вспучивание больше 0.3 мм ${dirt}, глубже радиуса ${deeper}`);
  }

  /* ---------- 3. отмена/возврат ---------- */
  key('z', { ctrlKey: true });
  it('12. Ctrl+Z возвращает и кузов, и канат',
    maxDiff(bodyBeforeFuse, P()) === 0 && sculpt.ropeCount === 1 && sculpt.ropeAbsorbed === 0,
    `сдвиг ${maxDiff(bodyBeforeFuse, P()).toExponential(1)}, канатов ${sculpt.ropeCount}, впаяно ${sculpt.ropeAbsorbed}`);
  key('z', { ctrlKey: true, shiftKey: true });
  it('13. Ctrl+Shift+Z впаивает обратно, канавка на месте',
    sculpt.ropeAbsorbed === 1 && maxDiff(bodyBeforeFuse, P()) > raise0 * 0.9,
    `впаяно ${sculpt.ropeAbsorbed}, глубина ${(maxDiff(bodyBeforeFuse, P()) / U).toFixed(2)} мм`);
}

/* ---------- 4. кромка: канат доходит до края кузова ---------- */
{
  reset();
  tool('rope');
  strokeSurface(100, 150, 100, 258, 40, true);
  const rp2 = sculpt.ropeList().slice(-1)[0];
  const before = snap();
  sculpt.fuseRope();
  const c = chk();
  const path2 = sculpt.ropePath(rp2.i) || [];
  let far = 0;
  for (let i = 0; i < before.length; i += 3) {
    if (Math.hypot(P()[i] - before[i], P()[i + 1] - before[i + 1], P()[i + 2] - before[i + 2]) < 1e-6) continue;
    const p = [P()[i], P()[i + 1], P()[i + 2]];
    let bd = Infinity;
    for (const q of path2) bd = Math.min(bd, (q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2);
    if (Math.sqrt(bd) > 4 * rp2.radius) far++;
  }
  it('14. у кромки канавка обрывается сама: сетка цела, дальние точки не тронуты',
    c.openEdges === 0 && c.nonManifold === 0 && far === 0,
    `дыр ${c.openEdges}, неманиф ${c.nonManifold}, точек дальше 4R от каната ${far}, длина каната ${(rp2.len / U).toFixed(1)} мм`);
}

/* ---------- 5. режим «ребро» (Shift): кожа поднимается, а не вдавливается ---------- */
{
  reset();
  const R2 = 2.5 * U, z0 = 100, posArr = P(), nrmArr = geo.attributes.normal.array;
  const pts = [], nrs = [];
  for (let k = 0; k <= 30; k++) {
    const x = 200 + (180 * k) / 30;
    const y = down(x, z0)[1];
    pts.push([x, y, z0]);
    let bi = 0, bd = Infinity;
    for (let v = 0; v < posArr.length; v += 3) {
      const d = (posArr[v] - x) ** 2 + (posArr[v + 1] - y) ** 2 + (posArr[v + 2] - z0) ** 2;
      if (d < bd) { bd = d; bi = v; }
    }
    nrs.push([nrmArr[bi], nrmArr[bi + 1], nrmArr[bi + 2]]);
  }
  const rp3 = sculpt.addRope(pts, nrs, 2.5);
  const before = snap();
  const res = sculpt.fuseRope(rp3.i, { mode: 'ridge' });
  let upMax = 0, sunk = 0;
  for (let v = 0; v < before.length / 3; v++) {
    const o = v * 3;
    const dx = P()[o] - before[o], dy = P()[o + 1] - before[o + 1], dz = P()[o + 2] - before[o + 2];
    const dl = Math.hypot(dx, dy, dz);
    if (dl < 1e-6) continue;
    let bk = 0, bd = Infinity;
    for (let i = 0; i < rp3.path.length; i++) {
      const d = (rp3.path[i][0] - before[o]) ** 2 + (rp3.path[i][1] - before[o + 1]) ** 2 + (rp3.path[i][2] - before[o + 2]) ** 2;
      if (d < bd) { bd = d; bk = i; }
    }
    const nr = nrs[Math.min(nrs.length - 1, bk)];
    const h = (dx * nr[0] + dy * nr[1] + dz * nr[2]) / U;
    if (h > upMax) upMax = h;
    if (h < -0.3) sunk++;
  }
  it('15. Shift (режим «ребро») поднимает кожу на радиус — канавки не остаётся',
    !!res && res.mode === 'ridge' && Math.abs(upMax - R2 / U) < 0.1 && sunk === 0,
    `режим ${res && res.mode}, подъём ${upMax.toFixed(2)} мм, вдавлено больше 0.3 мм ${sunk}, ` +
    `канатов в сцене ${sculpt.ropeCount}`);
}

/* ---------- 6. кнопка «◍ Впаять» без Shift — именно канавка ---------- */
{
  reset();
  const z0 = 100, posArr = P(), nrmArr = geo.attributes.normal.array;
  const pts = [], nrs = [];
  for (let k = 0; k <= 30; k++) {
    const x = 200 + (180 * k) / 30;
    const y = down(x, z0)[1];
    pts.push([x, y, z0]);
    let bi = 0, bd = Infinity;
    for (let v = 0; v < posArr.length; v += 3) {
      const d = (posArr[v] - x) ** 2 + (posArr[v + 1] - y) ** 2 + (posArr[v + 2] - z0) ** 2;
      if (d < bd) { bd = d; bi = v; }
    }
    nrs.push([nrmArr[bi], nrmArr[bi + 1], nrmArr[bi + 2]]);
  }
  sculpt.addRope(pts, nrs, 2.5);
  const before = snap(), abs0 = sculpt.ropeAbsorbed;
  els['sc-fuse'].click();
  let deepest = 0, up = 0;
  for (let v = 0; v < before.length / 3; v++) {
    const o = v * 3;
    const dx = P()[o] - before[o], dy = P()[o + 1] - before[o + 1], dz = P()[o + 2] - before[o + 2];
    if (Math.hypot(dx, dy, dz) < 1e-6) continue;
    let bk = 0, bd = Infinity;
    for (let i = 0; i < pts.length; i++) {
      const d = (pts[i][0] - before[o]) ** 2 + (pts[i][1] - before[o + 1]) ** 2 + (pts[i][2] - before[o + 2]) ** 2;
      if (d < bd) { bd = d; bk = i; }
    }
    const nr = nrs[bk];
    const h = (dx * nr[0] + dy * nr[1] + dz * nr[2]) / U;
    if (-h > deepest) deepest = -h;
    if (h > 0.3) up++;
  }
  it('16. кнопка «◍ Впаять» без Shift оставляет канавку, а не ребро',
    sculpt.ropeAbsorbed === abs0 + 1 && sculpt.ropeCount === 0 && deepest > 2.2 && up === 0,
    `впаяно всего ${sculpt.ropeAbsorbed}, канатов в сцене ${sculpt.ropeCount}, глубина ${deepest.toFixed(2)} мм, вспучивание больше 0.3 мм ${up}`);
}

console.log(`\nИТОГ «полуканат»: ${n - bad} из ${n} проверок зелёные, провалов ${bad}`);
process.exit(bad ? 1 : 0);
