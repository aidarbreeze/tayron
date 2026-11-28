/* Куб-рамка: 13 проверок настоящего инструмента «7 ▭ Рамка».
   Запуск: sh tests/run_box.sh   (соберёт body.json и стенд из sculpt.js) */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = process.env.TAYRON_ROOT || path.resolve(HERE, '..');
const BODY = process.env.TAYRON_BODY || path.join(HERE, 'body.json');
const HARNESS = process.env.TAYRON_HARNESS || path.join(HERE, '.harness/sculpt_real.mjs');
const THREE = await import(path.join(ROOT, 'suspension/vendor/three.module.js'));
const { initSculpt } = await import(HARNESS);
const { buildTopology, checkMesh } = await import(path.join(ROOT, 'suspension/meshops.js'));

const U = 5.8824;
const HTML = fs.readFileSync(path.join(ROOT, 'suspension/index.html'), 'utf8');
const raw = JSON.parse(fs.readFileSync(BODY, 'utf8'));

/* ---------- стенд ---------- */
const els = {};
const mk = (id) => {
  const e = { id, value: '', style: {}, hidden: false, files: [], checked: false, _html: '', _text: '',
    classList: { s: new Set(), add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
      toggle(c, v) { v === undefined ? (this.s.has(c) ? this.s.delete(c) : this.s.add(c)) : (v ? this.s.add(c) : this.s.delete(c)); },
      contains(c) { return this.s.has(c); } },
    addEventListener(t, f) { ((this._ev ||= {})[t] ||= []).push(f); },
    fire(t, e = {}) { (this._ev && this._ev[t] || []).forEach(f => f(e)); },
    click() { this.fire('click'); } };
  Object.defineProperty(e, 'innerHTML', { get() { return this._html; }, set(v) { this._html = v; this._text = ''; } });
  Object.defineProperty(e, 'textContent', { get() { return this._text || this._html.replace(/<[^>]+>/g, ''); }, set(v) { this._text = v || ''; this._html = ''; } });
  return (els[id] = e);
};
const IDS = [...HTML.matchAll(/id="([a-z0-9-]+)"/g)].map(m => m[1]);
IDS.forEach(mk);
els['sc-brush'].value = '1.2'; els['sc-depth'].value = '0.6'; els['sc-amt'].value = '25';
const scRows = IDS.filter(id => id.endsWith('row')).map(id => els[id]);
const canvas = { style: {}, _ev: {}, addEventListener(t, f) { this._ev[t] = f; }, setPointerCapture() {} };
let dl = null;
globalThis.document = { getElementById: (id) => els[id] || null,
  querySelectorAll: (s) => (s === '.sc-row' ? scRows : []), body: { appendChild() {} },
  createElement: () => { dl = { style: {}, click() {}, remove() {}, setAttribute() {} }; return dl; } };
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
const it = (name, cond, note = '') => { if (!cond) bad++; n++;
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${note ? ' — ' + note : ''}`); };
let n = 0, bad = 0;
const info = () => els['sc-info'].innerHTML || els['sc-info'].textContent;
const P = () => geo.attributes.position.array, I = () => geo.index.array;
const ev = (x, y, shift = false) => ({ button: 0, clientX: x, clientY: y, pointerId: 1, shiftKey: shift, preventDefault() {} });
const key = (k, mods = {}) => keys.forEach(f => f({ key: k, target: { tagName: 'BODY' }, ctrlKey: false,
  metaKey: false, altKey: false, shiftKey: false, preventDefault() {}, ...mods }));
const scr = (x, y, z) => { const v = new THREE.Vector3(x, y, z); mesh.localToWorld(v); v.project(camera);
  return [(v.x * 0.5 + 0.5) * innerWidth, (-v.y * 0.5 + 0.5) * innerHeight]; };
const down = (x, z) => { const r = new THREE.Raycaster(new THREE.Vector3(x, 600, z), new THREE.Vector3(0, -1, 0), 0, 4000);
  const h = r.intersectObject(mesh, false); if (!h.length) throw new Error('мимо кузова: ' + x + ',' + z); return h[0].point.toArray(); };
const tool = (t) => { sculpt.setTool(null); if (t) sculpt.setTool(t); };
const stroke = (pts, steps = 16) => {
  const sp = pts.map(p => scr(...p));
  canvas._ev.pointerdown(ev(sp[0][0], sp[0][1]));
  for (let i = 1; i < sp.length - 1; i++) canvas._ev.pointermove(ev(sp[i][0], sp[i][1]));
  canvas._ev.pointerup(ev(sp[sp.length - 1][0], sp[sp.length - 1][1]));
};
const strokeSurface = (x0, z0, x1, z1, cnt = 20) => {
  const pts = []; for (let i = 0; i <= cnt; i++) { const t = i / cnt; pts.push(down(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t)); }
  const sp = pts.map(p => scr(...p));
  canvas._ev.pointerdown(ev(sp[0][0], sp[0][1]));
  for (let i = 1; i < sp.length; i++) canvas._ev.pointermove(ev(sp[i][0], sp[i][1]));
  canvas._ev.pointerup(ev(sp[sp.length - 1][0], sp[sp.length - 1][1]));
};
const snap = () => Float32Array.from(P());
const maxDiff = (a, b) => { let d = 0; const L = Math.min(a.length, b.length);
  for (let i = 0; i < L; i++) { const x = Math.abs(a[i] - b[i]); if (x > d) d = x; } return d; };
const maxShift = (a, b) => { let d = 0; const L = Math.min(a.length, b.length);
  for (let i = 0; i < L; i += 3) { const x = Math.hypot(a[i] - b[i], a[i + 1] - b[i + 1], a[i + 2] - b[i + 2]); if (x > d) d = x; } return d; };
const chk = () => checkMesh(buildTopology(P(), I()), I(), P(), U);
const reset = () => { els['sc-clear'].click(); sculpt.setTool(null); };
const pos0 = snap(), idx0 = Uint32Array.from(I());
const f0 = idx0.length / 3, v0 = pos0.length / 3;
reset();


/* ================= куб-рамка ================= */
const plain = () => info().replace(/<[^>]+>/g, '');
const boxStr = (b) => b ? `x ${(b.x0/U).toFixed(1)}…${(b.x1/U).toFixed(1)}, y ${(b.y0/U).toFixed(1)}…${(b.y1/U).toFixed(1)}, z ${(b.z0/U).toFixed(1)}…${(b.z1/U).toFixed(1)} мм` : 'нет';

/* рамка по правому борту (y < 0) */
reset(); tool('rect');
const p = P();
let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, cnt = 0;
for (let i = 0; i < p.length / 3; i++) {
  const x = p[i*3], y = p[i*3+1], z = p[i*3+2];
  if (x < 240 || x > 380 || z < 120 || z > 220 || y > -5) continue;
  const s = scr(x, y, z);
  x0 = Math.min(x0, s[0]); x1 = Math.max(x1, s[0]); y0 = Math.min(y0, s[1]); y1 = Math.max(y1, s[1]); cnt++;
}
canvas._ev.pointerdown(ev(x0 + 2, y0 + 2));
canvas._ev.pointermove(ev(x1 - 2, y1 - 2));
canvas._ev.pointerup(ev(x1 - 2, y1 - 2));

let box = sculpt.box;
it('1. рамка ставит куб', !!box && sculpt.boxVisible && sculpt.regionCount > 100,
  `${sculpt.regionCount} вершин · ${boxStr(box)}`);

const size0 = sculpt.boxSizeMM;
/* габарит куба по x/z — это габарит узлов, попавших в рамку на правом борту */
let px0 = 1e9, px1 = -1e9, pz0 = 1e9, pz1 = -1e9, insidePicked = 0, pickedAll = 0;
for (let i = 0; i < p.length / 3; i++) {
  const x = p[i*3], y = p[i*3+1], z = p[i*3+2];
  if (y > -1) continue;
  const sc = scr(x, y, z);
  if (sc[0] < x0 || sc[0] > x1 || sc[1] < y0 || sc[1] > y1) continue;
  px0 = Math.min(px0, x); px1 = Math.max(px1, x); pz0 = Math.min(pz0, z); pz1 = Math.max(pz1, z);
  pickedAll++;
  if (x >= box.x0 && x <= box.x1 && y >= box.y0 && y <= box.y1 && z >= box.z0 && z <= box.z1) insidePicked++;
}
const gap = [Math.abs(box.x0 - px0), Math.abs(box.x1 - px1), Math.abs(box.z0 - pz0), Math.abs(box.z1 - pz1)].map(v => v / U);
const fits = gap.every(v => v < 5);   // куб не выходит за рамку (разница — скрытые узлы)
it('2. куб по габариту выделения, во всю толщину борта', box && fits && insidePicked / pickedAll > 0.98 &&
  box.y1 <= 0 && (-box.y0 / U) > 20 && sculpt.regionCount > pickedAll,
  `размер ${size0.map(v => v.toFixed(1)).join(' × ')} мм = рамка, ${insidePicked} из ${pickedAll} узлов рамки внутри, ` +
  `в кубе ${sculpt.regionCount} узлов (вся толщина борта ${(-box.y0/U).toFixed(1)} мм), не вылез за рамку больше ${Math.max(...gap).toFixed(2)} мм · внутри ${insidePicked} из ${pickedAll}`);

it('3. направление по умолчанию — на левую', box && box.to === 1 && /источник правая → копия левая/.test(plain()),
  `to=${box && box.to} · «${plain().slice(0, 80)}»`);

const before = sculpt.regionCount;
const flip = sculpt.boxFlip();
const b2 = sculpt.box;
it('4. «⇄ Сторона» переворачивает куб и направление', flip && b2 && b2.y0 > 0 && b2.y1 === -box.y0 && b2.to === -1 && sculpt.regionCount > 0,
  `${boxStr(b2)} · to=${b2.to} · вершин ${before} → ${sculpt.regionCount}`);

/* перетаскивание центральной ручки мышью */
const c0 = sculpt.boxPtr.slice(), hc = sculpt.boxHandleAt('move');
canvas._ev.pointerdown(ev(hc[0], hc[1]));
canvas._ev.pointermove(ev(hc[0] + 40, hc[1] + 12));
canvas._ev.pointerup(ev(hc[0] + 40, hc[1] + 12));
const b3 = sculpt.box, c1 = sculpt.boxPtr;
const moved = Math.hypot(c1[0]-c0[0], c1[1]-c0[1], c1[2]-c0[2]) / U;
it('5. куб тащится за центральную ручку', moved > 1 && Math.abs((b3.x1-b3.x0)-(b2.x1-b2.x0)) < 1e-6 && b3.y0 > 0,
  `центр уехал на ${moved.toFixed(1)} мм, размер тот же, сторона левая`);

/* тянем грань +X */
const gx0 = b3.x1;
const hf = sculpt.boxHandleAt('face', 0, 1);
canvas._ev.pointerdown(ev(hf[0], hf[1]));
canvas._ev.pointermove(ev(hf[0] + 25, hf[1] + 8));
canvas._ev.pointerup(ev(hf[0] + 25, hf[1] + 8));
const b4 = sculpt.box;
it('6. грань куба меняет его размер', b4.x1 > gx0 + 0.5*U && Math.abs(b4.x0-b3.x0) < 1e-6,
  `x1 ${(gx0/U).toFixed(1)} → ${(b4.x1/U).toFixed(1)} мм, x0 на месте (${(b4.x0/U).toFixed(1)})`);

/* грань по Y не пускает куб через плоскость симметрии */
const hf2 = sculpt.boxHandleAt('face', 1, -1);
canvas._ev.pointerdown(ev(hf2[0], hf2[1]));
canvas._ev.pointermove(ev(hf2[0] - 300, hf2[1] + 300));
canvas._ev.pointerup(ev(hf2[0] - 300, hf2[1] + 300));
const b5 = sculpt.box;
it('7. куб не переезжает через плоскость симметрии', b5.y0 >= 0.25*U - 1e-6,
  `y0 = ${(b5.y0/U).toFixed(2)} мм (зазор 0.25 мм), размер ${sculpt.boxSizeMM.map(v=>v.toFixed(1)).join(' × ')}`);

/* Shift + протяжка внутри куба — тоже перенос */
const bc0 = sculpt.boxPtr.slice();
const pIn = scr(sculpt.boxPtr[0], sculpt.boxPtr[1], sculpt.boxPtr[2]);
canvas._ev.pointerdown(ev(pIn[0], pIn[1], true));
canvas._ev.pointermove(ev(pIn[0] - 30, pIn[1] + 10, true));
canvas._ev.pointerup(ev(pIn[0] - 30, pIn[1] + 10, true));
const bc1 = sculpt.boxPtr;
const shiftMoved = Math.hypot(bc1[0]-bc0[0], bc1[1]-bc0[1], bc1[2]-bc0[2]) / U;
it('7b. Shift + протяжка внутри куба двигает куб', shiftMoved > 0.5 && !!sculpt.box,
  `центр уехал на ${shiftMoved.toFixed(1)} мм, куб на месте (${boxStr(sculpt.box)})`);

/* зеркальная копия из куба: свежая рамка по скромному участку */
reset(); tool('rect');
canvas._ev.pointerdown(ev(x0 + 2, y0 + 2));
canvas._ev.pointermove(ev((x0 + x1) / 2, (y0 + y1) / 2));
canvas._ev.pointerup(ev((x0 + x1) / 2, (y0 + y1) / 2));
const small = sculpt.box;
const faces0 = sculpt.stats().faces;
els['sc-mirror'].click();
const c = chk();
const faces1 = sculpt.stats().faces;
const grown = Math.abs(faces1 - faces0) < faces0 * 0.5;
it('8. «Зеркально» копирует из куба, сетка замкнута', faces1 !== faces0 && grown && c.openEdges === 0 && c.nonManifold === 0,
  `куб ${(small.x1-small.x0)/U|0}×${(small.y1-small.y0)/U|0}×${(small.z1-small.z0)/U|0} мм, граней ${faces0} → ${faces1}, дыр ${c.openEdges}, неманиф ${c.nonManifold}`);

it('9. куб пережил копию и остался на месте', !!sculpt.box && Math.abs(sculpt.box.x0 - small.x0) < 1e-6,
  boxStr(sculpt.box));

/* убрать куб: выделение остаётся */
reset(); tool('rect');
canvas._ev.pointerdown(ev(x0 + 2, y0 + 2)); canvas._ev.pointermove(ev((x0 + x1) / 2, (y0 + y1) / 2)); canvas._ev.pointerup(ev((x0 + x1) / 2, (y0 + y1) / 2));
const nSel = sculpt.regionCount;
const drop = sculpt.boxDrop();
it('10. «✕ Убрать куб» оставляет выделение', drop && sculpt.box === null && !sculpt.boxVisible &&
  nSel > 100 && sculpt.regionCount === nSel,
  `куба нет, выделение ${sculpt.regionCount} вершин (было ${nSel})`);

/* смена инструмента прячет куб, возврат — показывает снова */
reset(); tool('rect');
canvas._ev.pointerdown(ev(x0 + 2, y0 + 2)); canvas._ev.pointermove(ev(x1 - 2, y1 - 2)); canvas._ev.pointerup(ev(x1 - 2, y1 - 2));
const vis1 = sculpt.boxVisible;
tool('select'); const vis2 = sculpt.boxVisible;
tool('rect'); const vis3 = sculpt.boxVisible;
it('11. куб виден только у инструмента «Рамка»', vis1 && !vis2 && vis3, `${vis1} / ${vis2} / ${vis3}`);

/* сброс убирает куб */
els['sc-clear'].click();
it('12. «Сброс» убирает куб', sculpt.box === null && !sculpt.boxVisible && sculpt.regionCount === 0);

console.log(`\nИТОГ куба-рамки: ${n - bad} из ${n} тестов зелёные, провалов ${bad}`);
process.exitCode = bad ? 1 : 0;
