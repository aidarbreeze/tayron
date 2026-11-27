/* 20 сквозных тестов лепки кузова: настоящий three, настоящий sculpt.js
   через initSculpt, клики по кнопкам и штрихи мышью.
   Запуск: sh tests/run20.sh   (сам соберёт body.json и стенд) */
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
const { meshTo3mf } = await import(path.join(ROOT, 'suspension/export3mf.js'));

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
const plain = () => info().replace(/<[^>]+>/g, '');
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

/* ---------- 1. разметка ---------- */
{
  const all = [...HTML.matchAll(/id="([a-z0-9-]+)"/g)].map(m => m[1]);
  const dup = all.filter((id, i) => all.indexOf(id) !== i);
  const btns = [...HTML.matchAll(/<button[^>]*id="([a-z0-9-]+)"/g)].map(m => m[1]);
  const js = fs.readdirSync(path.join(ROOT, 'suspension')).filter(f => f.endsWith('.js'))
    .map(f => fs.readFileSync(path.join(ROOT, 'suspension', f), 'utf8')).join('\n');
  const dead = btns.filter(id => !js.includes(`'${id}'`) && !js.includes(`'${id.replace(/-(sphere|cylinder|box)$/, '-')}'`));
  it('1. разметка: id уникальны, все кнопки описаны в js', dup.length === 0 && dead.length === 0,
    `${all.length} id, кнопок ${btns.length}, дублей ${dup.length}, «мёртвых» ${dead.length}${dead.length ? ': ' + dead.join(' ') : ''}`);
}

/* ---------- 2. модули и API ---------- */
{
  const need = ['setTool', 'deselect', 'mirrorCopy', 'makeGroove', 'rectSelect', 'saveProject', 'loadProject',
    'runCheck', 'burnRope', 'ropeList', 'simplifyAt', 'stats', 'ropeAlign', 'toggleWire'];
  const missing = need.filter(k => sculpt[k] === undefined && typeof sculpt[k] !== 'function');
  it('2. initSculpt отдал весь API', missing.length === 0, `${need.length - missing.length}/${need.length} методов, нет: ${missing.join(' ') || '—'}`);
}

/* ---------- 3. кузов ---------- */
{
  const c = chk();
  it('3. кузов загружен и замкнут', c.faces === f0 && c.vertices === v0 && c.watertight,
    `${c.faces} граней, ${c.vertices} вершин, объём ${(c.volumeMM3 / 1000).toFixed(2)} см³, дыр ${c.openEdges}`);
}

/* ---------- 4. кнопки инструментов ---------- */
{
  const BTN = [['sc-mode', 'select'], ['sc-del', 'erase'], ['sc-pen', 'pen'], ['sc-shape', 'shape'],
    ['sc-point', 'point'], ['sc-measure', 'measure'], ['sc-rect', 'rect'], ['sc-groove', 'groove'],
    ['sc-rope', 'rope'], ['sc-lite', 'lite']];
  let good = 0;
  for (const [id, t] of BTN) { els[id].click(); if (sculpt.tool === t) good++; els[id].click(); if (sculpt.tool === null) good++; }
  it('4. десять кнопок включают и выключают инструмент', good === BTN.length * 2, `${good} из ${BTN.length * 2} переключений`);
}

/* ---------- 5. «Участок» ---------- */
{
  reset(); tool('select');
  strokeSurface(400, 130, 400, 190, 20);
  const sel = sculpt.regionCount;
  it('5. «Участок» закрашивает область', sel > 50 && /выбрано|верш/.test(plain()),
    `${sel} вершин · «${plain().slice(0, 50)}»`);
}

/* ---------- 6. «Сгладить» ---------- */
{
  const before = snap();
  els['sc-go'].click();
  const d1 = maxShift(before, P());
  const after1 = snap();
  els['sc-go'].click();
  const d2 = maxShift(after1, P());
  it('6. «Сгладить» сдвигает и затухает', d1 / U > 0.001 && d2 / U < d1 / U,
    `1-е нажатие ${(d1 / U).toFixed(3)} мм, 2-е ${(d2 / U).toFixed(3)} мм`);
}

/* ---------- 7. «Выровнять» ---------- */
{
  const before = snap();
  els['sc-flat'].click();
  const d = maxShift(before, P());
  it('7. «Выровнять» садит область в плоскость', d / U > 0.001, `макс. сдвиг ${(d / U).toFixed(3)} мм`);
}

/* ---------- 8. «Выдавить» 0.6 мм ---------- */
{
  reset(); tool('select');
  for (const z of [140, 160, 180]) strokeSurface(400, z - 10, 400, z + 10, 10);
  strokeSurface(400, 130, 400, 190, 16);
  els['sc-depth'].value = '0.6'; els['sc-depth'].fire('input');
  const before = snap();
  const sel = sculpt.regionCount;
  els['sc-out'].click();
  const d = maxShift(before, P());
  it('8. «Выдавить» уезжает на «Глубину»', Math.abs(d / U - 0.6) < 0.02 && sel > 60,
    `${sel} вершин, макс. сдвиг ${(d / U).toFixed(3)} мм при заданных 0.60`);
}

/* ---------- 9. «Вдавить» возвращает ---------- */
{
  const before = snap();
  els['sc-in'].click();
  const d = maxShift(before, P());
  it('9. «Вдавить» возвращает на ту же глубину', Math.abs(d / U - 0.6) < 0.02, `обратный сдвиг ${(d / U).toFixed(3)} мм`);
}

/* ---------- 10. «Ластик» + зашивка ---------- */
{
  reset(); tool('erase');
  els['sc-brush'].value = '3'; els['sc-brush'].fire('input');
  const p = down(400, 160);
  stroke([p]);
  const c = chk();
  const fN = I().length / 3;
  it('10. «Ластик» удаляет точки и зашивает дыру', fN < f0 && c.openEdges === 0 && c.nonManifold === 0,
    `граней ${f0} → ${fN} (−${f0 - fN}), дыр ${c.openEdges}, неманиф ${c.nonManifold}`);
}

/* ---------- 11. «Точка» ---------- */
{
  reset(); tool('point');
  const before = sculpt.stats();
  const p = down(400, 160);
  stroke([p]);
  const st = sculpt.stats(), c = chk();
  it('11. «Точка» вставляет узел и две грани', st.faces === before.faces + 2 && c.degenerate === 0,
    `вершин ${before.vertices} → ${st.vertices}, граней ${before.faces} → ${st.faces}, вырожд ${c.degenerate}`);
}

/* ---------- 12. «Рамка» → зеркальная копия ---------- */
{
  reset(); tool('rect');
  /* окно на правом борту: считаем по экранным координатам живых вершин */
  const p = P();
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9, cnt = 0;
  for (let i = 0; i < p.length / 3; i++) {
    const x = p[i * 3], y = p[i * 3 + 1], z = p[i * 3 + 2];
    if (x < 240 || x > 380 || z < 120 || z > 220 || y > -5) continue;   // только правый борт
    const s = scr(x, y, z);
    x0 = Math.min(x0, s[0]); x1 = Math.max(x1, s[0]); y0 = Math.min(y0, s[1]); y1 = Math.max(y1, s[1]); cnt++;
  }
  canvas._ev.pointerdown(ev(x0 + 2, y0 + 2));
  canvas._ev.pointermove(ev(x1 - 2, y1 - 2));
  canvas._ev.pointerup(ev(x1 - 2, y1 - 2));
  const sel = sculpt.regionCount;
  els['sc-mirror'].click();
  const c = chk();
  it('12. «Рамка» + «Зеркально» копирует сторону', sel > 100 && sculpt.dirty === true && c.openEdges === 0,
    `в рамке ${cnt} вершин, выделено ${sel}, после копии дыр ${c.openEdges} · «${plain().slice(0, 60)}»`);
}

/* ---------- 13. «Борозда» ---------- */
{
  reset(); tool('groove');
  els['sc-brush'].value = '2'; els['sc-brush'].fire('input');
  els['sc-depth'].value = '1'; els['sc-depth'].fire('input');
  const before = snap();
  strokeSurface(400, 130, 400, 190, 16);
  const g = sculpt.groove, c = chk();
  let depth = 0;
  for (let i = 0; i < before.length; i += 3) {
    const d = Math.hypot(P()[i] - before[i], P()[i + 1] - before[i + 1], P()[i + 2] - before[i + 2]);
    if (d > depth) depth = d;
  }
  it('13. «Борозда» режет желоб заданной глубины', !!g && Math.abs(depth / U - 1) < 0.15 && c.openEdges === 0,
    `глубина ${(depth / U).toFixed(2)} мм при заданной 1.00, дыр ${c.openEdges}, ${((c.faces - f0) / 1).toFixed(0)} новых граней`);
}

/* ---------- 14. «Канат» создаётся, кузов не меняется ---------- */
{
  reset(); tool('rope');
  const before = snap();
  strokeSurface(400, 125, 400, 190, 18);
  const rp = sculpt.ropeList()[0];
  it('14. «Канат» — отдельный объект, кузов не тронут',
    !!rp && rp.inScene && sculpt.stats().faces === f0 && maxDiff(before, P()) === 0,
    `«${rp && rp.name}», длина ${rp ? (rp.len / U).toFixed(1) : 0} мм, ⌀${rp ? (2 * rp.radius / U).toFixed(1) : 0} мм, граней кузова ${sculpt.stats().faces}`);
}

/* ---------- 15. прожиг канала ---------- */
{
  const before = snap();
  const t0 = Date.now();
  const burn = sculpt.burnRope();
  const ms = Date.now() - t0;
  let depth = 0;
  for (let i = 0; i < before.length; i += 3) {
    const d = Math.hypot(P()[i] - before[i], P()[i + 1] - before[i + 1], P()[i + 2] - before[i + 2]);
    if (d > depth) depth = d;
  }
  const c = chk();
  it('15. прожиг канала: глубина и целая сетка', depth / U > 1 && c.openEdges === 0 && c.nonManifold === 0,
    `глубина ${(depth / U).toFixed(2)} мм, +${burn && burn.added} точек, дыр ${c.openEdges}, неманиф ${c.nonManifold}, ${ms} мс`);
}

/* ---------- 16. «Упростить» ---------- */
{
  reset(); tool('lite');
  els['sc-brush'].value = '6'; els['sc-brush'].fire('input');
  els['sc-amt'].value = '100'; els['sc-amt'].fire('input');
  strokeSurface(400, 130, 400, 190, 20);
  const fN = I().length / 3, c = chk();
  it('16. «Упростить» убирает грани, сетка цела', fN < f0 && c.openEdges === 0 && c.degenerate === 0,
    `граней ${f0} → ${fN} (−${f0 - fN}), вершин ${v0} → ${P().length / 3}, дыр ${c.openEdges}, вырожд ${c.degenerate}`);
}

/* ---------- 17. отмена и возврат ---------- */
{
  const simplified = I().length / 3;
  els['sc-undo'].click();
  const afterUndo = I().length / 3, dU = maxDiff(pos0, P());
  els['sc-redo'].click();
  const afterRedo = I().length / 3;
  it('17. отмена и возврат восстанавливают сетку', afterUndo === f0 && dU === 0 && afterRedo === simplified,
    `после отмены ${afterUndo} граней (сдвиг ${dU}), после возврата ${afterRedo}`);
}

/* ---------- 18. «Сброс» ---------- */
{
  reset();
  const c = chk();
  it('18. «Сброс» возвращает исходный кузов', I().length / 3 === f0 && P().length / 3 === v0 && maxDiff(pos0, P()) === 0,
    `${I().length / 3} граней, ${P().length / 3} вершин, объём ${(c.volumeMM3 / 1000).toFixed(2)} см³`);
}

/* ---------- 19. «Проверка» и сохранение 3MF ---------- */
{
  els['sc-check'].click();
  const txt = plain();
  els['sc-save'].click();
  const saveTxt = plain();
  /* файл 3MF собираем тем же кодом, что и кнопка */
  const bytes = meshTo3mf(Float32Array.from(P()), Uint32Array.from(I()), { name: 'Кузов', meta: 'тест' });
  const pk = bytes[0] === 0x50 && bytes[1] === 0x4b;
  it('19. «Проверка» и сохранение 3MF', /сетка замкнута/.test(txt) && pk && /скачать Tayron_body_smooth\.3mf/.test(saveTxt),
    `«${txt.slice(0, 30)}…», файл ${(bytes.length / 1048576).toFixed(2)} МБ, ZIP ${pk ? 'да' : 'нет'}`);
}

/* ---------- 20. проект: сохранить → загрузить ---------- */
{
  reset(); tool('select');
  strokeSurface(400, 130, 400, 190, 20);
  els['sc-depth'].value = '0.4'; els['sc-depth'].fire('input');
  els['sc-go'].click();
  const edited = snap();
  const data = sculpt.saveProject();
  const json = JSON.stringify(data);
  reset();
  const back = sculpt.loadProject(json);
  const d = maxDiff(edited, P());
  it('20. правка сохраняется и загружается обратно', back === true && d / U < 1e-4,
    `файл ${(json.length / 1024).toFixed(1)} КБ, сдвинуто вершин ${data.verts.length / 4}, расхождение ${(d / U * 1000).toFixed(3)} мкм`);
}

console.log(`\nИТОГ приложения: ${n - bad} из ${n} тестов зелёные, провалов ${bad}`);
process.exitCode = bad ? 1 : 0;
