/* Удобство инструментов: 16 проверок горячих клавиш, памяти настроек,
   состояния кнопок и повтора действия.
   Запуск: sh tests/run_ux.sh   (соберёт body.json и стенд из sculpt.js) */
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
/* нажатие как в браузере: разные e.key и e.code (русская раскладка) */
const kdown = (f) => { const e = { target: { tagName: 'BODY' }, ctrlKey: false, metaKey: false,
  altKey: false, shiftKey: false, key: '', code: '', _pd: false, preventDefault() { this._pd = true; } };
  Object.assign(e, f); keys.forEach(h => h(e)); return e; };

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


const plain = () => info().replace(/<[^>]+>/g, '');
const setSlider = (id, v) => { els[id].value = String(v); els[id].fire('input'); };
const paint = (x0, z0, x1, z1, cnt = 16) => {
  const pts = []; for (let i = 0; i <= cnt; i++) { const t = i / cnt; pts.push(down(x0 + (x1 - x0) * t, z0 + (z1 - z0) * t)); }
  const sp = pts.map(p => scr(...p));
  canvas._ev.pointerdown(ev(sp[0][0], sp[0][1]));
  for (let i = 1; i < sp.length; i++) canvas._ev.pointermove(ev(sp[i][0], sp[i][1]));
  canvas._ev.pointerup(ev(sp[sp.length - 1][0], sp[sp.length - 1][1]));
};
const faces = () => I().length / 3;
const brushVal = () => els['sc-brush'].value;
setSlider('sc-brush', '1.2'); setSlider('sc-depth', '0.6');
reset();
/* проверка «повторять нечего» — до первого действия, иначе повторять уже есть что */
const earlyEnter = kdown({ key: 'Enter', code: 'Enter' });
const earlyMsg = plain();
const F0 = faces(), P0 = snap();

/* 1. Ctrl+Z на русской раскладке */
tool('select'); paint(400, 130, 400, 190);
els['sc-go'].click();
const d1 = maxShift(P0, P());
const e1 = kdown({ ctrlKey: true, key: 'я', code: 'KeyZ' });
const d1u = maxShift(P0, P());
it('1. Ctrl+Z работает на русской раскладке (e.key = «я»)', e1._pd && d1 > 1e-3 && d1u === 0,
  `сглажено на ${(d1 / U).toFixed(3)} мм → отмена вернула ровно исходник (${d1u})`);

/* 2. возврат: Ctrl+Shift+Z и Ctrl+Y */
const e2 = kdown({ ctrlKey: true, shiftKey: true, key: 'Я', code: 'KeyZ' });
const d2 = maxShift(P0, P());
kdown({ ctrlKey: true, key: 'я', code: 'KeyZ' });
const e3 = kdown({ ctrlKey: true, key: 'н', code: 'KeyY' });
const d3 = maxShift(P0, P());
it('2. возврат работает и Ctrl+Shift+Z, и Ctrl+Y', e2._pd && e3._pd && Math.abs(d2 - d1) < 1e-6 && Math.abs(d3 - d1) < 1e-6,
  `сдвиг после возврата ${(d2 / U).toFixed(3)} и ${(d3 / U).toFixed(3)} мм = тот же`);

/* 3. Ctrl+S сохраняет правку, а не открывает диалог браузера */
reset(); tool('select'); paint(400, 130, 400, 190); els['sc-out'].click();
const beforeSave = sculpt.dirty;
const e4 = kdown({ ctrlKey: true, key: 'ы', code: 'KeyS' });
it('3. Ctrl+S (раскладка RU) сохраняет проект, а не «сохранить страницу»',
  e4._pd && beforeSave === true && sculpt.dirty === false,
  `правка была несохранённой, после Ctrl+S dirty = ${sculpt.dirty}`);

/* 4. [ и ] на русской раскладке */
reset();
tool('select');
setSlider('sc-brush', '1.2');
const b0 = +brushVal();
kdown({ key: 'х', code: 'BracketLeft' });
const b1 = +brushVal();
kdown({ key: 'ъ', code: 'BracketRight' });
kdown({ key: 'ъ', code: 'BracketRight' });
const b2 = +brushVal();
it('4. [ и ] меняют кисть при русской раскладке (key «х»/«ъ»)', b0 === 1.2 && b1 === 0.7 && b2 === 1.7,
  `${b0} → ${b1} → ${b2} мм`);

/* 5. Enter повторяет последнее действие */
reset(); setSlider('sc-brush', '1.2'); tool('select');
paint(400, 130, 400, 190);
els['sc-go'].click();
const a1 = maxShift(P0, P());
const r1 = snap();
const e5 = kdown({ key: 'Enter', code: 'Enter' });
const a2 = maxShift(r1, P());
it('5. Enter повторяет последнее действие («Сгладить»)', e5._pd && a2 > 1e-4,
  `первое нажатие ${(a1 / U).toFixed(3)} мм, повтор ${(a2 / U).toFixed(3)} мм`);

/* 6. пробел — тоже повтор */
const r2 = snap();
kdown({ key: ' ', code: 'Space' });
const a3 = maxShift(r2, P());
it('6. Пробел повторяет то же действие', a3 > 1e-4, `ещё ${(a3 / U).toFixed(3)} мм`);

/* 7. Enter без действия — честное сообщение */
it('7. Enter без действия объясняет, что повторять нечего', earlyEnter._pd && /повторять нечего/.test(earlyMsg),
  `«${earlyMsg.slice(0, 60)}»`);

/* 8. «Кисть» общая у лепки и ластика: настроил раз — работает везде */
reset();
tool('select'); setSlider('sc-brush', '1.2');
const selBrush = +brushVal();
tool('erase'); setSlider('sc-brush', '3');
const eraseBrush = +brushVal();
tool('select');
const backSel = +brushVal();
tool('pen');
const penBrush = +brushVal();
it('8. «Кисть» общая у участка, ластика и пера (свои только у борозды/рамки/упрощения)',
  selBrush === 1.2 && eraseBrush === 3 && backSel === 3 && penBrush === 3,
  `участок ${selBrush} → ластик ${eraseBrush} → участок ${backSel} → перо ${penBrush}`);

/* 9. ширина борозды не портит кисть участка */
reset(); tool('select'); setSlider('sc-brush', '1.5'); tool(null);
tool('groove'); setSlider('sc-brush', '4');
tool('select');
const selAfterGroove = +brushVal();
tool('groove');
const grooveBack = +brushVal();
it('9. «Ширина» борозды 4 мм не трогает «Кисть» участка', selAfterGroove === 1.5 && grooveBack === 4,
  `участок ${selAfterGroove} мм, борозда ${grooveBack} мм`);

/* 10. глубина у выдавливания и борозды тоже своя */
reset(); tool('select'); setSlider('sc-depth', '0.6'); tool(null);
tool('groove'); setSlider('sc-depth', '1');
tool('select');
const depthSel = els['sc-depth'].value, depthGroove = (() => { tool('groove'); return els['sc-depth'].value; })();
it('10. «Глубина» тоже помнится отдельно', +depthSel === 0.6 && +depthGroove === 1,
  `участок ${depthSel} мм, борозда ${depthGroove} мм`);

/* 10b. «Степень» упрощения помнится отдельно от лепки */
reset(); tool('select'); setSlider('sc-amt', '25');
tool('lite'); setSlider('sc-amt', '100');
const liteAmt = els['sc-amt'].value;
tool('select');
const selAmt = els['sc-amt'].value;
tool('lite');
const liteBack = els['sc-amt'].value;
it('10b. «Степень» упрощения своя: 100 % не переезжает в лепку', +liteAmt === 100 && +selAmt === 25 && +liteBack === 100,
  `упрощение ${liteAmt} % → участок ${selAmt} % → упрощение ${liteBack} %`);

/* 10c. пробел на кнопке — это её нажатие, а не двойной повтор */
reset(); tool('select'); strokeSurface(400, 130, 400, 190, 14);
els['sc-go'].click();
const btnTarget = { tagName: 'BUTTON', closest: () => btnTarget };
const before1 = snap();
const eSpace = kdown({ key: ' ', code: 'Space', target: btnTarget });
it('10c. пробел на кнопке не повторяет действие второй раз', eSpace._pd === false && maxShift(before1, P()) === 0,
  `сдвиг после пробела на кнопке — ${maxShift(before1, P())}`);

/* 11. кнопки отмены гаснут, когда отменять нечего */
reset();
els['sc-clear'].click();          // после сброса история не пуста — проверим по шагам
const undoEl = els['sc-undo'], redoEl = els['sc-redo'];
const redoDisabled = redoEl.disabled;
tool('select'); paint(400, 130, 400, 190); els['sc-go'].click();
const undoOn = undoEl.disabled === false;
els['sc-undo'].click();
const redoOn = redoEl.disabled === false;
it('11. «Отменить»/«Вернуть» гаснут, когда нечего отменять', redoDisabled === true && undoOn && redoOn,
  `возврат в начале ${redoDisabled ? 'неактивен' : 'активен'}, после правки отмена ${undoOn ? 'активна' : 'нет'}, после отмены возврат ${redoOn ? 'активен' : 'нет'}`);

/* 12. метка несохранённого на «💾 Сохранить» */
reset();
const saveBtn = els['sc-proj-save'];
const cleanText = saveBtn.textContent;
tool('select'); paint(400, 130, 400, 190); els['sc-go'].click();
const dirtyText = saveBtn.textContent;
sculpt.saveProject();
const afterText = saveBtn.textContent;
it('12. «💾 Сохранить» помечает несохранённую правку и снимает метку', /●/.test(dirtyText) && cleanText === '💾 Сохранить' && afterText === '💾 Сохранить',
  `«${cleanText}» → «${dirtyText}» → «${afterText}»`);

/* 13. «Сброс» отменяется */
reset(); tool('select');
strokeSurface(400, 130, 400, 190, 14);
const edited = snap();
els['sc-clear'].click();
const cleared = maxShift(P0, P());
els['sc-undo'].click();
const back = maxShift(edited, P());
it('13. «Сброс» можно отменить (Ctrl+Z возвращает правку)', cleared === 0 && back === 0,
  `после сброса сдвиг ${cleared}, после отмены ${back}`);

/* 14. «0» и Esc не ломают инструмент на любой раскладке */
reset(); tool('select'); strokeSurface(400, 130, 400, 190, 10);
const selN = sculpt.regionCount;
const e0 = kdown({ key: '0', code: 'Digit0' });
const after0 = sculpt.regionCount;
const esc = kdown({ key: 'Escape', code: 'Escape' });
it('14. «0» снимает выделение, Esc выходит из инструмента', e0._pd && selN > 0 && after0 === 0 && esc._pd && sculpt.tool === null,
  `выделение ${selN} → ${after0}, инструмент после Esc: ${sculpt.tool}`);

console.log(`\nИТОГ удобства: ${n-bad} из ${n} тестов зелёные, провалов ${bad}`);
process.exitCode = bad ? 1 : 0;
