/* Страница студии так, как её видит браузер: настоящий three, настоящий
   sculpt.js, настоящий studio.js, настоящий сервер studio.py по HTTP.
   Подменены только DOM, localStorage, WebGL и адрес страницы — то, чего
   в node нет.

   Проверяем путь пользователя: загрузил файл → открыл /edit?id=… → правит
   мышью → «Сохранить 3MF» → версия на диске → «Правку» → автосохранение
   в проект → «Вернуть правку» → Ctrl+Z.

   Запуск:  node tests/studio_page.mjs      (нужен node 18+ и python3) */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import * as THREE from './studio_dom/three_page.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.dirname(HERE);

/* ---------- 1. сервер studio.py ---------- */
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'studio_page_'));
const port = 8300 + Math.floor(Math.random() * 600);
const BASE = `http://127.0.0.1:${port}`;
const srv = spawn('python3', ['studio.py', '--port', String(port), '--dir', work, '--quiet'],
  { cwd: REPO, stdio: ['ignore', 'pipe', 'pipe'] });
let srvLog = '';
srv.stdout.on('data', (d) => { srvLog += d; });
srv.stderr.on('data', (d) => { srvLog += d; });
const stop = () => { try { srv.kill('SIGTERM'); } catch (_) {} };
process.on('exit', stop);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const realFetch = globalThis.fetch;
for (let i = 0; i < 200; i++) {
  try { const r = await realFetch(BASE + '/api/health'); if (r.ok) break; } catch (_) { /* ждём */ }
  await sleep(50);
}
const health = await (await realFetch(BASE + '/api/health')).json().catch(() => ({}));
if (!health.ok) { console.error('сервер не поднялся:\n' + srvLog); process.exit(1); }

/* ---------- 2. загрузка GLB как из браузера ---------- */
const glb = fs.readFileSync(path.join(REPO, 'suspension/assets/body.glb'));
const form = new FormData();
form.append('file', new Blob([glb], { type: 'model/gltf-binary' }), 'body.glb');
form.append('scale_mm', '150');
const up = await (await realFetch(BASE + '/api/upload', { method: 'POST', body: form })).json();
const PID = up.id;
const META = await (await realFetch(`${BASE}/api/project/${PID}`)).json();
const BODY = await (await realFetch(`${BASE}/api/project/${PID}/body.json`)).json();

/* ---------- 3. подопытные копии модулей (меняем только пути импортов) ---------- */
const fixImports = (text) => text
  .replace(/from 'three'/, `from '${HERE}/studio_dom/three_page.mjs'`)
  .replace(/from 'three\/addons\/controls\/OrbitControls\.js'/, `from '${HERE}/studio_dom/orbit_shim.mjs'`)
  .replace(/from '\.\/([a-zA-Z0-9_.]+)'/g, `from '${REPO}/suspension/$1'`);
const build = path.join(os.tmpdir(), 'studio_page_build');
fs.mkdirSync(build, { recursive: true });
const page = fixImports(fs.readFileSync(path.join(REPO, 'suspension/studio.js'), 'utf8'))
  .replace(`'${REPO}/suspension/sculpt.js'`, `'${build}/sculpt_page.mjs'`);
if (/from '(three|\.\/sculpt)/.test(page)) throw new Error('не переписали импорты studio.js');
fs.writeFileSync(path.join(build, 'studio_real.mjs'), page);
fs.writeFileSync(path.join(build, 'sculpt_page.mjs'),
  fixImports(fs.readFileSync(path.join(REPO, 'suspension/sculpt.js'), 'utf8')));

/* ---------- 4. DOM, localStorage, WebGL ---------- */
const idsOf = (file) => [...fs.readFileSync(path.join(REPO, file), 'utf8')
  .matchAll(/id="([a-z0-9-]+)"/g)].map((m) => m[1]);
const els = {};
const mk = (id) => {
  const e = {
    id, value: '', style: {}, hidden: false, files: [], checked: false, title: '', _html: '', _text: '',
    classList: { s: new Set(), add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
      toggle(c, v) { v ? this.s.add(c) : this.s.delete(c); }, contains(c) { return this.s.has(c); } },
    addEventListener(t, f) { ((this._ev ||= {})[t] ||= []).push(f); },
    removeEventListener() {},
    fire(t, e = {}) { ((this._ev || {})[t] || []).forEach((f) => f(e)); },
    click() {
      if (typeof this.onclick === 'function') this.onclick({ target: this, preventDefault() {} });
      this.fire('click');
    },
    appendChild() {}, setAttribute() {},
    getBoundingClientRect() { return { left: 0, top: 0, width: 1200, height: 700 }; },
  };
  Object.defineProperty(e, 'innerHTML', { get() { return this._html; }, set(v) { this._html = v; this._text = ''; } });
  Object.defineProperty(e, 'textContent', {
    get() { return this._text || this._html.replace(/<[^>]+>/g, ''); }, set(v) { this._text = String(v ?? ''); this._html = ''; } });
  return (els[id] = e);
};
[...new Set([...idsOf('suspension/index.html'), ...idsOf('suspension/studio.html')])].forEach(mk);
els['sc-brush'].value = '1.2';
els['sc-depth'].value = '0.6';
els['sc-amt'].value = '25';
const scRows = Object.keys(els).filter((id) => id.endsWith('row')).map((id) => els[id]);
globalThis.document = {
  getElementById: (id) => els[id] || null,
  querySelectorAll: (s) => (s === '.sc-row' ? scRows : []),
  body: { appendChild() {}, innerHTML: '' },
  createElement: () => ({ style: {}, click() {}, remove() {}, setAttribute() {}, appendChild() {} }),
};
globalThis.innerWidth = 1200;
globalThis.innerHeight = 800;
globalThis.devicePixelRatio = 1;
globalThis.location = { search: '?id=' + PID, href: BASE + '/edit?id=' + PID };
const ls = new Map();
globalThis.localStorage = {
  getItem: (k) => (ls.has(k) ? ls.get(k) : null),
  setItem: (k, v) => ls.set(k, String(v)),
  removeItem: (k) => ls.delete(k),
};
globalThis.addEventListener = (t, f) => { ((globalThis._ev ||= {})[t] ||= []).push(f); };
globalThis.removeEventListener = () => {};
let raf = 0;
globalThis.requestAnimationFrame = (f) => { if (raf++ < 3) setTimeout(() => f(performance.now()), 0); return raf; };
/* относительные адреса страницы → настоящий сервер */
const NET = { posts: [], gets: [] };
globalThis.fetch = (url, opt = {}) => {
  const u = String(url);
  const full = u.startsWith('http') ? u : BASE + u;
  ((opt.method || 'GET').toUpperCase() === 'POST' ? NET.posts : NET.gets).push(u);
  return realFetch(full, opt);
};
globalThis.prompt = () => 'Кузов Tayron (из студии)';

/* ---------- 5. проверки ---------- */
let n = 0, bad = 0, lost = '';
const waitFor = async (fn, ms = 30000) => { const t0 = Date.now();
  while (Date.now() - t0 < ms) { if (await fn()) return true; await sleep(50); } return false; };
const it = (name, cond, note = '') => { if (!cond) bad++; n++;
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${note ? ' — ' + note : ''}`); };
const plain = () => (els['sc-info'].innerHTML || els['sc-info'].textContent || '').replace(/<[^>]+>/g, '');

{
  const t0 = performance.now();
  await import(path.join(build, 'studio_real.mjs'));
  const ms = Math.round(performance.now() - t0);
  const S = globalThis.studio;
  it('1. страница студии поднялась, проект взят из адреса',
    !!S && S.PID === PID, `${ms} мс, id ${PID}`);
  it('2. в шапке название проекта и файл, из которого он пришёл',
    els['sb-name'].textContent === META.name && els['sb-name'].title.includes(META.source),
    `«${els['sb-name'].textContent}», подсказка «${els['sb-name'].title}»`);
  const g = S.mesh.geometry;
  it('3. вся сетка проекта в сцене одной геометрией',
    g.attributes.position.count === BODY.pos.length && g.index.count === BODY.idx.length,
    `${g.attributes.position.count} вершин, ${g.index.count / 3} граней, индекс ${g.index.array.constructor.name}`);
  it('4. масштаб печати взят из проекта (U = длина / 150 мм)',
    Math.abs(S.U - META.U) < 1e-6 && Math.abs(S.U - 5.8824) > 1e-5,
    `U = ${S.U.toFixed(6)} (в проекте ${META.U.toFixed(6)}, в main.js было бы 5.8824)`);
  const bb = g.boundingBox.clone();
  bb.applyMatrix4(S.mesh.matrixWorld);
  const c = bb.getCenter(new THREE.Vector3());
  const r = bb.getSize(vec()).length() / 2;
  const dist = S.camera.position.distanceTo(S.controls.target);
  it('5. камера смотрит в центр габарита и видит модель целиком',
    S.controls.target.distanceTo(c) < 1e-6 && dist > r && dist < r * 4,
    `центр [${c.toArray().map((v) => v.toFixed(0))}], дистанция ${dist.toFixed(0)} при радиусе ${r.toFixed(0)}`);
  it('6. модель стоит «на колёсах»: высота — вторая по длине ось',
    Math.abs(bb.getSize(vec()).y - Math.max(META.stats.bbox_mm[1], 0) * S.U) / (META.stats.bbox_mm[1] * S.U) < 0.02,
    `высота ${(bb.getSize(vec()).y / S.U).toFixed(1)} мм из проекта ${META.stats.bbox_mm[1]} мм`);
  it('7. в шапке число граней и длина в миллиметрах по масштабу проекта',
    els['sb-mesh'].textContent.replace(/[\s\u00a0]/g, '').includes(String(META.stats.faces)) &&
    els['sb-mesh'].textContent.includes(String(META.stats.bbox_mm[0])) && els['sb-mesh'].textContent.includes('мм'),
    `«${els['sb-mesh'].textContent}»`);
  it('8. без автосохранения кнопка «Вернуть правку» спрятана',
    !els['sb-restore'].style.display || els['sb-restore'].style.display === 'none',
    `display = ${els['sb-restore'].style.display}`);

  /* правка мышью */
  const cvs = S.renderer.domElement;
  const evAt = (x, y) => ({ button: 0, clientX: x, clientY: y, pointerId: 1, shiftKey: false, preventDefault() {} });
  S.camera.updateMatrixWorld(true);
  const scr = (p) => { const v = new THREE.Vector3(...p); S.mesh.localToWorld(v); v.project(S.camera);
    return [(v.x * 0.5 + 0.5) * innerWidth, (-v.y * 0.5 + 0.5) * innerHeight]; };
  const hitFromTop = (x, z) => {
    const ray = new THREE.Raycaster(new THREE.Vector3(x, 5000, z), new THREE.Vector3(0, -1, 0), 0, 20000);
    const h = ray.intersectObject(S.mesh, false);
    if (!h.length) throw new Error('мимо модели: ' + x + ',' + z);
    return h[0].point.toArray();
  };
  const pos0 = Float32Array.from(S.mesh.geometry.attributes.position.array);
  S.sculpt.setTool('select');
  els['sc-brush'].value = '12'; els['sc-brush'].fire('input');
  els['sc-depth'].value = '2'; els['sc-depth'].fire('input');
  const pts = [];
  for (let i = 0; i <= 24; i++) pts.push(hitFromTop(400, 130 + (60 * i) / 24));
  const sp = pts.map(scr);
  cvs._ev.pointerdown.forEach((f) => f(evAt(...sp[0])));
  for (let i = 1; i < sp.length; i++) cvs._ev.pointermove.forEach((f) => f(evAt(...sp[i])));
  cvs._ev.pointerup.forEach((f) => f(evAt(...sp[sp.length - 1])));
  els['sc-out'].click();                                   // «Выдавить» на 2 мм
  const pos1 = S.mesh.geometry.attributes.position.array;
  let moved = 0, maxmm = 0;
  for (let i = 0; i < pos0.length; i += 3) {
    const d = Math.hypot(pos1[i] - pos0[i], pos1[i + 1] - pos0[i + 1], pos1[i + 2] - pos0[i + 2]);
    if (d > 1e-6) { moved++; maxmm = Math.max(maxmm, d / S.U); }
  }
  it('9. правка мышью работает на загруженной модели',
    moved > 100 && maxmm > 1.5 && maxmm < 2.6 && S.sculpt.dirty === true,
    `${moved} вершин сдвинуто, максимум ${maxmm.toFixed(2)} мм (просили 2), «${plain().slice(0, 60)}»`);

  els['sb-save3mf'].click();                               // кнопка в шапке → «Сохранить» панели
  await waitFor(async () => /версия/.test(els['studio-msg'].textContent));
  const vers = (await (await realFetch(`${BASE}/api/project/${PID}/versions`)).json()).versions;
  const v3mf = vers.find((v) => v.kind === '3mf');
  const dl = await realFetch(`${BASE}/download/${PID}/${v3mf.file}`);
  const bytes = Buffer.from(await dl.arrayBuffer());
  it('10. «Сохранить 3MF» кладёт версию в проект (файл на диске, не в «Загрузки»)',
    !!v3mf && bytes.subarray(0, 2).toString('latin1') === 'PK' && bytes.length === v3mf.size,
    `${v3mf.file}, ${(bytes.length / 1048576).toFixed(2)} МБ, скачался ровно того же размера`);

  els['sb-savestl'].click();
  await waitFor(async () => {
    const v = (await (await realFetch(`${BASE}/api/project/${PID}/versions`)).json()).versions;
    return v.some((x) => x.kind === 'stl');
  });
  const vs = (await (await realFetch(`${BASE}/api/project/${PID}/versions`)).json()).versions.find((v) => v.kind === 'stl');
  const stlBytes = Buffer.from(await (await realFetch(`${BASE}/download/${PID}/${vs.file}`)).arrayBuffer());
  const stlFaces = stlBytes.readUInt32LE(80);
  it('11. «STL» — двоичный STL ровно по числу граней модели',
    stlFaces === S.sculpt.mesh.faces && stlBytes.length === 84 + 50 * stlFaces,
    `${vs.file}, ${(stlBytes.length / 1048576).toFixed(1)} МБ, ${stlFaces} граней`);

  els['sb-vers'].click();                                  // список версий в окне
  await waitFor(async () => els['studio-vers'].innerHTML.includes(v3mf.file));
  it('12. «Версии» показывают файлы с размерами и ссылками на скачивание',
    els['studio-vers'].innerHTML.includes(v3mf.file) && els['studio-vers'].innerHTML.includes('/download/') &&
    (els['studio-vers'].innerHTML.match(/class="v"/g) || []).length >= 2,
    `${(els['studio-vers'].innerHTML.match(/class="v"/g) || []).length} строки`);

  const postsBefore = NET.posts.length;
  let prevented = false;
  (globalThis._ev.keydown || []).forEach((f) => f({ key: 's', code: 'KeyS', ctrlKey: true, metaKey: false,
    preventDefault() { prevented = true; }, target: { tagName: 'BODY' } }));
  await waitFor(async () => (await realFetch(`${BASE}/api/project/${PID}/state`)).ok);
  it('13. Ctrl+S сохраняет правку в проект, а не страницу браузера',
    prevented && NET.posts.length > postsBefore && NET.posts.some((u) => u.endsWith('/state')),
    `preventDefault ${prevented}, POST ${NET.posts.slice(postsBefore).join(', ')}`);

  const savedState = await (await realFetch(`${BASE}/api/project/${PID}/state`)).json();
  it('14. состояние правки лежит в проекте и годится для восстановления',
    savedState && savedState.app === 'tayron-sculpt' && Array.isArray(savedState.verts),
    `ключи: ${Object.keys(savedState).slice(0, 5).join(', ')}`);

  els['sb-name'].ondblclick();                             // переименование двойным кликом
  await waitFor(async () => (await (await realFetch(`${BASE}/api/project/${PID}`)).json()).name === 'Кузов Tayron (из студии)');
  const renamed = await (await realFetch(`${BASE}/api/project/${PID}`)).json();
  it('15. двойной клик по названию переименовывает проект на сервере',
    renamed.name === 'Кузов Tayron (из студии)' && els['sb-name'].textContent === renamed.name,
    `«${renamed.name}»`);

  /* отмена и возврат правки */
  const jsonNow = S.sculpt.projectJSON;
  const posNow = Float32Array.from(S.mesh.geometry.attributes.position.array);
  S.sculpt.loadProject(JSON.parse(jsonNow));
  const back = S.mesh.geometry.attributes.position.array;
  let same = 0;
  for (let i = 0; i < posNow.length; i += 3) if (Math.abs(back[i] - posNow[i]) < 1e-4) same++;
  it('16. projectJSON → loadProject возвращает ту же геометрию',
    same === posNow.length / 3, `${same} из ${posNow.length / 3} вершин совпали`);

  (globalThis._ev.keydown || []).forEach((f) => f({ key: 'z', code: 'KeyZ', ctrlKey: true, metaKey: false,
    preventDefault() {}, target: { tagName: 'BODY' } }));
  const undone = S.mesh.geometry.attributes.position.array;
  let diff = 0;
  for (let i = 0; i < pos0.length; i++) if (undone[i] !== pos0[i]) diff++;
  it('17. Ctrl+Z отменяет правку байт в байт', diff === 0, `различий ${diff} из ${pos0.length} чисел`);

  it('18. тело модели скачано один раз — лишних запросов нет',
    NET.gets.filter((u) => u.endsWith('/body.json')).length === 1,
    `запросы: ${NET.gets.filter((u) => u.includes('/api/')).join(' ')}`);

  const flipNote = () => lost;
  it('19. «Перевернуть» и «Вписать» меняют вид и запоминаются за проектом',
    (() => {
      const before = S.mesh.quaternion.clone();
      els['sb-flip'].click();
      const turned = !S.mesh.quaternion.equals(before);
      const marked = S.flipped === true && localStorage.getItem('tayron-flip-' + PID) === '1';
      S.camera.position.set(0, 0, 0);
      els['sb-fit'].click();                               // «Вписать» возвращает камеру к модели
      const fitted = S.camera.position.length() > 1 &&
        S.controls.target.distanceTo(S.camera.position) > 1;
      els['sb-flip'].click();                              // обратно
      lost = `поворот ${turned}, память ${marked}, вписать ${fitted}`;
      return turned && marked && fitted;
    })(), flipNote());
}

function vec() { return new THREE.Vector3(); }

console.log(`\nИТОГ страницы студии: ${n - bad} из ${n} проверок зелёные, провалов ${bad}`);
stop();
process.exit(bad ? 1 : 0);
