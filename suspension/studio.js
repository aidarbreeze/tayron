/* Tayron Studio · редактор: та же панель и тот же sculpt.js, что и в основном
   приложении, но модель приходит из файла пользователя (`/api/project/<id>`),
   а «Сохранить» пишет версию на диск через сервер studio.py.

   Что отличается от `main.js` (там — готовая машина с колёсами и физикой):
   * нет подвески, колёс и стола — правится ровно та сетка, что загрузили;
   * масштаб печати берётся из проекта (`meta.U`), а не жёсткое 5.8824;
   * ориентация: по умолчанию Z вверх (как в 3MF/слайсерах), кнопка
     переключает Y вверх — STL из некоторых программ приходит так;
   * автосохранение состояния в проект каждые 20 с, если есть правка. */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { initSculpt } from './sculpt.js';

const $ = (id) => document.getElementById(id);
const msg = (text, ms = 4000) => {
  const el = $('studio-msg');
  if (!el) return;
  el.textContent = text;
  el.style.display = 'block';
  clearTimeout(msg._t);
  msg._t = setTimeout(() => { el.style.display = 'none'; }, ms);
};

const params = new URLSearchParams(location.search);
const PID = params.get('id') || '';
if (!/^[0-9a-f]{8,20}$/.test(PID)) {
  document.body.innerHTML = '<p style="color:#e7ecf3;font:14px sans-serif;padding:24px">' +
    'Нет проекта в адресе. <a style="color:#38e1c4" href="/">К списку моделей</a></p>';
  throw new Error('нет id проекта');
}

const api = {
  meta: () => fetch(`/api/project/${PID}`).then(r => r.json()),
  body: () => fetch(`/api/project/${PID}/body.json`).then(r => r.json()),
  state: () => fetch(`/api/project/${PID}/state`).then(r => (r.ok ? r.text() : null)),
  saveState: (text) => fetch(`/api/project/${PID}/state`, { method: 'POST', body: text })
    .then(r => (r.ok ? r.json() : Promise.reject(r.status))),
  saveFile: (kind, bytes, name) => fetch(`/api/project/${PID}/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, data: toBase64(bytes), name, note: 'из редактора' }),
  }).then(async (r) => {
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || ('HTTP ' + r.status));
    return j;
  }),
  versions: () => fetch(`/api/project/${PID}/versions`).then(r => r.json()),
  rename: (name) => fetch(`/api/project/${PID}/rename`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  }).then(r => r.json()),
};

function toBase64(bytes) {
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH));
  }
  return btoa(s);
}

/* ---------- сцена ---------- */
const view = $('view');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e13);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
view.appendChild(renderer.domElement);
renderer.domElement.style.display = 'block';

const camera = new THREE.PerspectiveCamera(36, 1, 1, 200000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.22));
const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(0.6, -0.9, 1.2).multiplyScalar(1200);
key.castShadow = true;
scene.add(key);
const fill = new THREE.DirectionalLight(0x88bbff, 0.75);
fill.position.set(-1, 0.4, 0.6).multiplyScalar(1200);
scene.add(fill);
const rim = new THREE.DirectionalLight(0xffd9a0, 0.5);
rim.position.set(0.2, 1, -0.6).multiplyScalar(1200);
scene.add(rim);

const material = new THREE.MeshStandardMaterial({
  color: 0xb9c6d4, metalness: 0.05, roughness: 0.55, side: THREE.DoubleSide,
});
const mesh = new THREE.Mesh(new THREE.BufferGeometry(), material);
mesh.castShadow = true;
mesh.receiveShadow = true;
scene.add(mesh);

const resize = () => {
  const w = view.clientWidth || innerWidth;
  const h = view.clientHeight || innerHeight;
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = '100%';
  renderer.domElement.style.height = '100%';
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
};
addEventListener('resize', resize);

/* ---------- геометрия проекта ---------- */
const meta = await api.meta();
if (meta.error) {
  msg('проект не найден: ' + meta.error);
  throw new Error(meta.error);
}
$('sb-name').textContent = meta.name || 'модель';
$('sb-name').title = (meta.source || '') + ' · ' + (meta.kind || '');

const body = await api.body();
const nVerts = body.pos.length;
const posArr = new Float32Array(nVerts * 3);
for (let i = 0; i < nVerts; i++) {
  posArr[i * 3] = body.pos[i][0];
  posArr[i * 3 + 1] = body.pos[i][1];
  posArr[i * 3 + 2] = body.pos[i][2];
}
const idxArr = new Uint32Array(body.idx);
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
geo.computeVertexNormals();
geo.computeBoundingBox();
geo.computeBoundingSphere();
mesh.geometry.dispose();
mesh.geometry = geo;

/* ---------- ориентация: файл мог прийти «лёжа» или вверх ногами ----------
   Правило простое и предсказуемое: самая длинная ось модели становится X
   (длина), средняя — Y (высота, вверх), самая короткая — Z (глубина).
   Для кузова это x/y/z как есть (сверяется с main.js), а STL/3MF из слайсера
   с осью Z вверх встаёт на колёса само. Кнопка «Перевернуть» — на случай,
   если файл пришёл вверх ногами, и она запоминается за проектом. */
const axes = ['x', 'y', 'z'].map((k, i) => ({ i, s: geo.boundingBox.getSize(new THREE.Vector3()).getComponent(i) }))
  .sort((a, b) => b.s - a.s);
const [AX_LEN, AX_UP, AX_DEP] = axes.map(a => a.i);

/* матрица перестановки осей: длина → X, высота → Y, глубина → Z */
const cols = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
cols[AX_LEN].set(1, 0, 0);
cols[AX_UP].set(0, 1, 0);
cols[AX_DEP].set(0, 0, 1);
let basis = new THREE.Matrix4().makeBasis(cols[0], cols[1], cols[2]);
if (basis.determinant() < 0) {           // зеркалить нельзя — меняем знак глубины
  cols[AX_DEP].set(0, 0, -1);
  basis = new THREE.Matrix4().makeBasis(cols[0], cols[1], cols[2]);
}
const flipKey = 'tayron-flip-' + PID;
let flipped = localStorage.getItem(flipKey) === '1';
const applyOrientation = () => {
  const m = basis.clone();
  if (flipped) m.premultiply(new THREE.Matrix4().makeRotationX(Math.PI));
  mesh.quaternion.setFromRotationMatrix(m);
  mesh.position.set(0, 0, 0);
  mesh.updateMatrixWorld(true);
};

const U = body.U || meta.U || 1;                 // единиц модели на 1 мм печати
const worldBox = () => new THREE.Box3().setFromObject(mesh);

const frames = () => {                          // поставить камеру по габариту
  const box = worldBox();
  const c = box.getCenter(new THREE.Vector3());
  const r = box.getSize(new THREE.Vector3()).length() / 2 || 1;
  const dir = new THREE.Vector3(0.78, 0.46, 0.62).normalize();
  const d = r / Math.sin((camera.fov * Math.PI) / 360) * 1.06;   // 1.06 — запас по краям
  camera.position.copy(c).addScaledVector(dir, d);
  camera.near = Math.max(0.01, r / 2000);
  camera.far = d * 12;
  camera.updateProjectionMatrix();
  controls.target.copy(c);
  controls.update();
  camera.updateMatrixWorld(true);          // чтобы первый же луч попал туда, куда смотрит камера
  return { center: c, radius: r, dist: d };
};

resize();
applyOrientation();
const fit = frames();

/* ---------- редактор ---------- */
const sculpt = initSculpt({
  mesh, material, camera, renderer, controls, U,
  /* файл уходит на диск студии, а не в «Загрузки» браузера */
  saveBytes: async (kind, bytes, name) => {
    msg(`собираю ${kind.toUpperCase()} · ${(bytes.length / 1048576).toFixed(1)} МБ…`, 60000);
    try {
      const res = await api.saveFile(kind, bytes, name);
      const ver = res.version || {};
      msg(`${name}: версия ${ver.file || '?'} сохранена (${((ver.size || bytes.length) / 1048576).toFixed(1)} МБ)`);
      refreshVersions();
      return `${name} → проект, версия ${ver.file || ''}`;
    } catch (e) {
      msg('не сохранилось: ' + e.message, 8000);
      throw e;
    }
  },
});

/* ---------- верхняя полоса ---------- */
const fmtMB = (n) => (n / 1048576).toFixed(1) + ' МБ';
const meshPill = () => {
  const faces = sculpt.mesh.faces;
  const sz = worldBox().getSize(new THREE.Vector3());
  const longest = Math.max(sz.x, sz.y, sz.z) / U;          // самое длинное ребро габарита, мм
  $('sb-mesh').textContent = `${faces.toLocaleString('ru-RU')} граней · ` +
    `${longest >= 10 ? longest.toFixed(0) : longest.toFixed(1)} мм`;
  $('sb-mesh').className = 'pill';
};
meshPill();

const statePill = (text, cls) => {
  const el = $('sb-state');
  el.textContent = text;
  el.className = 'pill ' + (cls || '');
};

$('sb-fit').onclick = () => { frames(); msg('вид вписан по габариту'); };
$('sb-flip').onclick = () => {
  flipped = !flipped;
  localStorage.setItem(flipKey, flipped ? '1' : '0');
  applyOrientation();
  frames();
  msg(flipped ? 'модель перевёрнута' : 'модель в исходной ориентации');
};

$('sb-save3mf').onclick = () => $('sc-save').click();
$('sb-savestl').onclick = () => $('sc-stl').click();
$('sb-saveproj').onclick = async () => {
  try {
    await api.saveState(sculpt.projectJSON);
    statePill('правка сохранена в проект', 'ok');
    msg('состояние правки сохранено на сервер — можно закрывать вкладку');
  } catch (e) {
    statePill('не сохранилось', 'warn');
    msg('не сохранилось: ' + e, 8000);
  }
};

/* версии */
async function refreshVersions() {
  const box = $('studio-vers');
  if (box.style.display !== 'block') return;
  const { versions = [] } = await api.versions();
  box.innerHTML = '<h4>Версии проекта</h4>' + (versions.length
    ? versions.slice().reverse().map(v => `<div class="v"><a href="/download/${PID}/${encodeURIComponent(v.file)}">${v.file}</a>` +
      `<span>${fmtMB(v.size)}</span></div>`).join('')
    : '<div class="v">пока ни одной — нажмите «Сохранить 3MF»</div>');
}
$('sb-vers').onclick = async () => {
  const box = $('studio-vers');
  box.style.display = box.style.display === 'block' ? 'none' : 'block';
  if (box.style.display === 'block') await refreshVersions();
};

/* вернуть автосохранение */
const saved = await api.state().catch(() => null);
if (saved) {
  const btn = $('sb-restore');
  btn.style.display = '';
  btn.onclick = () => {
    try {
      const ok = sculpt.loadProject(saved);
      if (ok) {
        msg('правка восстановлена из автосохранения');
        btn.style.display = 'none';
        meshPill();
      } else msg('автосохранение не подошло к этой модели', 6000);
    } catch (e) { msg('автосохранение битое: ' + e, 6000); }
  };
  statePill('есть автосохранение', 'warn');
}

/* автосохранение: раз в 20 с, только если правка есть и что-то менялось */
let lastState = sculpt.projectJSON;
setInterval(async () => {
  if (!sculpt.dirty) { statePill('правка не сохранена', ''); return; }
  const json = sculpt.projectJSON;
  if (json === lastState) return;
  try {
    await api.saveState(json);
    lastState = json;
    statePill('автосохранено ' + new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }), 'ok');
  } catch (e) {
    statePill('автосохранение не прошло', 'warn');
  }
}, 20000);

/* Ctrl+S — сохранить состояние на сервер (в браузере это «сохранить страницу») */
addEventListener('keydown', (e) => {
  if (!(e.ctrlKey || e.metaKey)) {                       // F — вписать вид
    if ((e.key || '').toLowerCase() === 'f' && !e.target?.value) { frames(); }
    return;
  }
  if (e.code !== 'KeyS' && (e.key || '').toLowerCase() !== 's') return;
  e.preventDefault();
  $('sb-saveproj').click();
}, true);

/* Ctrl+Z отдан sculpt.js; здесь только подсказка про версии */
$('sb-name').ondblclick = async () => {
  const name = prompt('Название модели:', meta.name);
  if (!name) return;
  const res = await api.rename(name);
  if (res && res.meta) { $('sb-name').textContent = res.meta.name; msg('переименовано: ' + res.meta.name); }
};

/* ---------- цикл отрисовки ---------- */
let lastT = performance.now();
function tick(t) {
  const dt = Math.min(0.05, (t - lastT) / 1000);
  lastT = t;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

/* наружу — для отладки в консоли браузера */
globalThis.studio = { sculpt, scene, camera, controls, renderer, mesh, material, api, PID, meta, U,
  frames, applyOrientation, fit, get flipped() { return flipped; } };
