import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/* ============================================================
   Tayron · подвеска без пружин: S-переходник гнётся в арке, на его конце
   стакан Ø7.1 сидит на втулке колеса Ø6 (натяг), колесо крутится на втулке.
   Масштаб печати: 150 мм по кузову, U = 5.8824 ед./мм
   ============================================================ */

const U        = 5.8824;   // ед. модели на 1 мм печати (печать 17 %)
const TRAVEL   = 1.4;      // ход до отбойника, мм печати
const MAXLOAD  = 200;      // г на угол: при ~90 г угол садится на отбойник
const G        = 9.81;

// геометрия механизма: файл модели → ось качания Q и центр колеса wc
const RIG = {
  0: { wc: [-261.59, 154.91, 71.45], hub: 149.32, root: [-261.59, 64.05, 71.45] },
  1: { wc: [261.41, 154.91, 71.40], hub: 149.32, root: [261.41, 64.05, 71.40] },
  2: { wc: [261.45, -155.55, 71.45], hub: -150.60, root: [261.45, -65.33, 71.45] },
  3: { wc: [-261.47, -155.49, 71.46], hub: -150.64, root: [-261.47, -65.37, 71.46] }
};

const CORNERS = [
  { id:'FL', name:'Переднее левое',  short:'ПЛ', file:1, x: 261.4, y: 154.9 },
  { id:'FR', name:'Переднее правое', short:'ПП', file:2, x: 261.4, y:-155.5 },
  { id:'RL', name:'Заднее левое',    short:'ЗЛ', file:0, x:-261.6, y: 154.9 },
  { id:'RR', name:'Заднее правое',   short:'ЗП', file:3, x:-261.5, y:-155.5 }
];
CORNERS.forEach(c => { c.dx = c.x / U; c.dy = c.y / U; });   // плечи в мм печати

/* ============================================================
   Физика: жёсткий кузов на 4 пружинах, с гравитацией и демпфером.
   Всё в единицах модели (1 ед. = 1 мм печати), время в секундах,
   масса в кг. Поэтому 1 Н/мм = 1000 кг/с², g = 9810 мм/с².
   Кузов имеет 3 степени свободы: просадка z (вниз +), крен, тангаж.
   ============================================================ */
let   MASS   = 0.10;           // кг, напечатанная модель (~100 г при 20 %)
const G_MM   = 9810;           // мм/с²
let   ZETA   = 0.22;           // демпфирование (доля критического)
const HARD   = 60;             // жёсткость отбойника, ×k
const HALF_TRACK = 154.9 / U;   // половина колеи, мм печати (~26 мм)
const HALF_BASE  = 261.5 / U;   // половина базы, мм печати (~44 мм)
let   IROLL  = MASS * HALF_TRACK * HALF_TRACK;   // кг·мм²
let   IPITCH = MASS * HALF_BASE  * HALF_BASE;    // кг·мм²
function setMass(m) { MASS = m; IROLL = m*HALF_TRACK*HALF_TRACK; IPITCH = m*HALF_BASE*HALF_BASE; }

const phys = { z:0, vz:0, roll:0, vroll:0, pit:0, vpit:0 };
let physTime = 0;

function stepPhys(dt, loads, k) {
  const kk = Math.max(0.05, k) * 1000;               // кг/с² на угол
  const cc = 2 * ZETA * Math.sqrt(kk * MASS);        // кг/с
  const s0 = MASS * G_MM / (4 * kk);                 // статическое сжатие, мм
  const sub = 12, h = dt / sub;
  for (let n = 0; n < sub; n++) {
    let Fz = 0, Ty = 0, Tx = 0, Lsum = 0, bot = false;
    const comp = [0,0,0,0];
    CORNERS.forEach((c, i) => {
      const zc = phys.z + phys.roll*c.dy + phys.pit*c.dx;    // + вниз
      const vc = phys.vz + phys.vroll*c.dy + phys.vpit*c.dx;
      let sg = s0 + zc;                                       // сжатие, мм
      let F = 0;
      if (sg > 0) F = kk*sg + cc*vc;                          // пружина + демпфер
      if (sg > TRAVEL) F += HARD*kk*(sg - TRAVEL);            // отбойник
      const L = loads[i]/1000 * G_MM;                         // вес нагрузки, кг·мм/с²
      comp[i] = Math.min(TRAVEL, Math.max(0, sg)) - s0;
      if (sg >= TRAVEL - 0.02) bot = true;
      Fz += F; Lsum += L;
      Tx += (L - F) * c.dy;                                   // момент крена
      Ty += (L - F) * c.dx;                                   // момент тангажа
    });
    phys.vz    += (G_MM + (Lsum - Fz)/MASS) * h;
    phys.vroll += (Tx / IROLL) * h;
    phys.vpit  += (Ty / IPITCH) * h;
    phys.z     += phys.vz * h;
    phys.roll  += phys.vroll * h;
    phys.pit   += phys.vpit * h;
    phys.z     = Math.max(-40, Math.min(TRAVEL + 6, phys.z));
    phys.roll  = Math.max(-0.2, Math.min(0.2, phys.roll));
    phys.pit   = Math.max(-0.2, Math.min(0.2, phys.pit));
    phys._comp = comp; phys._bot = bot;
  }
  physTime += dt;
  const comp = phys._comp || [0,0,0,0];
  return {
    zmm: phys.z,                       // мм (= ед. модели), + вниз
    roll: phys.roll, pit: phys.pit,
    disp: comp,                        // сжатие каждого угла относительно статики
    bottomed: !!phys._bot
  };
}

function kick(vz) { phys.vz += vz; }

/* ============================ сцена ============================ */
const host = document.getElementById('view');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0e13);
scene.fog = new THREE.Fog(0x0b0e13, 2400, 5600);

const camera = new THREE.PerspectiveCamera(36, innerWidth/innerHeight, 5, 9000);
const CAM_HOME = new THREE.Vector3(980, 560, 1180);
const TARGET_HOME = new THREE.Vector3(0, 125, 0);
camera.position.copy(CAM_HOME);

const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
host.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(TARGET_HOME);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 620;
controls.maxDistance = 3200;
controls.maxPolarAngle = Math.PI*0.495;
controls.update();

/* ---------- свет ---------- */
scene.add(new THREE.HemisphereLight(0xc4dcff, 0x1a1712, 0.85));
scene.add(new THREE.AmbientLight(0xffffff, 0.22));

const key = new THREE.DirectionalLight(0xffffff, 2.1);
key.position.set(760, 1250, 900);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
const sc = key.shadow.camera;
sc.left = -900; sc.right = 900; sc.top = 900; sc.bottom = -900; sc.near = 300; sc.far = 4000;
key.shadow.bias = -0.0012;
key.shadow.normalBias = 1.2;
scene.add(key);

const fill = new THREE.DirectionalLight(0x88bbff, 0.75);
fill.position.set(-900, 620, -700);
scene.add(fill);

const rim = new THREE.DirectionalLight(0xffd9a0, 0.5);
rim.position.set(-400, 300, 1100);
scene.add(rim);

/* ---------- пол ---------- */
const ground = new THREE.Mesh(
  new THREE.CircleGeometry(2100, 64),
  new THREE.MeshStandardMaterial({ color:0x11161d, roughness:0.95, metalness:0.0 })
);
ground.rotation.x = -Math.PI/2;
ground.receiveShadow = true;
scene.add(ground);

const grid = new THREE.GridHelper(2400, 48, 0x1e2a36, 0x141c24);
grid.position.y = 0.6;
grid.material.transparent = true;
grid.material.opacity = 0.55;
scene.add(grid);

/* ---------- контейнер модели (модель Z-вверх) ---------- */
const carRoot = new THREE.Group();
carRoot.rotation.x = -Math.PI/2;
scene.add(carRoot);

const bodyGroup = new THREE.Group();     // сюда применяется крен/тангаж/просадка
carRoot.add(bodyGroup);

const MAT = {
  body:   new THREE.MeshStandardMaterial({ color:0xb9c6d4, metalness:0.05, roughness:0.52 }),
  wheel:  new THREE.MeshStandardMaterial({ color:0x2a2e34, metalness:0.03, roughness:0.88 }),
  sleeve: new THREE.MeshStandardMaterial({ color:0xf0a13c, metalness:0.05, roughness:0.55, emissive:0x1d0f00, emissiveIntensity:0.35 }),
  beam:   new THREE.MeshStandardMaterial({ color:0x36c9ad, metalness:0.05, roughness:0.55, emissive:0x06302a, emissiveIntensity:0.5 }),
  link:   new THREE.MeshStandardMaterial({ color:0xf0a13c, metalness:0.05, roughness:0.5, emissive:0x241000, emissiveIntensity:0.4 }),
  bush:   new THREE.MeshStandardMaterial({ color:0xd9d2c4, metalness:0.18, roughness:0.45, emissive:0x14120e, emissiveIntensity:0.5 })
};

const arms = [];        // { group, wheelPivot, corner }
const loadLabels = [];

/* ---------- загрузка ---------- */
const loader = new GLTFLoader();
/* GLB из trimesh приходит без NORMAL (только POSITION) — без нормалей
   MeshStandardMaterial свет не считает и модель выглядит чёрной.
   Поэтому нормали считаем здесь и включаем плоское затенение. */
function fixMesh(o) {
  if (!o.isMesh) return;
  o.geometry.computeVertexNormals();
  o.geometry.computeBoundingSphere();
  o.castShadow = true;
  o.receiveShadow = true;
  const m = o.material;
  if (m) { m.flatShading = true; m.side = THREE.DoubleSide; m.needsUpdate = true; }
}
const load = (f) => new Promise((res, rej) => loader.load(f, g => {
  g.scene.traverse(fixMesh);
  res(g.scene);
}, undefined, rej));

(async () => {
  const [bodyScene, ...rest] = await Promise.all([
    load('./assets/body.glb'),
    ...CORNERS.flatMap(c => [
      load(`./assets/c${c.file}_link.glb`),
      load(`./assets/c${c.file}_wheel.glb`)
    ])
  ]);

  bodyScene.traverse(o => { if (o.isMesh) { o.material = MAT.body; o.castShadow = true; o.receiveShadow = true; } });
  bodyGroup.add(bodyScene);

  CORNERS.forEach((c, i) => {
    const [linkScene, wheelScene] = rest.slice(i*2, i*2+2);
    const rig = RIG[c.file];

    // S-переходник: закреплён в кузове, гнётся при ходе колеса
    linkScene.traverse(o => { if (o.isMesh) { o.material = MAT.link; o.castShadow = true; } });
    const linkGroup = new THREE.Group();
    linkGroup.add(linkScene);
    bodyGroup.add(linkGroup);

    // колесо и втулка лежат в одном GLB: втулка — та сетка, где меньше треугольников
    const partsW = [];
    wheelScene.traverse(o => { if (o.isMesh) partsW.push(o); });
    partsW.sort((a, b) => a.geometry.attributes.position.count - b.geometry.attributes.position.count);
    const bushMesh = partsW[0], wheelMesh = partsW[1];

    // колесо: своя группа, ходит вертикально относительно кузова и крутится вокруг оси Y
    const wheelGroup = new THREE.Group();
    if (wheelMesh) {
      wheelMesh.material = MAT.wheel; wheelMesh.castShadow = true;
      wheelMesh.position.fromArray(rig.wc).multiplyScalar(-1);  // меш экспортирован в абсолютных координатах
      wheelGroup.add(wheelMesh);
    }
    bodyGroup.add(wheelGroup);

    // втулка: вклеена в колесо, при разборке выходит первой — отдельным шагом
    const bushGroup = new THREE.Group();
    if (bushMesh) {
      bushMesh.material = MAT.bush; bushMesh.castShadow = true;
      bushMesh.position.fromArray(rig.wc).multiplyScalar(-1);
      bushGroup.add(bushMesh);
    }
    bushGroup.position.fromArray(rig.wc);
    bodyGroup.add(bushGroup);

    // профиль изгиба ленты: f — доля пути от корня к ступице
    const root = new THREE.Vector3().fromArray(rig.root);
    const tip  = new THREE.Vector3().fromArray(rig.wc);
    const dir  = tip.clone().sub(root); const L = dir.length(); dir.normalize();
    const meshes = [];
    linkScene.traverse(o => {
      if (!o.isMesh) return;
      const g = o.geometry;
      const orig = g.attributes.position.array.slice();
      const n = g.attributes.position.count;
      const f = new Float32Array(n);
      const v = new THREE.Vector3();
      for (let k = 0; k < n; k++) {
        v.fromArray(orig, k*3).sub(root);
        const t = THREE.MathUtils.clamp(v.dot(dir) / L, 0, 1);
        f[k] = (3*t*t - t*t*t) / 2;          // прогиб консоли под концевой силой
      }
      g.userData = { orig, f };
      meshes.push(g);
    });

    // базовая позиция колеса — в абсолютных координатах модели
    wheelGroup.position.fromArray(rig.wc);
    arms.push({ linkGroup, meshes, wheelGroup, bushGroup, rig, corner: c, signY: Math.sign(c.y), spin: 0 });
  });

  document.getElementById('loading').style.opacity = 0;
  setTimeout(() => document.getElementById('loading').remove(), 520);
  animate();
})().catch(err => {
  document.getElementById('loading').innerHTML =
    '<p style="color:#ff8a8a">Не удалось загрузить модель.<br>' + err + '</p>';
  console.error(err);
});

/* ======================= сборка / разборка ======================= */
const exp = { v: 0, target: 0 };
const smooth = x => (x <= 0 ? 0 : x >= 1 ? 1 : x*x*(3 - 2*x));
const stg = (v, a, b) => smooth((v - a) / (b - a));

/* ============================ нагрузка ============================ */
const state = {
  loads: [0,0,0,0],     // целевые, г
  cur:   [0,0,0,0],     // сглаженные
  k: 0.63,
  demo: false,
  xray: false,
  spin: false
};

/* ---------- ползунки ---------- */
const slidersEl = document.getElementById('sliders');
const sliders = CORNERS.map((c, i) => {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `<span class="tag">${c.short}</span>
    <input type="range" min="0" max="${MAXLOAD}" step="10" value="0" id="s-${c.id}">
    <span class="val" id="v-${c.id}">0 г</span>`;
  slidersEl.appendChild(row);
  const inp = row.querySelector('input');
  const out = row.querySelector('.val');
  inp.addEventListener('input', () => {
    stopDemo();
    state.loads[i] = +inp.value;
    out.textContent = inp.value + ' г';
  });
  return { inp, out };
});

function setLoads(arr) {
  state.loads = arr.slice();
  CORNERS.forEach((c,i) => {
    sliders[i].inp.value = Math.round(arr[i]);
    sliders[i].out.textContent = Math.round(arr[i]) + ' г';
  });
}

function stopDemo() {
  if (!state.demo) return;
  state.demo = false;
  document.querySelector('[data-preset="demo"]').classList.remove('on');
}

/* ---------- сценарии ---------- */
const PRESETS = {          // граммы на угол
  zero:  [0,0,0,0],
  fl:    [60,0,0,0],          // 60 г на переднее левое
  roof:  [80,80,80,80],       // 320 г на крышу (на грани пробоя)
  trunk: [0,0,50,50],         // 100 г в багажник
  side:  [60,0,60,0],         // 120 г на левый борт
  diag:  [60,0,0,60]          // по диагонали
};
document.querySelectorAll('[data-preset]').forEach(b => {
  b.addEventListener('click', () => {
    const p = b.dataset.preset;
    document.querySelectorAll('[data-preset]').forEach(x => x.classList.remove('on'));
    if (p === 'demo') {
      stopDemo();
      state.demo = true;
      b.classList.add('on');
      return;
    }
    stopDemo();
    b.classList.add('on');
    setLoads(PRESETS[p]);
  });
});

const exBtn = document.getElementById('explode');
if (exBtn) exBtn.addEventListener('click', () => {
  exp.target = exp.target > 0.5 ? 0 : 1;
  exBtn.textContent = exp.target > 0.5 ? '◀ Собрать' : '▶ Разобрать';
});
const exSlider = document.getElementById('ex');
if (exSlider) exSlider.addEventListener('input', e => {
  exp.target = +e.target.value;
  if (exBtn) exBtn.textContent = exp.target > 0.5 ? '◀ Собрать' : '▶ Разобрать';
});
const shakeBtn = document.getElementById('shake');
if (shakeBtn) shakeBtn.addEventListener('click', () => { kick(520); });
const spinBtn = document.getElementById('spin');
if (spinBtn) spinBtn.addEventListener('click', () => {
  state.spin = !state.spin;
  spinBtn.classList.toggle('on', state.spin);
  spinBtn.textContent = state.spin ? '⏸ Остановить колёса' : '▶ Крутить колёса';
});
const massEl = document.getElementById('mass');
if (massEl) massEl.addEventListener('input', e => {
  setMass(+e.target.value);
  document.getElementById('mv').textContent = (+e.target.value).toFixed(1) + ' кг';
});
const dampEl = document.getElementById('damp');
if (dampEl) dampEl.addEventListener('input', e => {
  ZETA = +e.target.value;
  document.getElementById('dv').textContent = (+e.target.value).toFixed(2);
});
document.getElementById('k').addEventListener('input', e => {
  state.k = +e.target.value;
  document.getElementById('kv').textContent = state.k + ' Н/мм';
});
document.getElementById('reset').addEventListener('click', () => {
  camera.position.copy(CAM_HOME);
  controls.target.copy(TARGET_HOME);
  controls.update();
});
document.getElementById('xray').addEventListener('click', e => {
  state.xray = !state.xray;
  e.currentTarget.classList.toggle('on', state.xray);
  MAT.body.transparent = state.xray;
  MAT.body.opacity = state.xray ? 0.16 : 1;
  MAT.body.depthWrite = !state.xray;
  MAT.body.needsUpdate = true;
});

/* ---------- клик по кузову ---------- */
const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downXY = null;

renderer.domElement.addEventListener('pointerdown', e => { downXY = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', e => {
  if (!downXY) return;
  const moved = Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]);
  downXY = null;
  if (moved > 6) return;               // это было вращение камеры
  if (!bodyGroup.children.length) return;

  ndc.set((e.clientX/innerWidth)*2 - 1, -(e.clientY/innerHeight)*2 + 1);
  ray.setFromCamera(ndc, camera);
  const hits = ray.intersectObject(bodyGroup, true);
  if (!hits.length) return;

  const p = carRoot.worldToLocal(hits[0].point.clone());
  let best = 0, bd = Infinity;
  CORNERS.forEach((c,i) => {
    const d = (p.x - c.x)**2 + (p.y - c.y)**2;
    if (d < bd) { bd = d; best = i; }
  });
  stopDemo();
  const delta = e.shiftKey ? -250 : 250;
  state.loads[best] = Math.min(MAXLOAD, Math.max(0, state.loads[best] + delta));
  setLoads(state.loads);
});

/* ---------- подписи нагрузки ---------- */
function makeLabel() {
  const cv = document.createElement('canvas');
  cv.width = 256; cv.height = 96;
  const tex = new THREE.CanvasTexture(cv);
  tex.minFilter = THREE.LinearFilter;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map:tex, transparent:true, depthWrite:false }));
  spr.scale.set(150, 56, 1);
  spr.center.set(0.5, 0);
  carRoot.add(spr);
  return { cv, tex, spr, last:null };
}
CORNERS.forEach(c => {
  const l = makeLabel();
  l.spr.position.set(c.x, c.y, 385);
  loadLabels.push(l);
});
function drawLabel(l, text, active) {
  if (l.last === text + active) return;
  l.last = text + active;
  const { cv, tex } = l;
  const g = cv.getContext('2d');
  g.clearRect(0, 0, cv.width, cv.height);
  g.font = '600 44px -apple-system,Segoe UI,Roboto,sans-serif';
  g.textAlign = 'center'; g.textBaseline = 'middle';
  g.fillStyle = active ? '#ffb454' : 'rgba(180,195,210,.35)';
  g.fillText(text, cv.width/2, cv.height/2 + 4);
  tex.needsUpdate = true;
  l.spr.material.opacity = active ? 1 : 0.45;
}

/* ============================ анимация ============================ */
const clock = new THREE.Clock();
const rollEl = document.getElementById('roll');
const pitchEl = document.getElementById('pitch');
const sagEl = document.getElementById('sag');
const warnEl = document.getElementById('warn');

const barsEl = document.getElementById('bars');
const bars = CORNERS.map(c => {
  const d = document.createElement('div');
  d.className = 'bar';
  d.innerHTML = `<span class="n">${c.short}</span><span class="t"><i></i></span><span class="mm">0</span>`;
  barsEl.appendChild(d);
  return { fill: d.querySelector('i'), mm: d.querySelector('.mm') };
});

let t0 = 0;

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);
  t0 += dt;

  // демо-режим: перекачка нагрузки влево-вправо
  if (state.demo) {
    const s = Math.sin(t0*1.1);
    const Lv = Math.max(0, s)*70, Rv = Math.max(0, -s)*70;
    setLoads([Lv, Rv, Lv*0.8, Rv*0.8]);
  }

  // сглаживание
  for (let i = 0; i < 4; i++) {
    state.cur[i] += (state.loads[i] - state.cur[i]) * Math.min(1, dt*8);
  }

  exp.v += (exp.target - exp.v) * Math.min(1, dt*5.5);
  const exv = exp.v;
  arms.forEach(a => {
    const sb = stg(exv, 0.30, 0.62);   // втулка — своим шагом, выпадает из паза вниз
    const sw = stg(exv, 0.00, 0.40);   // колесо
    const sL = stg(exv, 0.40, 0.82);   // S-переходник
    a.bushGroup.position.set(a.rig.wc[0], a.rig.wc[1] - a.signY*22*sb, a.rig.wc[2] - 130*sb);
    a.wheelGroup.position.set(a.rig.wc[0], a.rig.wc[1] + a.signY*330*sw, a.rig.wc[2] + 30*sw);
    a.linkGroup.position.set(0, a.signY*150*sL, 40*sL);
  });
  const pose = (exv > 0.03)
    ? { zmm:0, roll:0, pit:0, disp:[0,0,0,0], bottomed:false }
    : stepPhys(Math.min(dt, 0.033), state.cur, state.k);

  // кузов
  bodyGroup.position.z = -pose.zmm * U;   // мм печати → единицы модели
  bodyGroup.rotation.set(-pose.roll, pose.pit, 0);

  // колёса ходят вертикально относительно кузова, S-лента при этом гнётся
  arms.forEach((a, i) => {
    const d = pose.disp[i] * U;                    // ед. модели, + вниз (сжатие)
    const exB = stg(exv, 0.30, 0.62), ex = stg(exv, 0.00, 0.40), exL = stg(exv, 0.40, 0.82);
    a.bushGroup.position.set(a.rig.wc[0], a.rig.wc[1] - a.signY*22*exB, a.rig.wc[2] - 130*exB);
    a.wheelGroup.position.set(a.rig.wc[0],
                              a.rig.wc[1] + a.signY*330*ex,
                              a.rig.wc[2] + 30*ex + d);
    a.linkGroup.position.set(0, a.signY*150*exL, 40*exL);
    if (state.spin) a.spin += dt * 6.0;                       // колесо крутится вокруг своей оси
    a.wheelGroup.rotation.y = a.spin;
    a.meshes.forEach(g => {
      const { orig, f } = g.userData;
      const arr = g.attributes.position.array;
      for (let k = 0; k < f.length; k++) arr[k*3+2] = orig[k*3+2] + d * f[k];
      g.attributes.position.needsUpdate = true;
      g.computeVertexNormals();
      g.computeBoundingSphere();
    });
  });

  // телеметрия
  const rollDeg = Math.abs(pose.roll)*180/Math.PI;
  const pitDeg  = Math.abs(pose.pit)*180/Math.PI;
  rollEl.innerHTML  = rollDeg.toFixed(1) + '<small>° ' + (rollDeg < 0.05 ? '' : (pose.roll > 0 ? 'влево' : 'вправо')) + '</small>';
  pitchEl.innerHTML = pitDeg.toFixed(1)  + '<small>° ' + (pitDeg  < 0.05 ? '' : (pose.pit  > 0 ? 'на нос' : 'на корму')) + '</small>';
  sagEl.innerHTML   = pose.zmm.toFixed(1) + '<small>мм</small>';
  warnEl.style.display = pose.bottomed ? 'block' : 'none';

  CORNERS.forEach((c,i) => {
    const mm = pose.disp[i];                       // мм, + вниз (сжатие)
    const pct = Math.min(50, Math.abs(mm)/TRAVEL*50);
    const f = bars[i].fill;
    f.style.width = pct + '%';
    if (mm >= 0) { f.style.left = '50%'; f.classList.remove('neg'); }
    else         { f.style.left = (50 - pct) + '%'; f.classList.add('neg'); }
    bars[i].mm.textContent = (mm >= 0 ? '+' : '−') + Math.abs(mm).toFixed(1);
    bars[i].mm.style.color = mm >= 0 ? '#8d9aac' : '#c9a06a';

    drawLabel(loadLabels[i], Math.round(state.cur[i]) + ' г', state.cur[i] > 3);
  });

  if (exSlider && document.activeElement !== exSlider && Math.abs(+exSlider.value - exp.v) > 0.01) {
    exSlider.value = exp.v.toFixed(2);
    document.getElementById('exv').textContent = Math.round(exp.v*100) + '%';
  }
  controls.update();
  renderer.render(scene, camera);
}

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
