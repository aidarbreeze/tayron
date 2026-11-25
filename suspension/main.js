import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/* ============================================================
   Tayron · активная подвеска на балках из пользовательского
   печатного проекта (Single_color.3mf), масштаб ×2.2
   ============================================================ */

const U        = 5.3327;   // мм на 1 единицу модели (база 2789 мм / 523 ед.)
const ARM      = 80;       // рычаг: ось качания → центр колеса, ед.
const TRAVEL   = 12.7;     // ход вверх до арки, ед. (±68 мм)
const MAXLOAD  = 200;      // кг на угол
const G        = 9.81;

// геометрия механизма: файл модели → ось качания Q и центр колеса wc
const RIG = {
  0: { Q: [-261.82,  74.92, 70.10], wc: [-261.59,  154.91, 71.45] },
  1: { Q: [ 262.90,  75.00, 69.50], wc: [ 261.40,  154.90, 71.40] },
  2: { Q: [ 261.20, -75.50, 71.10], wc: [ 261.40, -155.50, 71.40] },
  3: { Q: [-263.10, -75.50, 72.70], wc: [-261.50, -155.50, 71.50] }
};

const CORNERS = [
  { id:'FL', name:'Переднее левое',  short:'ПЛ', file:1, x: 261.4, y: 154.9 },
  { id:'FR', name:'Переднее правое', short:'ПП', file:2, x: 261.4, y:-155.5 },
  { id:'RL', name:'Заднее левое',    short:'ЗЛ', file:0, x:-261.6, y: 154.9 },
  { id:'RR', name:'Заднее правое',   short:'ЗП', file:3, x:-261.5, y:-155.5 }
];

const sumX2 = CORNERS.reduce((a,c) => a + (c.x*U)**2, 0);   // мм²
const sumY2 = CORNERS.reduce((a,c) => a + (c.y*U)**2, 0);   // мм²

/* ---------- статика: жёсткий кузов на 4 пружинах ---------- */
function solvePose(loads, k) {
  let S = 0, Sx = 0, Sy = 0;
  CORNERS.forEach((c,i) => { const F = loads[i]*G; S += F; Sx += F*c.x*U; Sy += F*c.y*U; });
  let zmm  = S / (4*k);            // мм вниз
  let roll = Sy / (k*sumY2);       // рад, + = левый борт вниз
  let pit  = Sx / (k*sumX2);       // рад, + = нос вниз
  let disp = CORNERS.map(c => (zmm + roll*c.y*U + pit*c.x*U) / U);  // в единицах, + вниз
  let maxD = Math.max(...disp.map(Math.abs));
  let bottomed = false;
  if (maxD > TRAVEL) {                       // упёрлись в арку
    const f = TRAVEL / maxD;
    zmm *= f; roll *= f; pit *= f; disp = disp.map(d => d*f);
    bottomed = true;
  }
  return { zmm, roll, pit, disp, bottomed };
}

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
  body:   new THREE.MeshStandardMaterial({ color:0xaebccb, metalness:0.28, roughness:0.38 }),
  wheel:  new THREE.MeshStandardMaterial({ color:0x1a1d21, metalness:0.15, roughness:0.72 }),
  sleeve: new THREE.MeshStandardMaterial({ color:0xffa733, metalness:0.35, roughness:0.4, emissive:0x3a1d00, emissiveIntensity:0.6 }),
  beam:   new THREE.MeshStandardMaterial({ color:0x2fd3b8, metalness:0.35, roughness:0.42, emissive:0x073a33, emissiveIntensity:0.8 })
};

const arms = [];        // { group, wheelPivot, corner }
const loadLabels = [];

/* ---------- загрузка ---------- */
const loader = new GLTFLoader();
const load = (f) => new Promise((res, rej) => loader.load(f, g => res(g.scene), undefined, rej));

(async () => {
  const [bodyScene, ...rest] = await Promise.all([
    load('./assets/body.glb'),
    ...CORNERS.flatMap(c => [
      load(`./assets/c${c.file}_sleeve.glb`),
      load(`./assets/c${c.file}_beam.glb`),
      load(`./assets/c${c.file}_wheel.glb`)
    ])
  ]);

  // кузов
  bodyScene.traverse(o => { if (o.isMesh) { o.material = MAT.body; o.castShadow = true; o.receiveShadow = true; } });
  bodyGroup.add(bodyScene);

  CORNERS.forEach((c, i) => {
    const [sleeveScene, beamScene, wheelScene] = rest.slice(i*3, i*3+3);
    const rig = RIG[c.file];

    // втулка — жёстко в кузове
    const sleeve = new THREE.Group();
    sleeve.position.fromArray(rig.Q);
    sleeveScene.traverse(o => { if (o.isMesh) { o.material = MAT.sleeve; o.castShadow = true; } });
    sleeve.add(sleeveScene);
    bodyGroup.add(sleeve);

    // рычаг: группа в оси качания, качается вокруг X
    const arm = new THREE.Group();
    arm.position.fromArray(rig.Q);
    beamScene.traverse(o => { if (o.isMesh) { o.material = MAT.beam; o.castShadow = true; } });
    arm.add(beamScene);

    // колесо — в pivot, который компенсирует качание (колесо остаётся вертикальным)
    const wpOffset = new THREE.Vector3().fromArray(rig.wc).sub(new THREE.Vector3().fromArray(rig.Q));
    const wheelPivot = new THREE.Group();
    wheelPivot.position.copy(wpOffset);
    wheelScene.traverse(o => { if (o.isMesh) { o.material = MAT.wheel; o.castShadow = true; } });
    wheelScene.position.copy(wpOffset).multiplyScalar(-1);
    wheelPivot.add(wheelScene);
    arm.add(wheelPivot);

    bodyGroup.add(arm);
    arms.push({ arm, wheelPivot, corner:c, sign: Math.sign(c.y) });
  });

  document.getElementById('loading').style.opacity = 0;
  setTimeout(() => document.getElementById('loading').remove(), 520);
  animate();
})().catch(err => {
  document.getElementById('loading').innerHTML =
    '<p style="color:#ff8a8a">Не удалось загрузить модель.<br>' + err + '</p>';
  console.error(err);
});

/* ============================ нагрузка ============================ */
const state = {
  loads: [0,0,0,0],     // целевые, кг
  cur:   [0,0,0,0],     // сглаженные
  k: 30,
  demo: false,
  xray: false
};

/* ---------- ползунки ---------- */
const slidersEl = document.getElementById('sliders');
const sliders = CORNERS.map((c, i) => {
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `<span class="tag">${c.short}</span>
    <input type="range" min="0" max="${MAXLOAD}" step="5" value="0" id="s-${c.id}">
    <span class="val" id="v-${c.id}">0 кг</span>`;
  slidersEl.appendChild(row);
  const inp = row.querySelector('input');
  const out = row.querySelector('.val');
  inp.addEventListener('input', () => {
    stopDemo();
    state.loads[i] = +inp.value;
    out.textContent = inp.value + ' кг';
  });
  return { inp, out };
});

function setLoads(arr) {
  state.loads = arr.slice();
  CORNERS.forEach((c,i) => {
    sliders[i].inp.value = Math.round(arr[i]);
    sliders[i].out.textContent = Math.round(arr[i]) + ' кг';
  });
}

function stopDemo() {
  if (!state.demo) return;
  state.demo = false;
  document.querySelector('[data-preset="demo"]').classList.remove('on');
}

/* ---------- сценарии ---------- */
const PRESETS = {
  zero:  [0,0,0,0],
  fl:    [100,0,0,0],
  roof:  [50,50,50,50],
  trunk: [0,0,90,90],
  side:  [80,0,80,0],
  diag:  [100,0,0,100]
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
  const delta = e.shiftKey ? -40 : 40;
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
    const Lv = Math.max(0, s)*110, Rv = Math.max(0, -s)*110;
    setLoads([Lv, Rv, Lv*0.8, Rv*0.8]);
  }

  // сглаживание
  for (let i = 0; i < 4; i++) {
    state.cur[i] += (state.loads[i] - state.cur[i]) * Math.min(1, dt*8);
  }

  const pose = solvePose(state.cur, state.k);

  // кузов
  bodyGroup.position.z = -pose.zmm / U;
  bodyGroup.rotation.set(-pose.roll, pose.pit, 0);

  // рычаги и колёса
  arms.forEach((a, i) => {
    const d = pose.disp[i];                        // ед., + вниз
    const alpha = a.sign * Math.asin(THREE.MathUtils.clamp(d / ARM, -0.999, 0.999));
    a.arm.rotation.x = alpha;
    a.wheelPivot.rotation.x = -alpha;              // колесо остаётся вертикальным
  });

  // телеметрия
  const rollDeg = Math.abs(pose.roll)*180/Math.PI;
  const pitDeg  = Math.abs(pose.pit)*180/Math.PI;
  rollEl.innerHTML  = rollDeg.toFixed(1) + '<small>° ' + (rollDeg < 0.05 ? '' : (pose.roll > 0 ? 'влево' : 'вправо')) + '</small>';
  pitchEl.innerHTML = pitDeg.toFixed(1)  + '<small>° ' + (pitDeg  < 0.05 ? '' : (pose.pit  > 0 ? 'на нос' : 'на корму')) + '</small>';
  sagEl.innerHTML   = Math.round(pose.zmm) + '<small>мм</small>';
  warnEl.style.display = pose.bottomed ? 'block' : 'none';

  CORNERS.forEach((c,i) => {
    const mm = pose.disp[i]*U;                     // + вниз (сжатие)
    const pct = Math.min(50, Math.abs(mm)/68*50);
    const f = bars[i].fill;
    f.style.width = pct + '%';
    if (mm >= 0) { f.style.left = '50%'; f.classList.remove('neg'); }
    else         { f.style.left = (50 - pct) + '%'; f.classList.add('neg'); }
    bars[i].mm.textContent = (mm >= 0 ? '−' : '+') + Math.abs(Math.round(mm));
    bars[i].mm.style.color = mm >= 0 ? '#8d9aac' : '#c9a06a';

    drawLabel(loadLabels[i], Math.round(state.cur[i]) + ' кг', state.cur[i] > 3);
  });

  controls.update();
  renderer.render(scene, camera);
}

addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
