/* ============================================================
   Редактор кузова на сайте.

   Инструменты (все двигают только существующие вершины):
   1. ✎ Участок — кисть закрашивает область (бирюзовым), дальше:
        «≡ Сгладить» (диффузия по рёбрам), «▭ Выровнять» (область садится в
        плоскость), «✦ Обострить» (обратное сглаживанию), «⬆ Выдавить» /
        «⬇ Вдавить» (сдвиг вдоль нормалей).
   2. ✏ Рисовать — перо: валик по штриху (Shift — канавка).
   3. ◆ Фигура — вдавливает в кузов сферу / цилиндр / куб (Shift — купол).
   4. ⌫ Ластик — удаляет точки (подсвечены красными точками), дырка зашивается
      треугольниками по своей кромке.
   5. 📏 Измерить — две точки на поверхности, расстояние в мм.

   ⇄ Симметрия повторяет правку зеркально по плоскости y = 0.
   ✓ Проверка — замкнутость, дыры, неманифолдные рёбра, объём, габариты.
   «Сетка» — триангуляция тонкими чёрными линиями, «Сечение» — разрез.

   История: 15 шагов, «↶ Отменить» / «↷ Вернуть» (Ctrl+Z / Ctrl+Shift+Z).
   Выгрузка: 3MF и STL прямо из браузера.

   Всё в единицах модели (1 ед. = 0.17 мм печати, U = 5.8824 ед./мм).
   ============================================================ */
import * as THREE from 'three';
import { meshTo3mf, meshToSTL, downloadBytes } from './export3mf.js';
import {
  buildTopology, freeMask, smoothDiffuse, boundaryFalloff, paintBall,
  expandToVertices, deleteAndFill, compactMesh, vertexNormals,
  pushBall, offsetRegion, imprintShape, flattenRegion, sharpenRegion,
  mirrorDeltas, checkMesh
} from './meshops.js';

const MAXSTEPS = 600;   // итераций за одно нажатие при «Степень» 100 %
const LAMBDA   = 0.5;   // доля шага диффузии
const SHAPE_NAMES = { sphere: 'сфера', cylinder: 'цилиндр', box: 'куб' };

export function initSculpt(opts) {
  const { mesh, material, camera, renderer, controls, U } = opts;
  const geom = mesh.geometry;
  let attr = geom.attributes.position;
  let count = attr.count;
  if (!geom.index) {
    const idx = new Uint32Array(count);
    for (let i = 0; i < count; i++) idx[i] = i;
    geom.setIndex(new THREE.BufferAttribute(idx, 1));
  }
  let src = attr.array;                   // координаты вершин геометрии (как в файле)
  const origIndex = Int32Array.from(geom.index.array);  // индексы исходной сетки
  let liveIndex = origIndex.slice();                    // текущие индексы
  let topo = buildTopology(src, liveIndex);
  let vk = topo.vk, pos0 = topo.pos0, canon = topo.canon;
  let topoDirty = false;                  // топология правилась (добавлены точки)

  const posInit = Float32Array.from(pos0);   // исходные координаты — для «Сбросить»
  let pos = Float32Array.from(pos0);
  let reg = new Uint8Array(vk);
  let dead = new Uint8Array(vk);          // узлы, убранные ластиком
  let pending = new Uint8Array(vk);       // узлы текущего штриха (для красных точек)
  let front = new Uint8Array(vk);         // «лицевые» узлы (видны с камеры)
  let added = new Uint8Array(vk);         // узлы, добавленные инструментом «Точка»
  let delTotal = 0;
  let free = new Uint8Array(vk);
  let regionN = 0;
  let nrm = null;                         // нормали узлов, считаются по требованию

  /* ---------- подсветка области ---------- */
  const origSrc = Float32Array.from(src);   // исходная геометрия — для «Сбросить»
  const origCount = count;
  let col = new Float32Array(count * 3).fill(1);
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  material.vertexColors = true;
  material.needsUpdate = true;
  let CANON_COL = new Float32Array(vk * 3).fill(1);
  function repaintColors() {
    for (let i = 0; i < count; i++) {
      const c = canon[i] * 3, o = i * 3;
      col[o] = CANON_COL[c]; col[o + 1] = CANON_COL[c + 1]; col[o + 2] = CANON_COL[c + 2];
    }
    geom.attributes.color.needsUpdate = true;
  }
  function paintMarks() {
    for (let v = 0; v < vk; v++) {
      const k = v * 3;
      if (reg[v]) { CANON_COL[k] = 0.34; CANON_COL[k + 1] = 1.0; CANON_COL[k + 2] = 0.80; }
      else        { CANON_COL[k] = 1;    CANON_COL[k + 1] = 1;    CANON_COL[k + 2] = 1;    }
    }
    repaintColors();
  }

  /* ---------- красные точки удаляемых вершин ---------- */
  const dotGeo = new THREE.BufferGeometry();
  let dotPos = new Float32Array(vk * 3);
  dotGeo.setAttribute('position', new THREE.BufferAttribute(dotPos, 3));
  dotGeo.setDrawRange(0, 0);
  const dots = new THREE.Points(dotGeo, new THREE.PointsMaterial({
    color: 0xff3b30, size: 5, sizeAttenuation: false,
    depthTest: false, transparent: true, opacity: 0.95
  }));
  dots.renderOrder = 10;
  dots.frustumCulled = false;
  mesh.add(dots);
  function updateDots() {
    let n = 0;
    for (let v = 0; v < vk; v++) {
      if (!pending[v]) continue;
      const o = v * 3;
      dotPos[n * 3] = pos[o]; dotPos[n * 3 + 1] = pos[o + 1]; dotPos[n * 3 + 2] = pos[o + 2];
      n++;
    }
    dotGeo.attributes.position.needsUpdate = true;
    dotGeo.setDrawRange(0, n);
    dotGeo.computeBoundingSphere();
  }
  function clearDots() {
    pending.fill(0);
    dotGeo.setDrawRange(0, 0);
    dotGeo.attributes.position.needsUpdate = true;
  }

  /* ---------- зелёные точки: вершины, добавленные на поверхность ---------- */
  const addGeo = new THREE.BufferGeometry();
  let addPos = new Float32Array(vk * 3);
  addGeo.setAttribute('position', new THREE.BufferAttribute(addPos, 3));
  addGeo.setDrawRange(0, 0);
  const addDots = new THREE.Points(addGeo, new THREE.PointsMaterial({
    color: 0x38e1c4, size: 6, sizeAttenuation: false,
    depthTest: false, transparent: true, opacity: 0.95
  }));
  addDots.renderOrder = 11;
  addDots.frustumCulled = false;
  mesh.add(addDots);
  function updateAddedDots() {
    let n = 0;
    for (let v = 0; v < vk; v++) {
      if (!added[v]) continue;
      const o = v * 3;
      addPos[n * 3] = pos[o]; addPos[n * 3 + 1] = pos[o + 1]; addPos[n * 3 + 2] = pos[o + 2];
      n++;
    }
    addGeo.attributes.position.needsUpdate = true;
    addGeo.setDrawRange(0, n);
    addGeo.computeBoundingSphere();
  }

  /* ---------- служебное: рост массивов при добавлении точек ---------- */
  function resizeNodeArrays(n) {
    /* массивы по узлам растут до n, координатные — до n * 3 */
    const grow = (a, m = n) => {
      const b = new a.constructor(m);
      b.set(a.subarray(0, Math.min(a.length, m)));
      return b;
    };
    reg = grow(reg); dead = grow(dead); pending = grow(pending);
    front = grow(front); added = grow(added); free = grow(free);
    pos = grow(pos, n * 3); pos0 = grow(pos0, n * 3);
    CANON_COL = grow(CANON_COL, n * 3);
    dotPos = grow(dotPos, n * 3);
    dotGeo.setAttribute('position', new THREE.BufferAttribute(dotPos, 3));
    addPos = grow(addPos, n * 3);
    addGeo.setAttribute('position', new THREE.BufferAttribute(addPos, 3));
  }
  function ensureTopo() {
    if (!topoDirty) return;
    const keep0 = pos0.slice();               // исходные координаты не должны теряться
    topo = buildTopology(src, liveIndex);
    if (topo.vk > reg.length) resizeNodeArrays(topo.vk);
    vk = topo.vk; canon = topo.canon;
    pos0 = new Float32Array(vk * 3);
    pos0.set(keep0.subarray(0, Math.min(keep0.length, pos0.length)));
    topo.pos0 = pos0; topo.count = count;
    free = freeMask(topo, reg);
    topoDirty = false;
  }

  /* ---------- добавление точки на поверхность ----------
     Точка ставится ровно на существующую грань (барицентрика по её вершинам) и
     делит её на три — новой поверхности не появляется, точка всегда лежит на
     кузове. Слишком близко к вершине не ставим: получились бы вырожденные
     треугольники. */
  function insertPointAt(faceIdx, p) {
    if (!(faceIdx >= 0) || faceIdx * 3 + 2 >= liveIndex.length) return { ok: false, why: 'грань не найдена' };
    const b0 = faceIdx * 3;
    const i0 = liveIndex[b0], i1 = liveIndex[b0 + 1], i2 = liveIndex[b0 + 2];
    const ax = src[i0 * 3], ay = src[i0 * 3 + 1], az = src[i0 * 3 + 2];
    const bx = src[i1 * 3], by = src[i1 * 3 + 1], bz = src[i1 * 3 + 2];
    const cx = src[i2 * 3], cy = src[i2 * 3 + 1], cz = src[i2 * 3 + 2];
    const v0x = bx - ax, v0y = by - ay, v0z = bz - az;
    const v1x = cx - ax, v1y = cy - ay, v1z = cz - az;
    const v2x = p[0] - ax, v2y = p[1] - ay, v2z = p[2] - az;
    const d00 = v0x * v0x + v0y * v0y + v0z * v0z;
    const d01 = v0x * v1x + v0y * v1y + v0z * v1z;
    const d11 = v1x * v1x + v1y * v1y + v1z * v1z;
    const d20 = v2x * v0x + v2y * v0y + v2z * v0z;
    const d21 = v2x * v1x + v2y * v1y + v2z * v1z;
    const den = d00 * d11 - d01 * d01;
    if (Math.abs(den) < 1e-12) return { ok: false, why: 'вырожденная грань' };
    const v = (d11 * d20 - d01 * d21) / den;
    const w = (d00 * d21 - d01 * d20) / den;
    const u = 1 - v - w;
    const e = 0.15;                       // не ближе 15 % к вершинам грани
    if (u < e || v < e || w < e) return { ok: false, why: 'слишком близко к вершине' };
    const px = ax * u + bx * v + cx * w;   // точка ровно на грани
    const py = ay * u + by * v + cy * w;
    const pz = az * u + bz * v + cz * w;

    const nv = count;                      // индекс новой вершины файла
    const newSrc = new Float32Array((count + 1) * 3);
    newSrc.set(src); newSrc[nv * 3] = px; newSrc[nv * 3 + 1] = py; newSrc[nv * 3 + 2] = pz;
    src = newSrc;
    attr = new THREE.BufferAttribute(src, 3);
    geom.setAttribute('position', attr);
    count += 1;
    const newCol = new Float32Array(count * 3).fill(1);
    newCol.set(col); col = newCol;
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));

    const ni = new Int32Array(liveIndex.length + 6);
    ni.set(liveIndex.subarray(0, b0), 0);
    ni[b0] = i0; ni[b0 + 1] = i1; ni[b0 + 2] = nv;
    ni[b0 + 3] = i1; ni[b0 + 4] = i2; ni[b0 + 5] = nv;
    ni[b0 + 6] = i2; ni[b0 + 7] = i0; ni[b0 + 8] = nv;
    ni.set(liveIndex.subarray(b0 + 3), b0 + 9);
    setIndex(ni);

    resizeNodeArrays(vk + 1);
    const node = vk;
    const nc = new Int32Array(count); nc.set(canon); nc[count - 1] = node;
    canon = nc;
    const nr = new Int32Array(vk + 1); nr.set(topo.rep); nr[node] = nv;
    vk += 1;
    topo.canon = canon; topo.rep = nr; topo.vk = vk; topo.count = count; topo.pos0 = pos0;
    pos[node * 3] = px; pos[node * 3 + 1] = py; pos[node * 3 + 2] = pz;
    pos0[node * 3] = px; pos0[node * 3 + 1] = py; pos0[node * 3 + 2] = pz;
    added[node] = 1;
    topoDirty = true; nrm = null;
    return { ok: true, node };
  }

  /* ---------- «лицевые» узлы: только та поверхность, что видна с камеры ---------- */
  /** клик/протяжка инструментом «Точка»: ставит узлы на существующую грань */
  function addPointsAt(clientX, clientY) {
    const h = hitFace(clientX, clientY);
    if (!h) return;
    if (lastPt) {
      const dx = h.p[0] - lastPt[0], dy = h.p[1] - lastPt[1], dz = h.p[2] - lastPt[2];
      if (dx * dx + dy * dy + dz * dz < (0.35 * U * 0.01) ** 2) return;   // слишком часто
    }
    if (!ptSnap) ptSnap = snapshot();
    const res = insertPointAt(h.faceIndex, h.p);
    if (!res.ok) {
      ptMsg = `точка не поставлена: ${res.why}`;
      if (info) info.textContent = ptMsg;
      return;
    }
    ptMsg = null;
    lastPt = h.p;
    addedTotal += 1;
    /* топологию пересобираем один раз — по отпусканию кнопки (см. stop) */
    nrm = null;
    flush();
    updateAddedDots();
    if (info) {
      info.textContent = `точек добавлено ${addedTotal} · узлов в сетке ${vk}, граней ${liveIndex.length / 3}`;
    }
  }

  function computeFront(center) {
    front.fill(0);
    if (!camera || !camera.position) { front.fill(1); return; }
    const cp = mesh.worldToLocal(camera.position.clone());
    /* узел «лицевой», если его нормаль смотрит на камеру; для склеенной
       (canon) сетки это точнее, чем «принадлежит лицевой грани» */
    const vn = vertexNormals(topo, liveIndex, pos);
    for (let v = 0; v < vk; v++) {
      const dx = pos[v * 3] - cp.x, dy = pos[v * 3 + 1] - cp.y, dz = pos[v * 3 + 2] - cp.z;
      const nl = Math.hypot(vn[v * 3], vn[v * 3 + 1], vn[v * 3 + 2]);
      if (!nl) { front[v] = 0; continue; }
      const nx = vn[v * 3] / nl, ny = vn[v * 3 + 1] / nl, nz = vn[v * 3 + 2] / nl;
      const dl = Math.hypot(dx, dy, dz) || 1;
      /* наклон к камере: нормаль против направления на камеру */
      const back = nx * (dx / dl) + ny * (dy / dl) + nz * (dz / dl);
      /* фронт у самой кромки силуэта (|back| мал) не трогаем — там узлы общие */
      front[v] = back < -0.12 ? 1 : 0;
    }
  }
  /** Маска узлов, которые нельзя трогать в этом штрихе. */
  let block = null;
  function makeBlock() {
    if (!frontOnly) { block = dead; return block; }
    const m = new Uint8Array(vk);
    for (let v = 0; v < vk; v++) m[v] = (dead[v] || !front[v]) ? 1 : 0;
    block = m;
    return m;
  }

  /** Область, очищенная от нелицевых узлов, — считается на момент действия. */
  function frontMaskedRegion() {
    if (!frontOnly) return reg;
    let cx = 0, cy = 0, cz = 0, n = 0;
    for (let v = 0; v < vk; v++) {
      if (!reg[v]) continue;
      cx += pos[v * 3]; cy += pos[v * 3 + 1]; cz += pos[v * 3 + 2]; n++;
    }
    if (!n) return reg;
    computeFront([cx / n, cy / n, cz / n]);
    const m = new Uint8Array(vk);
    let k = 0;
    for (let v = 0; v < vk; v++) if (reg[v] && front[v]) { m[v] = 1; k++; }
    return (k === n || !k) ? reg : m;
  }

  /* ---------- сетка тонкими чёрными линиями ---------- */
  let wire = null, wireOn = false;
  const wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', new THREE.BufferAttribute(src, 3));
  function rebuildWire() {
    if (!wireOn) return;
    const seen = new Set();
    const lines = [];
    const ix = liveIndex;
    const push = (a, b) => {
      const k = Math.min(a, b) * 1e6 + Math.max(a, b);
      if (seen.has(k)) return;
      seen.add(k); lines.push(a, b);
    };
    for (let i = 0; i < ix.length; i += 3) {
      push(ix[i], ix[i + 1]); push(ix[i + 1], ix[i + 2]); push(ix[i + 2], ix[i]);
    }
    wireGeo.setIndex(new THREE.BufferAttribute(new Uint32Array(lines), 1));
    wireGeo.attributes.position.needsUpdate = true;
    wireGeo.computeBoundingSphere();
  }
  function ensureWire() {
    if (!wire) {
      wire = new THREE.LineSegments(wireGeo, new THREE.LineBasicMaterial({
        color: 0x000000, transparent: true, opacity: 0.42
      }));
      wire.frustumCulled = false;
      mesh.add(wire);
    }
    wire.visible = wireOn;
    if (wireOn) rebuildWire();
  }

  /* ---------- измерение: линия между двумя точками ---------- */
  const measGeo = new THREE.BufferGeometry();
  measGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const measLine = new THREE.Line(measGeo, new THREE.LineBasicMaterial({
    color: 0x38e1c4, depthTest: false, transparent: true
  }));
  measLine.renderOrder = 11;
  measLine.frustumCulled = false;
  measLine.visible = false;
  mesh.add(measLine);
  let measPts = [];

  /* ---------- сечение ---------- */
  const secPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  let secOn = false, secBox = null;
  function applySection(v = 0) {
    if (!secOn) {
      material.clippingPlanes = null; material.needsUpdate = true;
      if (wire) { wire.material.clippingPlanes = null; wire.material.needsUpdate = true; }
      return;
    }
    if (!secBox) {
      geom.computeBoundingBox();
      if (!geom.boundingBox) return;                 // нет габаритов — резать нечем
      const bb = geom.boundingBox.clone();
      mesh.updateWorldMatrix(true, false);
      bb.applyMatrix4(mesh.matrixWorld);
      secBox = bb;
    }
    renderer.localClippingEnabled = true;
    const zMin = secBox.min.z, zMax = secBox.max.z;
    const c = zMin + (zMax - zMin) * v;
    secPlane.constant = -c;
    material.clippingPlanes = [secPlane];
    material.needsUpdate = true;
    if (wire) { wire.material.clippingPlanes = [secPlane]; wire.material.needsUpdate = true; }
  }

  /* ---------- координаты → геометрия ---------- */
  function flush(withNormals = true) {
    expandToVertices(topo, pos, src);
    attr.needsUpdate = true;
    if (wireOn) wireGeo.attributes.position.needsUpdate = true;
    if (withNormals) {
      geom.computeVertexNormals();
      geom.computeBoundingSphere();
      geom.computeBoundingBox();
      if (wireOn) wireGeo.computeBoundingSphere();
    }
  }

  /* ---------- история ---------- */
  const hist = [], redo = [];
  function snapshot() {
    return {
      pos: Float32Array.from(pos), reg: reg.slice(), dead: dead.slice(),
      added: added.slice(), n: regionN, del: delTotal, index: liveIndex,
      // массивы топологии заменяются целиком (никогда не правятся на месте),
      // поэтому ссылок достаточно — копия не нужна
      T: { src, index: liveIndex, count, vk }
    };
  }
  function pushHist(snap) {
    hist.push(snap || snapshot());
    if (hist.length > 15) hist.shift();
    redo.length = 0;
  }
  function setIndex(ix) {
    liveIndex = ix;
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(ix), 1));
    if (wireOn) rebuildWire();
  }
  function restore(s) {
    const T = s.T;
    if (T && (T.src !== src || T.count !== count || T.index !== liveIndex)) {
      /* точка вставки добавляла вершину — возвращаем и геометрию, и топологию */
      src = Float32Array.from(T.src);
      count = T.count;
      attr = new THREE.BufferAttribute(src, 3);
      geom.setAttribute('position', attr);
      col = new Float32Array(count * 3).fill(1);
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
      setIndex(Uint32Array.from(T.index));
      topo = buildTopology(src, liveIndex);
      canon = topo.canon;
      vk = Math.min(topo.vk, reg.length);
      topo.vk = vk; topo.count = count; topo.pos0 = pos0;
      topoDirty = false;
    } else if (s.index !== liveIndex) {
      setIndex(s.index);
    }
    const fit = (dst, srcArr) => { dst.set(srcArr.subarray(0, Math.min(srcArr.length, dst.length))); dst.fill(0, Math.min(srcArr.length, dst.length)); };
    fit(pos, s.pos); fit(reg, s.reg); fit(dead, s.dead); fit(added, s.added);
    regionN = s.n; delTotal = s.del;
    nrm = null;
    paintMarks(); free = freeMask(topo, reg);
    updateAddedDots();
    flush(); syncInfo();
  }
  const getNormals = () => (nrm || (nrm = vertexNormals(topo, liveIndex, pos)));

  /* ---------- кисть ---------- */
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let brushMM = 0.5;        // радиус кисти в мм печати (0.1 … 20)
  let depthMM = 0.5;        // глубина лепки в мм печати
  let amount = 25;          // «степень» в процентах: сколько итераций за нажатие
  let painting = false;
  let invert = false;
  let strokeSnap = null;
  let strokeDel = 0;
  let tool = null;          // 'select' | 'erase' | 'pen' | 'shape' | 'measure'
  let shapeKind = 'sphere';
  let sym = false;          // симметрия по y = 0
  let frontOnly = true;     // трогать только видимую (лицевую) поверхность
  let penTicks = 0;
  let lastPt = null;        // последняя добавленная точка (для протяжки)
  let addedTotal = 0;       // сколько точек добавлено за сеанс
  let ptSnap = null;        // снимок до первой вставки точки за штрих
  let ptMsg = null;         // сообщение инструмента «Точка» (переживает отпускание кнопки)

  const brushR = () => brushMM * U;
  const depthD = () => depthMM * U;

  function hit(clientX, clientY) {
    ndc.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObject(mesh, false);
    if (!hits.length) return null;
    return mesh.worldToLocal(hits[0].point.clone());
  }
  function hitPoint(clientX, clientY) {
    const p = hit(clientX, clientY);
    return p ? [p.x, p.y, p.z] : null;
  }
  /** попадание вместе с номером грани — нужно инструменту «Точка» */
  function hitFace(clientX, clientY) {
    ndc.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObject(mesh, false);
    if (!hits.length) return null;
    const h = hits[0];
    const p = mesh.worldToLocal(h.point.clone());
    return { p: [p.x, p.y, p.z], faceIndex: h.faceIndex };
  }
  function nearestNode(p, skip) {
    let best = -1, bd = Infinity;
    for (let v = 0; v < vk; v++) {
      if (skip && skip[v]) continue;
      const o = v * 3;
      const dx = pos[o] - p[0], dy = pos[o + 1] - p[1], dz = pos[o + 2] - p[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < bd) { bd = d2; best = v; }
    }
    return best;
  }

  /* ---------- 1. выбор участка ---------- */
  function paintAt(c) {
    const skip = block || dead;           // что нельзя трогать в этом штрихе
    let added = paintBall(topo, reg, c, brushR(), pos, skip);
    if (!added) {                         // кисть мельче сетки — берём ближайшую точку
      const v = nearestNode(c, skip);
      if (v >= 0 && !reg[v] && !skip[v]) { reg[v] = 1; added = 1; }
    }
    if (added) { regionN += added; paintMarks(); }
    return added;
  }
  function stamp(clientX, clientY) {
    const c = hitPoint(clientX, clientY);
    if (!c) return;
    paintAt(c);
    if (sym) paintAt([c[0], -c[1], c[2]]);
    syncInfo();
  }

  /* ---------- 2. ластик ---------- */
  function eraseAt(c) {
    const r2 = brushR() * brushR();
    let n = 0;
    const mark = (v) => {
      if (dead[v] || (block && block[v])) return;
      dead[v] = 1; pending[v] = 1; delTotal++; n++;
      if (reg[v]) { reg[v] = 0; regionN--; }
    };
    for (let v = 0; v < vk; v++) {
      const o = v * 3;
      const dx = pos[o] - c[0], dy = pos[o + 1] - c[1], dz = pos[o + 2] - c[2];
      if (dx * dx + dy * dy + dz * dz <= r2) mark(v);
    }
    if (!n) { const v = nearestNode(c, dead); if (v >= 0) mark(v); }
    return n;
  }
  function eraseStamp(clientX, clientY) {
    const c = hitPoint(clientX, clientY);
    if (!c) return 0;
    let n = eraseAt(c);
    if (sym) n += eraseAt([c[0], -c[1], c[2]]);
    if (n) { strokeDel += n; paintMarks(); updateDots(); syncInfo(); }
    return n;
  }

  /* ---------- 3. перо ---------- */
  let penMoved = 0;
  function penAt(c, sign) {
    if (++penTicks % 8 === 0) nrm = null;
    const res = pushBall(topo, pos, c, brushR(), sign * depthD(), getNormals(), block || dead);
    if (!res.moved) {                     // кисть мельче сетки — двигаем одну точку
      const v = nearestNode(c, dead);
      if (v >= 0) {
        const o = v * 3, n = getNormals();
        pos[o] += n[o] * sign * depthD();
        pos[o + 1] += n[o + 1] * sign * depthD();
        pos[o + 2] += n[o + 2] * sign * depthD();
        res.moved = 1; res.maxd = Math.abs(sign * depthD());
      }
    }
    penMoved += res.moved;
    return res.moved;
  }
  function penStamp(clientX, clientY, sign) {
    const c = hitPoint(clientX, clientY);
    if (!c) return 0;
    const m = penAt(c, sign) + (sym ? penAt([c[0], -c[1], c[2]], sign) : 0);
    return m;
  }

  /* ---------- 4. фигура ---------- */
  function shapeAt(c, sign) {
    const n = getNormals();
    let nx = 0, ny = 0, nz = 0, wsum = 0;
    const R = brushR(), r2 = R * R;
    for (let v = 0; v < vk; v++) {
      const o = v * 3;
      const dx = pos[o] - c[0], dy = pos[o + 1] - c[1], dz = pos[o + 2] - c[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r2) continue;
      const w = 1 - Math.sqrt(d2) / R + 1e-6;
      nx += n[o] * w; ny += n[o + 1] * w; nz += n[o + 2] * w; wsum += w;
    }
    if (wsum > 0) { nx /= wsum; ny /= wsum; nz /= wsum; }
    else { nx = 0; ny = 0; nz = 1; }
    const shape = {
      kind: shapeKind, center: c, n: [nx, ny, nz],
      size: { r: R, h: Math.max(R * 0.8, depthD()), a: R * 0.8, b: R * 0.8 }
    };
    return imprintShape(topo, pos, shape, sign * depthD(), n, block || dead);
  }
  function shapeStamp(clientX, clientY, sign) {
    const c = hitPoint(clientX, clientY);
    if (!c) return null;
    let res = shapeAt(c, sign);
    if (sym) {
      const r2 = shapeAt([c[0], -c[1], c[2]], sign);
      res = { moved: res.moved + r2.moved, maxd: Math.max(res.maxd, r2.maxd), kind: shapeKind };
    }
    return { ...res, kind: shapeKind };
  }

  /* ---------- 5. измерение ---------- */
  function addMeasurePoint(clientX, clientY) {
    const c = hitPoint(clientX, clientY);
    if (!c) return;
    if (measPts.length >= 2) measPts = [];
    measPts.push(c);
    const arr = measGeo.attributes.position.array;
    if (measPts.length === 1) {
      measLine.visible = false;
      if (info) info.textContent = 'измерение: поставьте вторую точку';
    } else {
      const [a, b] = measPts;
      arr[0] = a[0]; arr[1] = a[1]; arr[2] = a[2];
      arr[3] = b[0]; arr[4] = b[1]; arr[5] = b[2];
      measGeo.attributes.position.needsUpdate = true;
      measGeo.computeBoundingSphere();
      measLine.visible = true;
      const d = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / U;
      const dx = Math.abs(b[0] - a[0]) / U, dy = Math.abs(b[1] - a[1]) / U, dz = Math.abs(b[2] - a[2]) / U;
      if (info) {
        info.innerHTML = `расстояние <b>${d.toFixed(2)} мм</b> · по осям ` +
          `${dx.toFixed(2)} × ${dy.toFixed(2)} × ${dz.toFixed(2)} мм ` +
          `<span style="color:var(--dim)">(клик — новая пара)</span>`;
      }
    }
  }

  /* ---------- зашивка дырки ---------- */
  function applyErase(n) {
    if (!n) return;
    const res = deleteAndFill(topo, liveIndex, pos, dead);
    setIndex(res.index);
    clearDots();
    let regNow = 0, delNow = 0;
    for (let v = 0; v < vk; v++) { if (reg[v]) regNow++; if (dead[v]) delNow++; }
    regionN = regNow; delTotal = delNow;
    nrm = null;
    paintMarks();
    flush();
    if (info) {
      info.textContent = `удалено точек: ${n}` +
        (res.extraDeleted ? ` (+${res.extraDeleted} на кромке)` : '') +
        ` · убрано граней ${res.removedFaces} · зашито дырок: ${res.holes} ` +
        `(${res.addedFaces} треугольников) · поверхность снова замкнута`;
    } else {
      syncInfo();
    }
    return res;
  }

  /* ---------- действия с закрашенной областью ---------- */
  function needRegion() {
    if (regionN) return true;
    syncInfo('сначала закрасьте участок кистью');
    return false;
  }
  function symAfter(before) {
    if (!sym || !before) return 0;
    const m = mirrorDeltas(topo, pos, before, 2.5);
    nrm = null;
    return m.moved;
  }

  function smoothOnce() {
    if (!needRegion()) return;
    const steps = Math.max(20, Math.round(amount / 100 * MAXSTEPS));
    pushHist();
    const before = sym ? Float32Array.from(pos) : null;
    free = freeMask(topo, reg);
    const rg = frontMaskedRegion();
    const w = boundaryFalloff(topo, rg, 4);
    const p = smoothDiffuse(topo, freeMask(topo, rg), steps, LAMBDA, pos);
    for (let v = 0; v < vk; v++) {
      if (w[v] >= 1) continue;
      const o = v * 3;
      for (let c = 0; c < 3; c++) p[o + c] = pos[o + c] + (p[o + c] - pos[o + c]) * w[v];
    }
    pos.set(p);
    const mirrored = symAfter(before);
    nrm = null;
    flush();
    if (mirrored && info) info.textContent = `сглажено · зеркально повторено на ${mirrored} вершинах`;
    else syncInfo();
  }

  function flattenOnce() {
    if (!needRegion()) return;
    pushHist();
    const before = sym ? Float32Array.from(pos) : null;
    const soft = Math.max(1, Math.min(3, Math.round(Math.sqrt(regionN) / 3)));
    const res = flattenRegion(topo, pos, frontMaskedRegion(), liveIndex, 1, soft);
    symAfter(before);
    nrm = null;
    flush();
    if (info) {
      info.textContent = res.moved
        ? `выровнено: ${res.moved} вершин легли в плоскость (макс. ${(res.maxd / U).toFixed(2)} мм)`
        : 'область слишком узкая — закрасьте участок пошире';
    }
  }

  function sharpenOnce() {
    if (!needRegion()) return;
    pushHist();
    const before = sym ? Float32Array.from(pos) : null;
    const soft = Math.max(1, Math.min(3, Math.round(Math.sqrt(regionN) / 3)));
    const res = sharpenRegion(topo, pos, frontMaskedRegion(), liveIndex, 0.5, 3, soft);
    symAfter(before);
    nrm = null;
    flush();
    if (info) info.textContent = `обострено: ${res.moved} вершин (макс. ${(res.maxd / U).toFixed(3)} мм)`;
  }

  function extrude(sign) {
    if (!needRegion()) return;
    pushHist();
    const before = sym ? Float32Array.from(pos) : null;
    const soft = Math.max(1, Math.min(4, Math.round(Math.sqrt(regionN) / 3)));
    const res = offsetRegion(topo, pos, frontMaskedRegion(), sign * depthD(), getNormals(), soft);
    const mirrored = symAfter(before);
    nrm = null;
    flush();
    if (info) {
      info.textContent = res.moved
        ? `${sign > 0 ? 'выдавлено' : 'вдавлено'} на ${depthMM.toFixed(2)} мм · ` +
          `сдвинуто ${res.moved} из ${regionN} вершин` +
          (mirrored ? ` · зеркально ${mirrored}` : '')
        : 'область слишком узкая — закрасьте участок пошире';
    }
  }

  function runCheck() {
    const r = checkMesh(topo, liveIndex, pos, U);
    const ok = r.watertight;
    if (info) {
      info.innerHTML =
        `<b style="color:${ok ? 'var(--acc)' : 'var(--warn)'}">` +
        (ok ? 'сетка замкнута — печатать можно' : 'есть дефекты сетки') + '</b><br>' +
        `граней ${r.faces.toLocaleString('ru-RU')}, вершин ${r.vertices.toLocaleString('ru-RU')} · ` +
        `дыр ${r.openEdges}, несогласованных рёбер ${r.flipEdges}, неманифолдных ${r.nonManifold}, вырожденных ${r.degenerate}<br>` +
        `объём ${(r.volumeMM3 / 1000).toFixed(1)} см³ · габарит ` +
        `${r.sizeMM.map(x => x.toFixed(1)).join(' × ')} мм · мин. ребро ` +
        (r.minEdgeMM < 0.01 ? `${(r.minEdgeMM * 1000).toFixed(1)} мкм` : `${r.minEdgeMM.toFixed(3)} мм`);
    }
    return r;
  }

  /* ---------- интерфейс ---------- */
  const $ = (id) => document.getElementById(id);
  const btnMode  = $('sc-mode');
  const btnDel   = $('sc-del');
  const btnPen   = $('sc-pen');
  const btnShape = $('sc-shape');
  const btnMeas  = $('sc-measure');
  const btnPoint = $('sc-point');
  const btnSym   = $('sc-sym');
  const rows     = Array.from(document.querySelectorAll('.sc-row'));
  const hint     = $('sc-hint');
  const info     = $('sc-info');
  const HINTS = {
    select: hint ? hint.innerHTML : '',
    erase: 'Левой кнопкой <b>сотрите</b> лишние точки: они подсвечиваются красными ' +
      'точками и уходят по отпусканию кнопки, а дырка зашивается треугольниками по ' +
      'своей кромке — поверхность сама соединится с ближайшими оставшимися точками.',
    pen: 'Левой кнопкой <b>ведите по поверхности</b>: под пером остаётся валик ' +
      'высотой «Глубина» и шириной «Кисть». С зажатым <b>Shift</b> — канавка (внутрь).',
    shape: 'Левой кнопкой <b>поставьте фигуру</b> на кузов — она вдавится в ' +
      'поверхность (● сфера / ▮ цилиндр / ■ куб, размер — «Кисть», глубина — ' +
      '«Глубина»). С зажатым <b>Shift</b> — купол наружу.',
    measure: 'Два клика по поверхности — расстояние в мм (и разница по осям). ' +
      'Третий клик начинает новую пару.',
    point: '<b>Клик или протяжка по кузову</b> — новая точка ровно на существующей ' +
      'грани (грань делится на три, новой поверхности не появляется). Появляются ' +
      'зелёные метки. Ближе 15 % к вершинам грани точка не ставится — иначе ' +
      'получились бы вырожденные треугольники.'
  };

  function syncInfo(note) {
    if (!info) return;
    if (note) { info.textContent = note; return; }
    const faces = liveIndex.length / 3;
    if (tool === 'erase') {
      info.textContent = `ластик ${brushMM.toFixed(1)} мм · в сетке ${faces} граней`;
    } else if (tool === 'point') {
      info.textContent = `точек добавлено ${addedTotal} · узлов ${vk}, граней ${faces}` +
        (frontOnly ? ' · только видимые грани' : '');
    } else if (tool === 'pen') {
      info.textContent = `перо ${brushMM.toFixed(1)} мм, высота ${depthMM.toFixed(2)} мм` +
        ` · в сетке ${faces} граней`;
    } else if (tool === 'shape') {
      info.textContent = `фигура «${SHAPE_NAMES[shapeKind]}»: радиус ${brushMM.toFixed(1)} мм, ` +
        `глубина ${depthMM.toFixed(2)} мм`;
    } else if (tool === 'measure') {
      info.textContent = 'измерение: два клика по поверхности';
    } else if (regionN) {
      const pct = (100 * regionN / vk).toFixed(0);
      info.textContent = `выбрано ${regionN} вершин (${pct} %), ` +
        `за нажатие ${Math.max(20, Math.round(amount / 100 * MAXSTEPS))} итераций`;
    } else {
      info.textContent = 'область не выбрана — закрасьте участок кистью';
    }
  }

  function syncRows() {
    const on = !!tool;
    rows.forEach(r => r.classList.toggle('hd', !on));
    if (hint) hint.classList.toggle('hd', !on);
    if (!on) return;
    const show = (id, v) => { const e = $(id); if (e) e.classList.toggle('hd', !v); };
    show('sc-amtrow', tool === 'select');
    show('sc-actions', tool === 'select');
    show('sc-extrude-row', tool === 'select');
    show('sc-shaperow', tool === 'shape');
    const depthRow = $('sc-depthrow');
    if (depthRow) depthRow.classList.toggle('hd',
      tool === 'erase' || tool === 'measure');
    const brushRow = $('sc-brushrow');
    if (brushRow) brushRow.classList.toggle('hd', tool === 'measure' || tool === 'point');
  }

  function syncControls() {
    const on = !!tool;
    const MB = THREE.MOUSE;
    if (on) {
      controls.mouseButtons = { LEFT: null, MIDDLE: MB.DOLLY, RIGHT: MB.ROTATE };
      controls.touches = { ONE: null, TWO: MB.DOLLY_ROTATE };
    } else {
      controls.mouseButtons = { LEFT: MB.ROTATE, MIDDLE: MB.DOLLY, RIGHT: MB.PAN };
      controls.touches = { ONE: MB.ROTATE, TWO: MB.DOLLY_PAN };
    }
    renderer.domElement.style.cursor = !on ? ''
      : tool === 'erase' ? 'cell' : tool === 'shape' ? 'copy'
      : tool === 'measure' ? 'crosshair' : 'crosshair';
  }

  function setTool(t) {
    tool = (tool === t) ? null : t;
    const tag = (btn, v, onText, offText) => {
      if (!btn) return;
      btn.classList.toggle('on', v);
      btn.textContent = v ? onText : offText;
    };
    tag(btnMode, tool === 'select', '■ Закончить участок', '✎ Участок');
    tag(btnDel, tool === 'erase', '■ Закончить удаление', '⌫ Ластик');
    tag(btnPen, tool === 'pen', '■ Закончить рисование', '✏ Рисовать');
    tag(btnShape, tool === 'shape', '■ Закончить фигуры', '◆ Фигура');
    tag(btnMeas, tool === 'measure', '■ Закончить измерение', '📏 Измерить');
    tag(btnPoint, tool === 'point', '■ Закончить добавление', '✚ Точка');
    if (hint) hint.innerHTML = HINTS[tool] || HINTS.select;
    if (tool !== 'measure') { measPts = []; measLine.visible = false; }
    block = null;                 // маска лицевых граней считается заново на каждом штрихе
    if (tool !== 'point') { ptSnap = null; lastPt = null; }
    syncRows();
    clearDots();
    syncControls();
    paintMarks();
    syncInfo();
  }

  const cvs = renderer.domElement;
  cvs.addEventListener('contextmenu', e => { if (tool) e.preventDefault(); });
  cvs.addEventListener('pointerdown', e => {
    if (!tool || e.button !== 0) return;
    if (tool === 'measure') { addMeasurePoint(e.clientX, e.clientY); return; }
    painting = true;
    invert = e.shiftKey;
    try { cvs.setPointerCapture(e.pointerId); } catch (_) {}
    if (tool === 'point') {
      lastPt = null;
      addPointsAt(e.clientX, e.clientY);
    } else if (tool === 'erase') {
      strokeDel = 0; clearDots(); strokeSnap = snapshot();
      const c0 = hitPoint(e.clientX, e.clientY);
      if (c0) computeFront(c0);
      makeBlock();
      eraseStamp(e.clientX, e.clientY);
    } else if (tool === 'pen') {
      pushHist(); nrm = null; penMoved = 0;
      const c0 = hitPoint(e.clientX, e.clientY);
      if (c0) computeFront(c0);
      makeBlock();
      penStamp(e.clientX, e.clientY, invert ? -1 : 1);
      flush();
    } else if (tool === 'shape') {
      pushHist(); nrm = null;
      const c0 = hitPoint(e.clientX, e.clientY);
      if (c0) computeFront(c0);
      makeBlock();
      const res = shapeStamp(e.clientX, e.clientY, invert ? -1 : 1);
      nrm = null;
      flush();
      if (res && info) {
        info.textContent = `фигура «${SHAPE_NAMES[res.kind]}» ${invert ? 'вытянула' : 'вдавила'} ` +
          `поверхность: ${res.moved} вершин, макс. ${(res.maxd / U).toFixed(2)} мм`;
      }
      painting = false;
    } else {
      const c0 = hitPoint(e.clientX, e.clientY);
      if (c0) computeFront(c0);
      makeBlock();
      stamp(e.clientX, e.clientY);
    }
  });
  cvs.addEventListener('pointermove', e => {
    if (!painting || !tool) return;
    if (tool === 'point') addPointsAt(e.clientX, e.clientY);
    else if (tool === 'erase') eraseStamp(e.clientX, e.clientY);
    else if (tool === 'pen') { penStamp(e.clientX, e.clientY, invert ? -1 : 1); flush(); }
    else if (tool === 'select') stamp(e.clientX, e.clientY);
  });
  const stop = () => {
    if (!painting) return;
    painting = false;
    if (tool === 'erase' && strokeDel) {
      const n = strokeDel;
      strokeDel = 0;
      pushHist(strokeSnap);
      strokeSnap = null;
      applyErase(n);
    } else if (tool === 'pen') {
      nrm = null;
      if (info) info.textContent = `перо: сдвинуто ${penMoved} вершин на ${depthMM.toFixed(2)} мм`;
    } else if (tool === 'point') {
      ensureTopo(); nrm = null;
      if (ptSnap) { hist.push(ptSnap); if (hist.length > 15) hist.shift(); redo.length = 0; ptSnap = null; }
      flush();
      updateAddedDots();
      if (ptMsg) { const m = ptMsg; ptMsg = null; syncInfo(m); }
      else syncInfo();
    }
  };
  cvs.addEventListener('pointerup', stop);
  cvs.addEventListener('pointercancel', stop);
  cvs.addEventListener('pointerleave', stop);

  if (btnMode) btnMode.addEventListener('click', () => setTool('select'));
  if (btnDel) btnDel.addEventListener('click', () => setTool('erase'));
  if (btnPen) btnPen.addEventListener('click', () => setTool('pen'));
  if (btnShape) btnShape.addEventListener('click', () => setTool('shape'));
  if (btnMeas) btnMeas.addEventListener('click', () => setTool('measure'));
  if (btnPoint) btnPoint.addEventListener('click', () => setTool('point'));

  /* ограничение правок лицевыми гранями */
  const chkFront = $('sc-front');
  if (chkFront) {
    chkFront.checked = frontOnly;
    chkFront.addEventListener('change', () => {
      frontOnly = chkFront.checked;
      chkFront.classList.toggle('on', frontOnly);
      chkFront.textContent = frontOnly ? '◐ Только видимые грани' : '◐ Грани: все';
      block = null; front.fill(frontOnly ? 0 : 1);
      syncInfo(frontOnly
        ? 'правки касаются только граней, видимых с камеры'
        : 'ограничение снято: кисть проходит кузов насквозь');
    });
  }

  /* симметрия */
  if (btnSym) btnSym.addEventListener('click', () => {
    sym = !sym;
    btnSym.classList.toggle('on', sym);
    btnSym.textContent = sym ? '⇄ Симметрия вкл' : '⇄ Симметрия';
    syncInfo(sym ? 'симметрия включена: правки повторяются зеркально по плоскости y = 0'
                 : 'симметрия выключена');
  });

  /* проверка сетки */
  const checkBtn = $('sc-check');
  if (checkBtn) checkBtn.addEventListener('click', runCheck);

  /* фигуры */
  ['sphere', 'cylinder', 'box'].forEach((k) => {
    const b = $('sc-fig-' + k);
    if (b) b.addEventListener('click', () => {
      shapeKind = k;
      ['sphere', 'cylinder', 'box'].forEach((k2) => {
        const b2 = $('sc-fig-' + k2);
        if (b2) b2.classList.toggle('on', k2 === k);
      });
      syncInfo();
    });
    if (b && k === shapeKind) b.classList.add('on');
  });

  const brushEl = $('sc-brush');
  if (brushEl) brushEl.addEventListener('input', () => {
    brushMM = +brushEl.value;
    $('sc-brushv').textContent = brushMM.toFixed(1) + ' мм';
    syncInfo();
  });
  const depthEl = $('sc-depth');
  if (depthEl) depthEl.addEventListener('input', () => {
    depthMM = +depthEl.value;
    $('sc-depthv').textContent = depthMM.toFixed(2) + ' мм';
    syncInfo();
  });
  const amtEl = $('sc-amt');
  if (amtEl) amtEl.addEventListener('input', () => {
    amount = +amtEl.value;
    $('sc-amtv').textContent = Math.round(amount) + '%';
    syncInfo();
  });
  const on = (id, fn, ev = 'click') => { const e = $(id); if (e) e.addEventListener(ev, fn); };
  on('sc-go', smoothOnce);
  on('sc-flat', flattenOnce);
  on('sc-sharp', sharpenOnce);
  on('sc-out', () => extrude(1));
  on('sc-in', () => extrude(-1));

  /* сетка */
  const wireBtn = $('wire');
  if (wireBtn) wireBtn.addEventListener('click', () => {
    wireOn = !wireOn;
    wireBtn.classList.toggle('on', wireOn);
    wireBtn.textContent = wireOn ? 'Сетка вкл' : 'Сетка';
    ensureWire();
    if (wireOn && wire) rebuildWire();
  });

  /* сечение */
  const secBtn = $('sec');
  const secRow = $('secrow');
  const secEl = $('sec-pos');
  function syncSection() {
    if (secRow) secRow.classList.toggle('hd', !secOn);
    applySection(secEl ? +secEl.value / 100 : 0);
  }
  if (secBtn) secBtn.addEventListener('click', () => {
    secOn = !secOn;
    secBtn.classList.toggle('on', secOn);
    secBtn.textContent = secOn ? 'Сечение вкл' : 'Сечение';
    syncSection();
  });
  if (secEl) secEl.addEventListener('input', () => {
    const v = +secEl.value;
    const e = $('secv'); if (e) e.textContent = v + '%';
    applySection(v / 100);
  });

  /* отмена / возврат / сброс */
  on('sc-undo', () => {
    const h = hist.pop();
    if (!h) { syncInfo('отменять нечего'); return; }
    redo.push(snapshot());
    restore(h);
  });
  on('sc-redo', () => {
    const r = redo.pop();
    if (!r) { syncInfo('возвращать нечего'); return; }
    hist.push(snapshot());
    restore(r);
  });
  on('sc-clear', () => {
    pushHist();
    if (count !== origCount || liveIndex.length !== origIndex.length) {
      /* точки добавлялись — возвращаем исходную геометрию целиком */
      src = Float32Array.from(origSrc); count = origCount;
      attr = new THREE.BufferAttribute(src, 3);
      geom.setAttribute('position', attr);
      col = new Float32Array(count * 3).fill(1);
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
      setIndex(origIndex.slice());
      topo = buildTopology(src, liveIndex);
      canon = topo.canon; vk = Math.min(topo.vk, reg.length);
      topo.vk = vk; topo.count = count; topo.pos0 = pos0; topoDirty = false;
    }
    pos0.fill(0); pos0.set(posInit.subarray(0, Math.min(posInit.length, pos0.length)));
    pos.fill(0); pos.set(posInit.subarray(0, Math.min(posInit.length, pos.length)));
    reg.fill(0); regionN = 0;
    dead.fill(0); delTotal = 0;
    added.fill(0); addedTotal = 0; front.fill(1); block = null;
    setIndex(origIndex.slice());
    nrm = null;
    paintMarks(); updateAddedDots(); flush(); syncInfo();
  });
  globalThis.addEventListener && globalThis.addEventListener('keydown', (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    const k = (e.key || '').toLowerCase();
    if (k === 'z' && !e.shiftKey) { e.preventDefault(); const h = hist.pop(); if (h) { redo.push(snapshot()); restore(h); } }
    else if ((k === 'z' && e.shiftKey) || k === 'y') { e.preventDefault(); const r = redo.pop(); if (r) { hist.push(snapshot()); restore(r); } }
  });

  /* выгрузка */
  function save(kind) {
    const cmp = compactMesh(src, liveIndex);
    if (kind === 'stl') {
      const bytes = meshToSTL(cmp.positions, cmp.index, 'кузов');
      const name = 'Tayron_body.stl';
      const res = downloadBytes(bytes, name, 'model/stl');
      if (info) {
        info.innerHTML = `готово: <a href="${res.url}" download="${name}" style="color:var(--acc)">` +
          `скачать ${name}</a> · ${(bytes.length / 1048576).toFixed(1)} МБ` +
          (cmp.dropped ? ` · ${cmp.dropped} вершин без граней выкинуто` : '');
      }
      return;
    }
    const bytes = meshTo3mf(cmp.positions, cmp.index, {
      name: 'Кузов',
      meta: 'VW Tayron: кузов после правки в браузере (печать 17 %)'
    });
    const name = 'Tayron_body_smooth.3mf';
    const res = downloadBytes(bytes, name);
    if (info) {
      info.innerHTML = `готово: <a href="${res.url}" download="${name}" style="color:var(--acc)">` +
        `скачать ${name}</a> · ${(bytes.length / 1048576).toFixed(1)} МБ` +
        (cmp.dropped ? ` · ${cmp.dropped} вершин без граней выкинуто` : '') +
        '. Если браузер не начал скачивание сам — нажмите ссылку.';
    }
  }
  on('sc-save', () => save('3mf'));
  on('sc-stl', () => save('stl'));

  syncInfo();

  return {
    get mode() { return !!tool; },
    get tool() { return tool; },
    get erasing() { return tool === 'erase'; },
    setTool,
    get regionCount() { return regionN; },
    get deletedCount() { return delTotal; },
    get symmetrical() { return sym; },
    runCheck,
    toggleWire() { wireOn = !wireOn; ensureWire(); },
    setSection(v) { secOn = v !== null; if (v === null) { applySection(0); } else applySection(v); },
    stats: () => ({
      vertices: vk, faces: liveIndex.length / 3,
      sourceVertices: count, welded: topo.welded,
      deleted: delTotal, facesNow: liveIndex.length / 3
    })
  };
}
