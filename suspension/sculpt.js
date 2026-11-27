/* ============================================================
   Редактор кузова на сайте.

   Инструменты (все двигают только существующие вершины):
   1. ✎ Участок — кисть закрашивает область (бирюзовым), дальше:
        «≡ Сгладить» (диффузия по рёбрам), «▭ Выровнять» (область садится в
        плоскость), «✦ Обострить» (обратное сглаживанию), «⬆ Выдавить» /
        «⬇ Вдавить» (сдвиг вдоль нормалей).
   2. ✏ Рисовать — перо: валик по штриху (Shift — канавка).
   3. ◆ Фигура — вдавливает в кузов сферу / цилиндр / куб (Shift — купол).
     В пятне касания сетка сначала сгущается до размера кисти, потом штампует —
     работает и на грубой сетке (куб 20 мм), и на мелкой (кузов).
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
  buildTopology, freeMask, smoothDiffuse, boundaryFalloff, paintBall, setFalloff, FALLOFFS,
  expandToVertices, deleteAndFill, compactMesh, vertexNormals,
  pushBall, offsetRegion, imprintShape, flattenRegion, sharpenRegion, clampFlips, boundaryEdgeKeys,
  relaxRegion, growRegion, shrinkRegion, pinchRegion,
  euclidFalloff,
  mirrorDeltas, checkMesh
} from './meshops.js';
import { pathLen, smoothPath, resamplePath, tubeArrays, smoothWindow, pathTurn, pathKink, relaxPath, straightPath } from './ropes.js';

const MAXSTEPS = 60;    // итераций за одно нажатие при «Степень» 100 %
                        // (было 600: одно нажатие на 25 % утаскивало узел на 1.6 мм —
                        //  это уже не «сгладить», а вмятина)
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
  let wireDirty = false;                  // индексы сетки-оверлея устарели

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
    /* Выделение принадлежит инструменту «Участок»: пока активен другой
       инструмент, бирюзовая подсветка спрятана, но сами закрашенные узлы
       остаются — клавиша 1 вернёт их на экран вместе с выделением. */
    const showReg = !tool || tool === 'select';
    for (let v = 0; v < vk; v++) {
      const k = v * 3;
      if (reg[v] && showReg) { CANON_COL[k] = 0.34; CANON_COL[k + 1] = 1.0; CANON_COL[k + 2] = 0.80; }
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
  function insertPointAt(faceIdx, p, doFlips = true) {
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
    if (u < e || v < e || w < e) {
      /* Клик у кромки: делить грань на три нельзя (иголка), но и отказывать
         нельзя — в настоящих редакторах точка встаёт всегда. Расщепляем
         ближайшее ребро: с каждой стороны кромки получаются два ровных
         треугольника, шов остаётся склеенным. */
      const pair = u <= v && u <= w ? [i1, i2] : (v <= w ? [i2, i0] : [i0, i1]);
      return splitEdgeAt(pair[0], pair[1], p, doFlips);
    }
    const px = ax * u + bx * v + cx * w;   // точка ровно на грани
    const py = ay * u + by * v + cy * w;
    const pz = az * u + bz * v + cz * w;

    const nv = count;                      // индекс новой вершины файла
    const newSrc = new Float32Array((count + 1) * 3);
    newSrc.set(src); newSrc[nv * 3] = px; newSrc[nv * 3 + 1] = py; newSrc[nv * 3 + 2] = pz;
    count += 1;
    applySrc(newSrc);
    const newCol = new Float32Array(count * 3).fill(1);
    newCol.set(col); col = newCol;
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));

    const ni = new Int32Array(liveIndex.length + 6);
    ni.set(liveIndex.subarray(0, b0), 0);
    ni[b0] = i0; ni[b0 + 1] = i1; ni[b0 + 2] = nv;
    ni[b0 + 3] = i1; ni[b0 + 4] = i2; ni[b0 + 5] = nv;
    ni[b0 + 6] = i2; ni[b0 + 7] = i0; ni[b0 + 8] = nv;
    ni.set(liveIndex.subarray(b0 + 3), b0 + 9);
    /* притягиваем ближайшие узлы к новой точке (перевороты по Делоне) */
    if (doFlips) attachNeighbours(nv, 8, ni);
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

  /** Расщепить ребро (a,b) в точке p — путь «Точки» у кромки грани.
      Обе грани, которым принадлежит ребро, делятся пополам, поэтому шов не
      расходится (никаких T-стыков) и вода не течёт. */
  function splitEdgeAt(a, b, p, doFlips = true) {
    const P = src;
    const ax = P[a * 3], ay = P[a * 3 + 1], az = P[a * 3 + 2];
    const bx = P[b * 3], by = P[b * 3 + 1], bz = P[b * 3 + 2];
    const ex = bx - ax, ey = by - ay, ez = bz - az;
    const L2 = ex * ex + ey * ey + ez * ez;
    if (L2 < 1e-12) return { ok: false, why: 'нулевое ребро' };
    let t = ((p[0] - ax) * ex + (p[1] - ay) * ey + (p[2] - az) * ez) / L2;
    t = Math.min(0.80, Math.max(0.20, t));      // не прилипаем к вершине и не режем крайность
    const px = ax + ex * t, py = ay + ey * t, pz = az + ez * t;

    const li = liveIndex, m = li.length / 3, hit = new Map();
    for (let f = 0; f < m; f++) {
      const i0 = li[f * 3], i1 = li[f * 3 + 1], i2 = li[f * 3 + 2];
      for (let k = 0; k < 3; k++) {
        const x = k === 0 ? i0 : k === 1 ? i1 : i2;
        const y = k === 0 ? i1 : k === 1 ? i2 : i0;
        const z = k === 0 ? i2 : k === 1 ? i0 : i1;
        if ((x === a && y === b) || (x === b && y === a)) hit.set(f, [x, y, z]);
      }
    }
    if (!hit.size) return { ok: false, why: 'ребро не найдено' };

    const nv = count;
    const newSrc = new Float32Array((count + 1) * 3);
    newSrc.set(src);
    newSrc[nv * 3] = px; newSrc[nv * 3 + 1] = py; newSrc[nv * 3 + 2] = pz;
    count += 1;
    applySrc(newSrc);
    const newCol = new Float32Array(count * 3).fill(1);
    newCol.set(col); col = newCol;
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));

    const ni = new Int32Array((m + hit.size) * 3);
    let o = 0;
    for (let f = 0; f < m; f++) {
      const h = hit.get(f);
      if (!h) {
        ni[o++] = li[f * 3]; ni[o++] = li[f * 3 + 1]; ni[o++] = li[f * 3 + 2];
        continue;
      }
      const [x, y, z] = h;                       // x→y — направленное ребро
      ni[o++] = x; ni[o++] = nv; ni[o++] = z;    // (x, nv, z)
      ni[o++] = nv; ni[o++] = y; ni[o++] = z;    // (nv, y, z)
    }
    setIndex(ni.subarray(0, o));
    if (doFlips) attachNeighbours(nv, 8, liveIndex);

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
    return { ok: true, node, split: true };
  }

  /* ---------- привязка соседей: локальные перевороты рёбер ----------
     После расщепления грани на три точка соединяется только с вершинами этой
     грани; если кликнули близко к вершине или ребру, два треугольника выходят
     «иголками» — на экране это выглядит, будто плоскость поехала. Поэтому
     сразу после вставки переворачиваем рёбра по критерию Делоне: новая точка
     «притягивает» к себе ближайшие узлы, иголки исчезают. Работаем только в
     окрестности (два кольца вокруг новой точки), чтобы не сканировать всю
     сетку на каждый переворот. */
  const crossN = (px, py, pz, qx, qy, qz) => [py * qz - pz * qy, pz * qx - px * qz, px * qy - py * qx];
  function triNormal(i0, i1, i2, P) {
    const a = i0 * 3, b = i1 * 3, c = i2 * 3;
    return crossN(P[b] - P[a], P[b + 1] - P[a + 1], P[b + 2] - P[a + 2],
                  P[c] - P[a], P[c + 1] - P[a + 1], P[c + 2] - P[a + 2]);
  }
  function inCircle(a, b, c, d, P) {
    /* d внутри описанной окружности abc — считаем в плоскости abc */
    const e1 = [P[b * 3] - P[a * 3], P[b * 3 + 1] - P[a * 3 + 1], P[b * 3 + 2] - P[a * 3 + 2]];
    const e2 = [P[c * 3] - P[a * 3], P[c * 3 + 1] - P[a * 3 + 1], P[c * 3 + 2] - P[a * 3 + 2]];
    let n = crossN(e1[0], e1[1], e1[2], e2[0], e2[1], e2[2]);
    const ln = Math.hypot(n[0], n[1], n[2]);
    if (!ln) return false;
    n = [n[0] / ln, n[1] / ln, n[2] / ln];
    let u = e1.slice();
    const lu = Math.hypot(u[0], u[1], u[2]) || 1;
    u = [u[0] / lu, u[1] / lu, u[2] / lu];
    const v = crossN(n[0], n[1], n[2], u[0], u[1], u[2]);
    const p2 = (i) => {
      const dx = P[i * 3] - P[a * 3], dy = P[i * 3 + 1] - P[a * 3 + 1], dz = P[i * 3 + 2] - P[a * 3 + 2];
      return [dx * u[0] + dy * u[1] + dz * u[2], dx * v[0] + dy * v[1] + dz * v[2]];
    };
    const A = [0, 0], B = p2(b), C = p2(c), D = p2(d);
    const adx = A[0] - D[0], ady = A[1] - D[1];
    const bdx = B[0] - D[0], bdy = B[1] - D[1];
    const cdx = C[0] - D[0], cdy = C[1] - D[1];
    const ab = adx * bdy - bdx * ady;
    const bc = bdx * cdy - cdx * bdy;
    const ca = cdx * ady - adx * cdy;
    const al = adx * adx + ady * ady, bl = bdx * bdx + bdy * bdy, cl = cdx * cdx + cdy * cdy;
    const det = al * bc + bl * ca + cl * ab;
    const area = (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]);
    return (area > 0 ? det > 0 : det < 0);
  }
  /** рёбра вокруг новой вершины: переворот, если он улучшает треугольники */
  function attachNeighbours(nvFile, maxFlips = 8, idx = liveIndex) {
    const key = (a, b) => (a < b ? a * 1e6 + b : b * 1e6 + a);
    /* 1) кольцо вокруг новой точки (один проход по сетке) */
    const ring = new Set([nvFile]);
    for (let i = 0; i < idx.length; i += 3) {
      if (idx[i] === nvFile || idx[i + 1] === nvFile || idx[i + 2] === nvFile) {
        ring.add(idx[i]); ring.add(idx[i + 1]); ring.add(idx[i + 2]);
      }
    }
    /* 2) грани, задевающие это кольцо, и карта их рёбер (второй проход) */
    const faces = [];
    for (let i = 0; i < idx.length; i += 3) {
      if (ring.has(idx[i]) || ring.has(idx[i + 1]) || ring.has(idx[i + 2])) faces.push(i);
    }
    const emap = new Map();
    const addEdges = (f, sign) => {
      for (let k = 0; k < 3; k++) {
        const a = idx[f + k], b = idx[f + (k + 1) % 3];
        const kk = key(a, b);
        if (sign > 0) { const e = emap.get(kk); if (e) e.push(f); else emap.set(kk, [f]); }
        else { const e = emap.get(kk); if (e) { const j = e.indexOf(f); if (j >= 0) e.splice(j, 1); if (!e.length) emap.delete(kk); } }
      }
    };
    for (const f of faces) addEdges(f, 1);
    const setTri = (f, a, b, c) => {
      addEdges(f, -1);
      idx[f] = a; idx[f + 1] = b; idx[f + 2] = c;
      addEdges(f, 1);
    };

    let flips = 0;
    for (let round = 0; round < maxFlips * 3 && flips < maxFlips; round++) {
      let done = false;
      const mine = [];
      for (const f of faces) {
        if (idx[f] === nvFile || idx[f + 1] === nvFile || idx[f + 2] === nvFile) mine.push(f);
      }
      for (const f of mine) {
        const k = idx[f] === nvFile ? 0 : (idx[f + 1] === nvFile ? 1 : 2);
        const a = idx[f + (k + 1) % 3], b = idx[f + (k + 2) % 3];
        const near = emap.get(key(a, b)) || [];
        const other = near.find((g) => g !== f);
        if (other === undefined) continue;
        const x = idx[other], y = idx[other + 1], z = idx[other + 2];
        const c = x !== a && x !== b ? x : (y !== a && y !== b ? y : z);
        if (c === a || c === b || c === nvFile) continue;
        if (!inCircle(a, b, c, nvFile, src)) continue;
        const nb = triNormal(x, y, z, src);
        /* вторая грань должна лежать в той же плоскости (переворот через ребро
           куба/подоконник проваливал бы треугольник внутрь детали) */
        const n0 = triNormal(nvFile, a, b, src);
        const l0 = Math.hypot(...n0), lb = Math.hypot(...nb);
        if (l0 < 1e-12 || lb < 1e-12) continue;
        if ((n0[0] * nb[0] + n0[1] * nb[1] + n0[2] * nb[2]) / (l0 * lb) < 0.85) continue;
        const n1 = triNormal(nvFile, a, c, src), n2 = triNormal(nvFile, c, b, src);
        const l1 = Math.hypot(...n1), l2 = Math.hypot(...n2);
        if (l1 < 1e-12 || l2 < 1e-12) continue;             // вырожденные — не трогаем
        if (n1[0] * nb[0] + n1[1] * nb[1] + n1[2] * nb[2] <= 0) continue;
        if (n2[0] * nb[0] + n2[1] * nb[1] + n2[2] * nb[2] <= 0) continue;
        setTri(f, nvFile, a, c);
        setTri(other, nvFile, c, b);
        flips++; done = true;
        break;
      }
      if (!done) break;
    }
    return flips;
  }

  /* ---------- «лицевые» узлы: только та поверхность, что видна с камеры ---------- */
  /** клик/протяжка инструментом «Точка»: ставит узлы на существующую грань */
  function addPointsAt(clientX, clientY) {
    const h = hitFace(clientX, clientY);
    if (!h) return;
    /* Повторный клик в то же место не должен плодить узел поверх узла: получилась
       бы нулевая грань (в Blender «Insert vertex» так же упирается в вершину).
       Допуск — 20 % наименьшего ребра грани под курсором, но не мельче 0.2 % U. */
    const b3 = h.faceIndex * 3;
    if (b3 >= 0 && b3 + 2 < liveIndex.length) {
      const i0 = liveIndex[b3], i1 = liveIndex[b3 + 1], i2 = liveIndex[b3 + 2];
      const e = (a, b) => Math.hypot(src[a * 3] - src[b * 3], src[a * 3 + 1] - src[b * 3 + 1], src[a * 3 + 2] - src[b * 3 + 2]);
      const minEdge = Math.min(e(i0, i1), e(i1, i2), e(i2, i0));
      /* Допуск «уже стоит» — 8 % ребра (на кузове это 0.08 мм): он ловит
         повторный клик в ту же точку, но не запрещает ставить точку внутри
         мелкой грани — с 20 % половина кузова была в «мёртвых зонах». */
      const tol = Math.max(0.08 * minEdge, 0.02 * U);
      const v = nearestNode(h.p);
      if (v >= 0) {
        const d = Math.hypot(pos[v * 3] - h.p[0], pos[v * 3 + 1] - h.p[1], pos[v * 3 + 2] - h.p[2]);
        if (d < tol) {
          ptMsg = `точка здесь уже стоит (${(d / U).toFixed(2)} мм до вершины) — сдвиньте курсор`;
          if (info) info.textContent = ptMsg;
          return;
        }
      }
    }
    if (!ptSnap) ptSnap = snapshot();
    const res = insertPointAt(h.faceIndex, h.p);
    if (!res.ok) {
      ptMsg = `точка не поставлена: ${res.why}`;
      if (info) info.textContent = ptMsg;
      return;
    }
    ptMsg = null;
    paintMarks();               // у новой вершины цвет по умолчанию — вернуть подсветку области
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
       (canon) сетки это точнее, чем «принадлежит лицевой грани».
       Нормали берём из общего кэша: без этого на кузове они считались дважды
       за нажатие (computeFront + перо) — лишние ~100 мс. */
    const vn = getNormals();
    /* Нормаль под курсором: кисть правит ту поверхность, куда попал луч, и не
       должна проходить сквозь тонкую стенку на её обратную сторону. */
    let hx = 0, hy = 0, hz = 0;
    if (center) {
      const hv = nearestNode(center);
      if (hv >= 0) {
        const hl = Math.hypot(vn[hv * 3], vn[hv * 3 + 1], vn[hv * 3 + 2]) || 1;
        hx = vn[hv * 3] / hl; hy = vn[hv * 3 + 1] / hl; hz = vn[hv * 3 + 2] / hl;
      }
    }
    const hasHit = hx || hy || hz;
    for (let v = 0; v < vk; v++) {
      const dx = pos[v * 3] - cp.x, dy = pos[v * 3 + 1] - cp.y, dz = pos[v * 3 + 2] - cp.z;
      const nl = Math.hypot(vn[v * 3], vn[v * 3 + 1], vn[v * 3 + 2]);
      if (!nl) { front[v] = 0; continue; }
      const nx = vn[v * 3] / nl, ny = vn[v * 3 + 1] / nl, nz = vn[v * 3 + 2] / nl;
      const dl = Math.hypot(dx, dy, dz) || 1;
      /* наклон к камере: нормаль против направления на камеру */
      const back = nx * (dx / dl) + ny * (dy / dl) + nz * (dz / dl);
      /* back = dot(нормаль, узел − камера): у поверхности, развёрнутой к камере,
         он отрицательный, у обратной стороны — около +1. Берём не «строго на
         камеру» (капот при взгляде почти по касательной даёт ≈ 0 и выпадал из
         маски), а «нормаль не отвёрнута от камеры больше 69°»: этого хватает,
         чтобы отсечь обратную сторону кузова и обратную стенку тонкой детали. */
      /* «та же сторона»: отсекаем только узлы, чья нормаль смотрит почти
         навстречу нормали под курсором (обратная сторона тонкой стенки).
         -0.6, а не 0: на грубой сетке (куб) нормали соседних граней расходятся
         на 70°, и более строгий порог отрезал бы лицевые узлы. */
      const same = !hasHit || (nx * hx + ny * hy + nz * hz) > -0.6;
      front[v] = (back < 0.35 && same) ? 1 : 0;
    }
  }
  /** Маска узлов, которые нельзя трогать в этом штрихе. */
  let block = null;      // 1 — узел трогать нельзя (этим пользуются закраска и ластик)
  let allow = null;      // 1 — можно (этим пользуются pushBall/imprintShape: у них «only»)
  function makeBlock() {
    const m = new Uint8Array(vk);
    allow = new Uint8Array(vk);
    for (let v = 0; v < vk; v++) {
      const bad = dead[v] || (frontOnly && !front[v]) ? 1 : 0;
      m[v] = bad;
      allow[v] = bad ? 0 : 1;
    }
    block = m;
    return m;
  }

  /** Область, очищенная от нелицевых узлов. Считается один раз на закраску:
      иначе после «Выдавить» нормали на бугре наклоняются и следующее действие
      трогает уже меньше вершин, чем первое. */
  let regionFront = null;
  function frontMaskedRegion() {
    if (!frontOnly) return reg;
    if (regionFront && regionFront.length === vk) return regionFront;
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
    if (!k) return reg;
    if (k === n) { regionFront = reg; return reg; }   // маска совпала с областью
    regionFront = m;
    return m;
  }

  /* ---------- сетка тонкими чёрными линиями ---------- */
  let wire = null, wireOn = false;
  const wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', attr);      // общий атрибут с геометрией
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
    hitCache = null;                       // геометрия изменилась — луч заново
    expandToVertices(topo, pos, src);
    attr.needsUpdate = true;
    /* Вершин стало больше (точка, зеркальная копия) — цвет и нормали обязаны
       быть того же размера, иначе атрибуты разъезжаются и поверхность чёрная:
       flatShading/vertexColors читают их по числу вершин. computeVertexNormals
       переиспользует СТАРЫЙ буфер нормалей, поэтому его надо удалить. */
    /* Цвет тоже обязан совпадать с числом вершин: короткий буфер атрибута
       читается вне длины — новые узлы (борозда, вставка точки) становились
       чёрными. Сверяем не длину массива, а сам атрибут геометрии. */
    const cAttr = geom.attributes.color;
    if (!cAttr || cAttr.count !== attr.count || col.length !== attr.count * 3) {
      if (col.length !== attr.count * 3) {
        const next = new Float32Array(attr.count * 3).fill(1);
        next.set(col.subarray(0, Math.min(col.length, next.length)));
        col = next;
      }
      if (CANON_COL.length < vk * 3) {
        const nc = new Float32Array(vk * 3).fill(1);
        nc.set(CANON_COL.subarray(0, Math.min(CANON_COL.length, nc.length)));
        CANON_COL = nc;
      }
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
      paintMarks();                 // залить цвета по текущему выделению
    }
    if (wireOn) {
      wireGeo.attributes.position.needsUpdate = true;
      if (wireDirty) { wireDirty = false; rebuildWire(); }
    }
    if (withNormals) {
      const nAttr = geom.attributes.normal;
      if (nAttr && nAttr.count !== attr.count) geom.deleteAttribute('normal');
      geom.computeVertexNormals();
      geom.computeBoundingSphere();
      geom.computeBoundingBox();
      if (wireOn) wireGeo.computeBoundingSphere();
    }
  }

  /** Заменить координатный массив целиком и перепривязать ВСЕ атрибуты.
      Иначе у сетки-оверлея остаётся старый буфер — после вставки точки на
      экране получается мусор из линий к несуществующим вершинам. */
  function applySrc(next) {
    src = next;
    attr = new THREE.BufferAttribute(src, 3);
    geom.setAttribute('position', attr);
    wireGeo.setAttribute('position', attr);     // тот же атрибут
    wireDirty = true;                           // сетку пересоберём один раз — в flush()
  }

  /* ---------- история ---------- */
  const hist = [], redo = [];
  function snapshot() {
    return {
      pos: Float32Array.from(pos), reg: reg.slice(), dead: dead.slice(),
      added: added.slice(), n: regionN, del: delTotal, index: liveIndex,
      // массивы топологии заменяются целиком (никогда не правятся на месте),
      // поэтому ссылок достаточно — копия не нужна
      T: { src, index: liveIndex, count, vk },
      ropeOff: ropes.map(r => r.off)      // канат лежит/утоплен — тоже часть правки
    };
  }
  let dirty = false;                 // в правке есть несохранённое — предупредим при закрытии
  function pushHist(snap) {
    dirty = true;
    hist.push(snap || snapshot());
    if (hist.length > 15) hist.shift();
    redo.length = 0;
  }
  function setIndex(ix) {
    liveIndex = ix;
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(ix), 1));
    wireDirty = true;                       // пересоберём один раз, в flush()
  }
  function restore(s) {
    const T = s.T;
    if (T && (T.src !== src || T.count !== count || T.index !== liveIndex)) {
      /* точка вставки добавляла вершину — возвращаем и геометрию, и топологию */
      count = T.count;
      applySrc(Float32Array.from(T.src));
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
    regionN = s.n; delTotal = s.del; regionFront = null;
    nrm = null;
    paintMarks(); free = freeMask(topo, reg);
    updateAddedDots();
    if (s.ropeOff && s.ropeOff.length === ropes.length) {
      ropes.forEach((rp, i) => {
        if (Math.abs(rp.off - s.ropeOff[i]) > 1e-9) { rp.off = s.ropeOff[i]; rebuildRope(rp); syncRopeRow(); }
      });
    }
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
  let rectStart = null;     // рамка: экранные координаты начала протяжки
  let rectLast = null;
  let rectSel = null;       // последняя рамка: {x0,y0,x1,y1} в пикселях
  let regFromRect = false;  // выделение поставлено рамкой (а не кистью)
  let sym = false;          // симметрия по y = 0
  let frontOnly = true;     // трогать только видимую (лицевую) поверхность
  let penTicks = 0;
  let lastPt = null;        // последняя добавленная точка (для протяжки)
  let addedTotal = 0;       // сколько точек добавлено за сеанс
  let ptSnap = null;        // снимок до первой вставки точки за штрих
  let ptMsg = null;         // сообщение инструмента «Точка» (переживает отпускание кнопки)

  const brushR = () => brushMM * U;
  const depthD = () => depthMM * U;

  /* Луч по 100 000 граней кузова — 20–40 мс. За одно событие мыши его спрашивают
     трижды (кольцо кисти + точка попадания + грань), поэтому результат
     запоминаем по координатам курсора: 90 мс на движение → 45. */
  let hitCache = null;
  function rayHit(clientX, clientY) {
    if (hitCache && hitCache.x === clientX && hitCache.y === clientY) return hitCache.h;
    ndc.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObject(mesh, false);
    let h = null;
    if (hits.length) {
      const p = mesh.worldToLocal(hits[0].point.clone());
      h = { p: [p.x, p.y, p.z], faceIndex: hits[0].faceIndex };
    }
    hitCache = { x: clientX, y: clientY, h };
    return h;
  }
  function hit(clientX, clientY) {
    const h = rayHit(clientX, clientY);
    return h ? { x: h.p[0], y: h.p[1], z: h.p[2] } : null;
  }
  function hitPoint(clientX, clientY) {
    const h = rayHit(clientX, clientY);
    return h ? h.p : null;
  }
  /** попадание вместе с номером грани — нужно инструменту «Точка» */
  function hitFace(clientX, clientY) {
    return rayHit(clientX, clientY);
  }
  /** ближайший узел, но не дальше maxD — иначе клик в дырку цеплял бы узел
      с другой стороны кузова */
  function nearestNodeNear(p, skip, maxD) {
    const v = nearestNode(p, skip);
    if (v < 0) return -1;
    const o = v * 3;
    const d = Math.hypot(pos[o] - p[0], pos[o + 1] - p[1], pos[o + 2] - p[2]);
    return d <= maxD ? v : -1;
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

  /* ---------- курсор кисти: кольцо на поверхности под мышью ----------
     Без него размер кисти угадываешь на глаз: кольцо показывает и радиус, и
     точку, куда попадёт инструмент. Живёт в локальных координатах меша, поэтому
     видно и на скруглениях. Цвет — по инструменту. */
  let cursor = null, cursorMat = null;
  function ensureCursor() {
    if (cursor) return cursor;
    if (!mesh || typeof mesh.add !== 'function') return null;
    if (!THREE.RingGeometry || !THREE.Mesh || !THREE.MeshBasicMaterial) return null;
    cursorMat = new THREE.MeshBasicMaterial({
      color: 0xffa63d, transparent: true, opacity: 0.9, side: THREE.DoubleSide,
      depthTest: false, depthWrite: false
    });
    cursor = new THREE.Mesh(new THREE.RingGeometry(0.84, 1, 56), cursorMat);
    cursor.renderOrder = 20;
    cursor.frustumCulled = false;
    cursor.visible = false;
    mesh.add(cursor);                       // в системе координат кузова
    return cursor;
  }
  const CURSOR_COLOR = { select: 0x4fd1c5, erase: 0xf25c5c, pen: 0xffd166, shape: 0xffa63d };
  function hideCursor() { if (cursor) cursor.visible = false; }
  function updateCursor(e) {
    const c = ensureCursor();
    if (!c) return;
    if (!tool || tool === 'point' || tool === 'measure') { c.visible = false; return; }
    const h = e ? hitFace(e.clientX, e.clientY) : null;
    if (!h) { c.visible = false; return; }
    const b = h.faceIndex * 3;
    if (!(b >= 0) || b + 2 >= liveIndex.length) { c.visible = false; return; }
    const i0 = liveIndex[b], i1 = liveIndex[b + 1], i2 = liveIndex[b + 2];
    let n = triNormal(i0, i1, i2, pos);
    const ln = Math.hypot(n[0], n[1], n[2]);
    if (!ln) { c.visible = false; return; }
    n = [n[0] / ln, n[1] / ln, n[2] / ln];
    const R = brushR();
    const p = h.p;
    const off = R * 0.02 + 0.01 * U;         // чуть над поверхностью, без мерцания
    if (c.position) c.position.set(p[0] + n[0] * off, p[1] + n[1] * off, p[2] + n[2] * off);
    if (c.scale) { if (c.scale.setScalar) c.scale.setScalar(R); else c.scale.set(R, R, R); }
    if (c.quaternion && c.quaternion.set) {
      /* поворот оси +z кольца на нормаль: q = [z × n, 1 + z·n] */
      const cx = -n[1], cy = n[0], cz = 0, cw = 1 + n[2];
      const cl = Math.hypot(cx, cy, cz, cw) || 1;
      c.quaternion.set(cx / cl, cy / cl, cz / cl, cw / cl);
    }
    if (cursorMat && cursorMat.color && cursorMat.color.setHex) {
      cursorMat.color.setHex(CURSOR_COLOR[tool] || 0xffa63d);
    }
    c.visible = true;
  }

  /* ---------- шаг штриха ----------
     Мышь приходит редкими событиями: при быстром движении пятна кисти стоят
     через 5–10 мм, и валик выходит «пунктиром» (провалы до нуля). В Blender это
     решает Spacing (10 % кисти) — у нас между событиями ставим промежуточные
     отпечатки с шагом 25 % радиуса. */
  let lastW = null;
  function strokeSegment(p, fn) {
    if (!lastW) { lastW = [p[0], p[1], p[2]]; fn(p); return; }
    const dx = p[0] - lastW[0], dy = p[1] - lastW[1], dz = p[2] - lastW[2];
    const d = Math.hypot(dx, dy, dz);
    /* Шаг 25 % кисти, но не больше 4 отпечатков на событие: иначе на кузове
       (52 657 узлов) одно движение мыши считается 250 мс, а не 30. */
    const step = Math.max(0.25 * brushR(), 0.02 * U);
    const n = Math.min(4, Math.round(d / step));
    for (let i = 1; i < n; i++) {
      const t = i / n;
      fn([lastW[0] + dx * t, lastW[1] + dy * t, lastW[2] + dz * t]);
    }
    fn(p);
    lastW = [p[0], p[1], p[2]];
  }

  /* ---------- 1. выбор участка ---------- */
  function paintAt(c) {
    const skip = block || dead;           // что нельзя трогать в этом штрихе
    let added = paintBall(topo, reg, c, brushR(), pos, skip);
    if (!added) {                         // кисть мельче сетки — берём ближайшую точку
      const v = nearestNode(c, skip);
      if (v >= 0 && !reg[v] && !skip[v]) { reg[v] = 1; added = 1; regFromRect = false; }
    }
    if (added) { regionN += added; regionFront = null; paintMarks(); }
    return added;
  }
  function stamp(clientX, clientY) {
    const c = hitPoint(clientX, clientY);
    if (!c) return;
    strokeSegment(c, q => { paintAt(q); if (sym) paintAt([q[0], -q[1], q[2]]); });
    syncInfo();
  }

  /* ---------- 2. ластик ---------- */
  function markDead(v) {
    if (v < 0 || dead[v] || (block && block[v])) return 0;
    dead[v] = 1; pending[v] = 1; delTotal++;
    if (reg[v]) { reg[v] = 0; regionN--; }
    return 1;
  }
  function eraseAt(c) {
    const r2 = brushR() * brushR();
    let n = 0;
    for (let v = 0; v < vk; v++) {
      const o = v * 3;
      const dx = pos[o] - c[0], dy = pos[o + 1] - c[1], dz = pos[o + 2] - c[2];
      if (dx * dx + dy * dy + dz * dz <= r2) n += markDead(v);
    }
    if (!n) n += markDead(nearestNodeNear(c, dead, brushR() * 1.6));
    return n;
  }
  /** Shift+ластик — точечно: уходит ровно один ближайший узел, кисть не работает. */
  function eraseOneAt(c) {
    return markDead(nearestNodeNear(c, block, brushR() * 1.6));
  }
  function eraseStamp(clientX, clientY) {
    const c = hitPoint(clientX, clientY);
    if (!c) return 0;
    const f = invert ? eraseOneAt : eraseAt;
    let n = 0;
    strokeSegment(c, q => { n += f(q); if (sym) n += f([q[0], -q[1], q[2]]); });
    if (n) { strokeDel += n; paintMarks(); updateDots(); syncInfo(); }
    return n;
  }

  /* ---------- 3. перо ---------- */
  let penMoved = 0;
  let penBase = null;       // положения на начало штриха пера
  /* Валик должен быть высотой «Глубина», а не «глубина × число событий мыши»:
     под пером узлы обрабатываются на каждом движении и копили бы 5–6 глубин.
     Поэтому меряем сдвиг от начала штриха и обрезаем его на «Глубине». */
  function clampPenStroke() {
    if (!penBase) return;
    const lim = depthD();
    for (let v = 0; v < vk; v++) {
      const o = v * 3;
      const dx = pos[o] - penBase[o], dy = pos[o + 1] - penBase[o + 1], dz = pos[o + 2] - penBase[o + 2];
      const l = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (l <= lim) continue;
      const k = lim / l;
      pos[o] = penBase[o] + dx * k;
      pos[o + 1] = penBase[o + 1] + dy * k;
      pos[o + 2] = penBase[o + 2] + dz * k;
    }
  }
  /** Шаг сетки под кистью: примерно 8 узлов поперёк пятна и не грубее глубины.
      Кисть меньше грани иначе двигает одну точку — получается «шип». */
  function detailFor(r) {
    /* Шаг сгущения ≈ диаметр/5 (у Blender dyntopo «Detail» по умолчанию ~10 %
       кисти). Было r/4: на кузове (рёбра ~1 мм) кисть 3 мм дробила пятно до
       0.75 мм — 3000 новых узлов и 700 мс на нажатие ради незаметной разницы. */
    return Math.max(r / 2.5, 0.12 * U);
  }
  /** Самое длинное ребро в пятне кисти: если оно уже не длиннее шага сгущения,
      сгущать нечего. Один проход по граням (O(F)) дешевле восьми проходов
      refineSphere — на кузове это 500 мс на каждое нажатие пера. */
  function localMaxEdge(c, r, thr = 0) {
    const r2 = r * r;
    let mx = 0, cnt = 0;
    for (let f = 0; f < liveIndex.length; f += 3) {
      const i0 = liveIndex[f], i1 = liveIndex[f + 1], i2 = liveIndex[f + 2];
      const o0 = i0 * 3, o1 = i1 * 3, o2 = i2 * 3;
      const d0 = (src[o0] - c[0]) ** 2 + (src[o0 + 1] - c[1]) ** 2 + (src[o0 + 2] - c[2]) ** 2;
      const d1 = (src[o1] - c[0]) ** 2 + (src[o1 + 1] - c[1]) ** 2 + (src[o1 + 2] - c[2]) ** 2;
      const d2 = (src[o2] - c[0]) ** 2 + (src[o2 + 1] - c[1]) ** 2 + (src[o2 + 2] - c[2]) ** 2;
      if (d0 > r2 && d1 > r2 && d2 > r2) continue;
      for (const [a, b] of [[o0, o1], [o1, o2], [o2, o0]]) {
        const e = Math.hypot(src[a] - src[b], src[a + 1] - src[b + 1], src[a + 2] - src[b + 2]);
        if (e > mx) mx = e;
        if (thr && e > thr) cnt++;
      }
    }
    return thr ? cnt : mx;
  }
  /** Перед правкой сгустить пятно кисти (иначе на грубой сетке правки нет). */
  function ensureDetail(c, r, budget = 3000) {
    /* Сгущать незачем, если длинных рёбер в пятне единицы: 3-4 длинных ребра
       дают 1-2 лишние точки, а полный проход стоит 40 мс. */
    const target = detailFor(r);
    if (localMaxEdge(c, r * 1.25, target) < 8) return 0;
    const n = refineSphere(c, r * 1.25, detailFor(r), budget);
    if (n) { ensureTopo(); nrm = null; }
    return n;
  }
  /** Средняя нормаль под кистью: направление, вдоль которого двигаем всё пятно
      (в Blender это «area normal»). Возвращает [x,y,z,вес] для pushBall. */
  function brushNormal(c, r) {
    const n = getNormals(), R = r * r;
    let nx = 0, ny = 0, nz = 0;
    for (let v = 0; v < vk; v++) {
      const o = v * 3;
      const dx = pos[o] - c[0], dy = pos[o + 1] - c[1], dz = pos[o + 2] - c[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > R) continue;
      const t = Math.max(0, 1 - Math.sqrt(d2) / r);
      const w = t * t * (3 - 2 * t);
      nx += n[o] * w; ny += n[o + 1] * w; nz += n[o + 2] * w;
    }
    const l = Math.hypot(nx, ny, nz);
    if (!(l > 1e-9)) return null;
    return [nx / l, ny / l, nz / l, 2.0];      // 2.0 — вес против нормали узла
  }
  function penAt(c, sign) {
    if (++penTicks % 8 === 0) nrm = null;
    const res = pushBall(topo, pos, c, brushR(), sign * depthD(), getNormals(), allow || block || dead,
                         brushNormal(c, brushR()));
    if (!res.moved) {                     // кисть мельче сетки — двигаем одну точку
      const v = nearestNodeNear(c, allow ? null : block, brushR() * 1.6);
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
    let m = 0;
    strokeSegment(c, q => { m += penAt(q, sign) + (sym ? penAt([q[0], -q[1], q[2]], sign) : 0); });
    /* clampFlips перебирает все грани (122 000 на кузове) — на каждое движение
       мыши это ~150 мс. Запускаем его один раз, по отпусканию кнопки. */
    clampPenStroke();
    nrm = null;                            // нормали после обрезки другие
    return m;
  }

  /* ---------- 4. фигура ---------- */
  /** Сгустить сетку в пятне кисти: расщепляем грани, у которых ребро длиннее
      шага, пока в пятне не станет достаточно узлов. Потом эти узлы и вдавливаем
      — фигура получается жёсткой, а не «размазанной» по трём большим граням. */
  function refineAt(center, radius, spacing, maxAdd = 400) {
    const r2 = radius * radius, s2 = spacing * spacing;
    const maxEdge2 = (i0, i1, i2) => {
      const e = (a, b) => {
        const dx = src[a * 3] - src[b * 3], dy = src[a * 3 + 1] - src[b * 3 + 1], dz = src[a * 3 + 2] - src[b * 3 + 2];
        return dx * dx + dy * dy + dz * dz;
      };
      return Math.max(e(i0, i1), e(i1, i2), e(i2, i0));
    };
    const cen = (i0, i1, i2) => [
      (src[i0 * 3] + src[i1 * 3] + src[i2 * 3]) / 3,
      (src[i0 * 3 + 1] + src[i1 * 3 + 1] + src[i2 * 3 + 1]) / 3,
      (src[i0 * 3 + 2] + src[i1 * 3 + 2] + src[i2 * 3 + 2]) / 3];
    /* ближайшая к центру кисти точка грани (нужна и для отбора, и для разреза:
       на грубой сетке грань может быть больше кисти целиком) */
    const near = (i0, i1, i2) => {
      const p = (i) => [src[i * 3], src[i * 3 + 1], src[i * 3 + 2]];
      const A = p(i0), B = p(i1), C = p(i2);
      const ux = B[0] - A[0], uy = B[1] - A[1], uz = B[2] - A[2];
      const vx = C[0] - A[0], vy = C[1] - A[1], vz = C[2] - A[2];
      const wx = center[0] - A[0], wy = center[1] - A[1], wz = center[2] - A[2];
      const d00 = ux * ux + uy * uy + uz * uz, d01 = ux * vx + uy * vy + uz * vz;
      const d11 = vx * vx + vy * vy + vz * vz;
      const d20 = wx * ux + wy * uy + wz * uz, d21 = wx * vx + wy * vy + wz * vz;
      const den = d00 * d11 - d01 * d01;
      let u = 1 / 3, v = 1 / 3, w = 1 / 3;
      if (Math.abs(den) > 1e-14) {
        v = (d11 * d20 - d01 * d21) / den;
        w = (d00 * d21 - d01 * d20) / den;
        u = 1 - v - w;
      }
      /* зажимаем внутрь треугольника: ближайшая точка */
      const cl = (x) => Math.max(0, Math.min(1, x));
      u = cl(u); v = cl(v); w = 1 - u - v;
      if (w < 0) { const k = (v + w) || 1; v /= k; w = 0; u = 1 - v; }
      if (w > 1) { w = 1; u = 0; v = 0; }
      const q = [
        A[0] + ux * v + vx * w, A[1] + uy * v + vy * w, A[2] + uz * v + vz * w];
      const d2 = (q[0] - center[0]) ** 2 + (q[1] - center[1]) ** 2 + (q[2] - center[2]) ** 2;
      return { d2, p: q };
    };
    const list = [];
    for (let f = 0; f < liveIndex.length / 3; f++) {
      const b = f * 3;
      const i0 = liveIndex[b], i1 = liveIndex[b + 1], i2 = liveIndex[b + 2];
      if (maxEdge2(i0, i1, i2) <= s2) continue;
      if (near(i0, i1, i2).d2 > r2) continue;      // грань задевает пятно кисти
      list.push(f);
    }
    /* карта уже существующих узлов — чтобы не вставить точку поверх соседней
       (иначе появляются совпавшие вершины и вырожденные треугольники) */
    const cell = 0.1;
    const grid = new Map();
    const keyOf = (p) => `${Math.round(p[0] / cell)}|${Math.round(p[1] / cell)}|${Math.round(p[2] / cell)}`;
    for (let v = 0; v < vk; v++) {
      const p = [pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]];
      const k = keyOf(p);
      const e = grid.get(k); if (e) e.push(p); else grid.set(k, [p]);
    }
    const tooClose = (p) => {
      const cx = Math.round(p[0] / cell), cy = Math.round(p[1] / cell), cz = Math.round(p[2] / cell);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const e = grid.get(`${cx + dx}|${cy + dy}|${cz + dz}`);
        if (!e) continue;
        for (const q of e) {
          if (Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) < 0.05) return true;
        }
      }
      return false;
    };
    const remember = (p) => {
      const k = keyOf(p);
      const e = grid.get(k); if (e) e.push(p); else grid.set(k, [p]);
    };
    /* сколько узлов уже есть в пятне: если сетка там и так достаточно частая,
       сгущение прекращаем — лишние точки только замедляют клик. Сколько узлов
       должно быть на диске радиуса r при шаге s: πr² / (0.87 s²) ≈ 3.6 r²/s². */
    let inside = 0;
    for (let v = 0; v < vk; v++) {
      const dx = src[v * 3] - center[0], dy = src[v * 3 + 1] - center[1], dz = src[v * 3 + 2] - center[2];
      if (dx * dx + dy * dy + dz * dz <= r2) inside++;
    }
    const target = Math.ceil(3.6 * r2 / (spacing * spacing));
    let added = 0;
    while (list.length && added < maxAdd && inside + added < target) {
      const f = list.shift();
      const b = f * 3;
      if (b + 2 >= liveIndex.length) continue;
      const i0 = liveIndex[b], i1 = liveIndex[b + 1], i2 = liveIndex[b + 2];
      if (maxEdge2(i0, i1, i2) <= s2) continue;
      const nc = near(i0, i1, i2);
      if (nc.d2 > r2) continue;
      /* точку ставим ближайшую к кисти; если она легла на вершину/ребро или
         рядом уже есть узел — сдвигаемся к центру грани, пока место не станет
         свободным (иначе появятся совпавшие узлы и вырожденные треугольники) */
      const c = cen(i0, i1, i2);
      let res = null;
      for (const t of [0, 0.25, 0.5, 0.75, 1]) {
        const p = [nc.p[0] + (c[0] - nc.p[0]) * t,
                   nc.p[1] + (c[1] - nc.p[1]) * t,
                   nc.p[2] + (c[2] - nc.p[2]) * t];
        if (tooClose(p)) continue;
        res = insertPointAt(f, p, false);          // в сгущении иголок нет — перевороты не нужны
        if (res.ok) { remember(p); break; }
      }
      if (!res || !res.ok) continue;
      added++;
      for (let k = 0; k < list.length; k++) if (list[k] > f) list[k] += 2;
      list.push(f, f + 1, f + 2);
    }
    if (added) {
      ensureTopo();
      nrm = null;
      paintMarks();
      flush();
    }
    return added;
  }

  function shapeAt(c, sign) {
    const brush = brushR();
    /* 1) в месте касания строим точки по размеру кисти. Сгущение — бисекцией
       самых длинных рёбер (как у пера): вставка точки «поближе к центру кисти»
       кучковала узлы и оставляла тонкие треугольники (aspect 78 на кубе). */
    /* шаг — не грубее половины глубины: иначе стенка штампа «складывается»
       в иголки (на кубе аспект треугольников доходил до 374) */
    const spacing = Math.max(0.10 * U, Math.min(brush / 4.5, depthD() * 0.5));
    const refined = refineSphere(c, brush * 1.15, spacing, 6000);
    if (refined) { ensureTopo(); nrm = null; }
    /* новые точки должны попасть в маску «можно» этого штриха */
    if (refined) {
      const m = new Uint8Array(vk);
      if (allow && allow.length) for (let v = 0; v < vk; v++) m[v] = v < allow.length ? allow[v] : 1;
      else m.fill(1);
      for (let v = 0; v < vk; v++) if (added[v] && !m[v]) m[v] = 1;
      allow = m; block = null;
    }
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
      size: { r: R, h: Math.max(R * 0.8, depthD()), a: R * 0.8, b: R * 0.8 },
      band: Math.max(R * 0.12, spacing * 0.5)      // узкая кромка — штамп жёсткий
    };
    const pre = Float32Array.from(pos);        // снимок ПОСЛЕ сгущения (узлы уже есть)
    const res = imprintShape(topo, pos, shape, sign * depthD(), n, allow);
    clampFlips(pos, pre, topo.tris);
    return { ...res, refined, pre };
  }
  /* Штрих фигурой: ведёте с зажатой кнопкой — штампы встают вдоль следа с шагом
     0.9 радиуса кисти. Каждый штамп на кузове — это сгущение сетки (0.1–0.6 с),
     поэтому за один штрих не больше SHAPE_MAX_STAMPS. */
  const SHAPE_MAX_STAMPS = 6;
  let shapeCount = 0, shapeLast = null;
  function shapeDrag(clientX, clientY, force = false) {
    if (!force && shapeCount >= SHAPE_MAX_STAMPS) return null;
    const c = hit(clientX, clientY);
    if (!c) return null;
    if (!force && shapeLast) {
      const dx = c[0] - shapeLast[0], dy = c[1] - shapeLast[1], dz = c[2] - shapeLast[2];
      if (Math.sqrt(dx * dx + dy * dy + dz * dz) < brushR() * 0.9) return null;  // тот же штамп
    }
    /* след ушёл на другую сторону кузова — маска лицевых узлов пересчитывается */
    computeFront(c);
    makeBlock();
    const res = shapeStamp(clientX, clientY, invert ? -1 : 1);
    if (!res) return null;
    shapeLast = c; shapeCount++;
    nrm = null;
    flush();
    if (info) {
      info.textContent = `фигура «${SHAPE_NAMES[res.kind]}» ${invert ? 'вытянула' : 'вдавила'} ` +
        `поверхность: ${res.moved} вершин на ${(res.maxd / U).toFixed(2)} мм` +
        (res.refined ? ` · в пятне добавлено ${res.refined} точек` : '') +
        (shapeCount > 1 ? ` · штамп ${shapeCount} из ${SHAPE_MAX_STAMPS}` : '');
    }
    return res;
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

  /* ---------- 8. борозда: ровный прямой желоб по протяжке ----------
     Пользователь ведёт линию по кузову; по отпускании кнопки вдоль этой линии
     ставится НОВЫЙ набор точек (сгущение сетки в полосе), и только эти точки
     вдавливаются по профилю капсулы: плоское дно постоянной глубины и узкая
     кромка. Кузов вне полосы не двигается вовсе — никакого «прогиба скелета». */

  /* Расстояние от точки до пути (ломаной) — квадрат, и номер сегмента. */
  function distToPath(pts, p, out = null) {
    let best = Infinity, bestSeg = 0, bestT = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
      const L2 = dx * dx + dy * dy + dz * dz;
      let t = L2 > 1e-12 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy + (p[2] - a[2]) * dz) / L2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const qx = p[0] - (a[0] + dx * t), qy = p[1] - (a[1] + dy * t), qz = p[2] - (a[2] + dz * t);
      const d2 = qx * qx + qy * qy + qz * qz;
      if (d2 < best) { best = d2; bestSeg = i; bestT = t; }
      else if (d2 > best * 8 + 4) continue;      // отрезок заведомо далеко
    }
    if (out) { out.seg = bestSeg; out.t = bestT; }
    return best;
  }
  /* Проекция точки на путь: lat — расстояние до оси, s — длина по дуге от начала,
     beyond — −1/1, если точка за торцом (нужно для ПЛОСКИХ торцов желоба). */
  function pathProj(pts, p) {
    let best = Infinity, bestSeg = 0, bestT = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
      const L2 = dx * dx + dy * dy + dz * dz;
      let t = L2 > 1e-12 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy + (p[2] - a[2]) * dz) / L2 : 0;
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      const qx = p[0] - (a[0] + dx * t), qy = p[1] - (a[1] + dy * t), qz = p[2] - (a[2] + dz * t);
      const d2 = qx * qx + qy * qy + qz * qz;
      if (d2 < best) { best = d2; bestSeg = i; bestT = t; }
    }
    let s = 0;
    for (let i = 0; i < bestSeg; i++) {
      const a = pts[i], b = pts[i + 1];
      s += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    }
    const A = pts[bestSeg], B = pts[bestSeg + 1];
    s += bestT * Math.hypot(B[0] - A[0], B[1] - A[1], B[2] - A[2]);
    /* за торцом? смотрим, с какой стороны от крайних сечений лежит точка */
    let beyond = 0;
    if (bestT <= 1e-9 && bestSeg === 0) {
      const t0 = [B[0] - A[0], B[1] - A[1], B[2] - A[2]];
      const q = [p[0] - A[0], p[1] - A[1], p[2] - A[2]];
      if (q[0] * t0[0] + q[1] * t0[1] + q[2] * t0[2] < 0) beyond = -1;
    } else if (bestT >= 1 - 1e-9 && bestSeg === pts.length - 2) {
      const A2 = pts[pts.length - 2], B2 = pts[pts.length - 1];
      const t1 = [B2[0] - A2[0], B2[1] - A2[1], B2[2] - A2[2]];
      const q = [p[0] - B2[0], p[1] - B2[1], p[2] - B2[2]];
      if (q[0] * t1[0] + q[1] * t1[1] + q[2] * t1[2] > 0) beyond = 1;
    }
    return { lat: Math.sqrt(best), s, beyond, seg: bestSeg, t: bestT };
  }
  /* Длина пути в единицах модели */
  function pathLen(pts) {
    let L = 0;
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      L += Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    }
    return L;
  }
  /* Путь по коже: в плане прямой, но прижатый к поверхности — на каждом шаге
     точка прижимается лучом снаружи по последней нормали. Борозда повторяет
     форму борта (как разметка по кузову), а не режет его по хорде. */
  function surfacePath(a, b) {
    const A = new THREE.Vector3(a[0], a[1], a[2]);
    const B = new THREE.Vector3(b[0], b[1], b[2]);
    const seg = new THREE.Vector3().subVectors(B, A);
    const dist = seg.length();
    if (dist < 1e-6) return [];
    const n = Math.max(2, Math.min(160, Math.ceil(dist / Math.max(1.0 * U, dist / 160))));
    const pts = [], nrms = [];
    /* Стартовое направление лучей — нормаль поверхности ПОД КУРСОРОМ. Раньше
       было жёстко (0,1,0): на левом борту кузова путь начинался на правом,
       и борозда резалась не там, где провели линию. */
    const vn0 = getNormals();
    const nvStart = nearestNode(a);
    let sx = 0, sy = 1, sz = 0;
    if (nvStart >= 0) {
      const l = Math.hypot(vn0[nvStart * 3], vn0[nvStart * 3 + 1], vn0[nvStart * 3 + 2]);
      if (l > 1e-9) { sx = vn0[nvStart * 3] / l; sy = vn0[nvStart * 3 + 1] / l; sz = vn0[nvStart * 3 + 2] / l; }
    }
    const nrm = new THREE.Vector3(sx, sy, sz);
    let first = true;
    for (let i = 1; i <= n; i++) {
      const target = new THREE.Vector3().copy(A).addScaledVector(seg, i / n);
      const o = mesh.localToWorld(target.clone().addScaledVector(nrm, 60 * U));
      const d = nrm.clone().transformDirection(mesh.matrixWorld).negate();
      ray.set(o, d);
      ray.near = 0; ray.far = 200 * U;
      const hits = ray.intersectObject(mesh, false);
      if (!hits.length) break;
      const h = hits[0];
      const p = mesh.worldToLocal(h.point.clone());
      /* первая точка пути может оказаться не на поверхности (клик мимо) */
      if (first) { pts[0] = [p.x, p.y, p.z]; first = false; }
      else pts.push([p.x, p.y, p.z]);
      if (h.face) nrm.copy(h.face.normal).normalize();
      nrms[pts.length - 1] = [nrm.x, nrm.y, nrm.z];
    }
    /* доводим путь до самой точки отпускания: иначе желоб короче протяжки */
    if (pts.length >= 2) {
      const last = pts[pts.length - 1];
      const dl = Math.hypot(b[0] - last[0], b[1] - last[1], b[2] - last[2]);
      if (dl > 0.05) {
        const nn = nrms[pts.length - 1];
        const o = mesh.localToWorld(new THREE.Vector3(b[0], b[1], b[2]).addScaledVector(
          new THREE.Vector3(nn[0], nn[1], nn[2]), 60 * U));
        ray.set(o, new THREE.Vector3(nn[0], nn[1], nn[2]).transformDirection(mesh.matrixWorld).negate());
        ray.near = 0; ray.far = 200 * U;
        const h = ray.intersectObject(mesh, false);
        if (h.length) {
          const q = mesh.worldToLocal(h[0].point.clone());
          pts.push([q.x, q.y, q.z]);
          nrms.push([nn[0], nn[1], nn[2]]);
        }
      }
    }
    for (let i = 0; i < pts.length; i++) if (!nrms[i]) nrms[i] = nrms[i - 1] || [0, 0, 1];
    if (pts.length >= 2) { pts.nrms = nrms; return pts; }
    return [];
  }

  /* Сгущение вдоль полосы делением РЁБЕР (бисекция), а не вставкой точки внутрь
     грани: так треугольники остаются ровными, а не превращаются в осколки.
     Ступенями: сначала широкая и грубая зона, потом узкая и мелкая — иначе на
     краю полосы остаются длинные «веера» из больших граней и кромка рвётся. */
  function refineBand(pts, r, spacing, budget = 14000, opts = {}) {
    /* Первая ступень должна ДОТЯНУТЬСЯ до полосы: если сетка крупнее полосы
       (плита 8 мм, а желоб 1 мм), рёбра вообще не попадают в отбор, и борозда
       молча не делается. Поэтому грубая ступень шире местного ребра. */
    let eMax = spacing * 4;
    {
      let bx0 = Infinity, by0 = Infinity, bz0 = Infinity, bx1 = -Infinity, by1 = -Infinity, bz1 = -Infinity;
      for (const q of pts) {
        if (q[0] < bx0) bx0 = q[0]; if (q[0] > bx1) bx1 = q[0];
        if (q[1] < by0) by0 = q[1]; if (q[1] > by1) by1 = q[1];
        if (q[2] < bz0) bz0 = q[2]; if (q[2] > bz1) bz1 = q[2];
      }
      const pad = r * 6 + 4;
      for (let i = 0; i < liveIndex.length; i += 3) {
        const a = liveIndex[i], b = liveIndex[i + 1], c = liveIndex[i + 2];
        const inside = (v) => src[v * 3] >= bx0 - pad && src[v * 3] <= bx1 + pad &&
          src[v * 3 + 1] >= by0 - pad && src[v * 3 + 1] <= by1 + pad &&
          src[v * 3 + 2] >= bz0 - pad && src[v * 3 + 2] <= bz1 + pad;
        if (!inside(a) || !inside(b) || !inside(c)) continue;
        const e = (i1, i2) => Math.hypot(src[i1 * 3] - src[i2 * 3], src[i1 * 3 + 1] - src[i2 * 3 + 1],
                                         src[i1 * 3 + 2] - src[i2 * 3 + 2]);
        const m = Math.max(e(a, b), e(b, c), e(c, a));
        if (m > eMax) eMax = m;
      }
    }
    /* Длинное ребро в стороне от полосы (решётка, арка, внутренности бампера)
       не должно растягивать грубую ступень на пол-кузова: иначе сгущение
       переделывает поверхность далеко от канала, и затенение «плывёт» —
       на кузове это выглядело как вмятина в 10 мм вокруг желоба 4 мм. */
    /* r — радиус желоба: кромку режем по нему во всех ступенях, а отбор делаем
       шире и грубее к краю, чтобы не было «вееров» из больших граней.
       Ступени НЕ уже 3.2r: если сетка крупнее полосы (плита 8 мм, желоб 1 мм),
       грубая ступень должна дотянуться до рёбер, иначе борозда молча не
       делается — проверено стендом «борозда/плоскость 8мм». */
    /* Ступени держим у полосы (не шире 2r). Грубая сетка не страдает: бокс в
       bandPass расширен на длину ребра, поэтому ребро, ПЕРЕСЕКАЮЩЕЕ полосу,
       делится в любой ступени, даже если оба его конца далеко от пути.
       Раньше ступени были шириной 3.2r — на изогнутой панели это дробило
       длинные рёбра в 7 мм от канала, и вместо канала получалась широкая
       мятая вмятина. */
    /* Ступени — «воронка» от 3 шагов каната к каналу: если сгущение обрывается
       у кромки, крупная грань соединяет узел в стенке канала с узлом в 4 мм от
       него, и затенение рисует «лепестки» — самое заметное, что было в первом
       варианте прожига. Теперь размер грани спадает плавно. */
    const stages = [
      { r: r * 3.4, s: Math.max(spacing * 6, eMax / 2) },
      { r: r * 2.7, s: Math.max(spacing * 4.5, eMax / 4) },
      { r: r * 2.1, s: spacing * 3.5 },
      { r: r * 1.55, s: spacing * 2.2 },
      { r: r * 1.05, s: spacing },
    ];
    let total = 0;
    const padE = Math.min(Math.max(eMax, spacing * 4), r * 6);
    for (const st of stages) {
      if (total >= budget) break;
      total += bandPass(pts, st.r, st.s, budget - total, r, st.r + padE);
    }
    /* доп. проходы «до отказа»: полоса в канале должна стать ровной сеткой,
       иначе вдавливание идёт по разным по длине граням и появляются заломы */
    for (let e = 0; e < (opts.extraPass || 0) && total < budget; e++) {
      const add = bandPass(pts, r * 1.02, spacing, budget - total, r, r * 1.02 + padE);
      total += add;
      if (!add) break;
    }
    return total;
  }

  function bandPass(pts, r, spacing, room, rimR = null, pad = null) {
    const s2 = spacing * spacing, r2 = r * r;
    const rr2 = (rimR || r) * (rimR || r);
    let bx0 = Infinity, by0 = Infinity, bz0 = Infinity, bx1 = -Infinity, by1 = -Infinity, bz1 = -Infinity;
    for (const q of pts) {
      if (q[0] < bx0) bx0 = q[0]; if (q[0] > bx1) bx1 = q[0];
      if (q[1] < by0) by0 = q[1]; if (q[1] > by1) by1 = q[1];
      if (q[2] < bz0) bz0 = q[2]; if (q[2] > bz1) bz1 = q[2];
    }
    const padV = pad || r + 2;         // с запасом на длину ребра (см. refineBand)
    const far = (p) => p[0] < bx0 - padV || p[0] > bx1 + padV || p[1] < by0 - padV ||
      p[1] > by1 + padV || p[2] < bz0 - padV || p[2] > bz1 + padV;
    const pt = (i) => [src[i * 3], src[i * 3 + 1], src[i * 3 + 2]];
    const lerp3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

    let total = 0;
    for (let pass = 0; pass < 6 && total < room; pass++) {
      /* 1) какие рёбра делим: внутри полосы — по длине, на кромке — в точке
            касания (даёт ровный край желоба) */
      const cuts = new Map();               // key → [x,y,z] новой вершины
      /* ключ по каноническим узлам: швы 3MF/GLB иначе делятся дважды */
      const keyOf = (a, b) => { const ca = canon[a], cb = canon[b]; return ca < cb ? ca * 1e6 + cb : cb * 1e6 + ca; };
      /* точки разрезов не должны совпадать с соседними — иначе нулевые рёбра */
      const cellC = 0.25;
      const clashD = 0.08;
      const cGrid = new Map();
      const cKey = (p) => `${Math.round(p[0] / cellC)}|${Math.round(p[1] / cellC)}|${Math.round(p[2] / cellC)}`;
      const clashC = (p) => {
        const cx = Math.round(p[0] / cellC), cy = Math.round(p[1] / cellC), cz = Math.round(p[2] / cellC);
        for (let d = -1; d <= 1; d++) for (let e = -1; e <= 1; e++) for (let f2 = -1; f2 <= 1; f2++) {
          const e2 = cGrid.get(`${cx + d}|${cy + e}|${cz + f2}`);
          if (!e2) continue;
          for (const q of e2) if (Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) < clashD) return true;
        }
        return false;
      };
      const takeC = (p) => { const k = cKey(p); const e = cGrid.get(k); if (e) e.push(p); else cGrid.set(k, [p]); };
      const mark = (ia, ib) => {
        const k = keyOf(ia, ib);
        if (cuts.has(k)) return;
        const A = pt(ia), B = pt(ib);
        if (far(A) && far(B)) return;
        const da = distToPath(pts, A), db = distToPath(pts, B);
        /* ребро может целиком лежать СНАРУЖИ полосы, но пересекать её
           (крупная грань поперёк желоба): смотрим расстояние от середины
           и от четвертей ребра до пути */
        if (da > r2 && db > r2) {
          let near = Infinity;
          for (const t of [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875]) {
            const q = distToPath(pts, lerp3(A, B, t));
            if (q < near) near = q;
            if (near <= r2) break;
          }
          if (near > r2) return;                            // ребро далеко от полосы
        }
        const rimA = da <= rr2, rimB = db <= rr2;          // кромка желоба
        const inA = da <= r2, inB = db <= r2;              // полоса отбора (шире)
        const len2 = (B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2 + (B[2] - A[2]) ** 2;
        /* 1) ребро пересекает кромку — режем ровно в точке касания: кромка
              становится линией рёбер, а не рваной границей */
        if (rimA !== rimB) {
          if (len2 <= s2 * 4) return;                     // у кромки уже мелко
          /* Кромку режем ПОСЕРЕДИНЕ ребра, а не в точке касания: бисекция
             ставила точку на самую кромку, и там вырастала стена из тонких
             «игл» — в затенении это выглядело как мятая бумага. */
          let lo = 0, hi = 1;
          for (let it = 0; it < 14; it++) {
            const m = (lo + hi) / 2;
            if (distToPath(pts, lerp3(A, B, m)) <= rr2) lo = m; else hi = m;
          }
          const cut = lerp3(A, B, (lo + hi) / 2);
          const guard = 0.12;
          if (Math.min(Math.hypot(cut[0] - A[0], cut[1] - A[1], cut[2] - A[2]),
                       Math.hypot(cut[0] - B[0], cut[1] - B[1], cut[2] - B[2])) < guard) return;
          if (clashC(cut)) return;
          takeC(cut); cuts.set(k, cut);
          return;
        }
        /* 2) ребро целиком в полосе (внутри кромки или в пояске вокруг неё) —
              делим, пока оно длиннее шага: и дно, и поясок становятся мелкими */
        if (len2 <= s2) return;
        /* конец снаружи, но ребро пересекает полосу — режем середину, пока
           ребро не станет короче шага (иначе на крупной сетке полоса не
           наполняется и борозда не делается вовсе) */
        const p = lerp3(A, B, 0.5);
        if (!inA || !inB) {
          if (distToPath(pts, p) > r2) return;
        }
        if (clashC(p)) return;
        takeC(p); cuts.set(k, p);
      };
      /* Красное деление: грань, которая заметно крупнее шага, но задевает
         полосу, делим ЦЕЛИКОМ (три середины), а не одно ребро. Иначе из грани
         18×18 мм получался «лепесток» 18×0.5 мм — он и рисовал мятые тени
         вокруг канала на крупной панели кузова. */
      const cutMid = (ia, ib) => {
        const k = keyOf(ia, ib);
        if (cuts.has(k)) return;
        const A = pt(ia), B = pt(ib);
        const len2 = (B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2 + (B[2] - A[2]) ** 2;
        if (len2 <= s2) return;
        const p = lerp3(A, B, 0.5);
        if (far(p) || clashC(p)) return;
        takeC(p); cuts.set(k, p);
      };
      const bigFace = opts.red ? s2 * 36 : Infinity;              // сторона больше 6 шагов
      const nf0 = liveIndex.length / 3;
      for (let f = 0; f < nf0; f++) {
        const b3 = f * 3;
        const i0 = liveIndex[b3], i1 = liveIndex[b3 + 1], i2 = liveIndex[b3 + 2];
        const A0 = pt(i0), A1 = pt(i1), A2 = pt(i2);
        const e2 = (a, b) => (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2;
        if (Math.max(e2(A0, A1), e2(A1, A2), e2(A2, A0)) > bigFace) {
          const c = [(A0[0] + A1[0] + A2[0]) / 3, (A0[1] + A1[1] + A2[1]) / 3, (A0[2] + A1[2] + A2[2]) / 3];
          if (distToPath(pts, c) <= (r * 1.8) * (r * 1.8)) {
            cutMid(i0, i1); cutMid(i1, i2); cutMid(i2, i0);
            continue;                             // этого прохода грани хватит
          }
        }
        mark(i0, i1);
        mark(i1, i2);
        mark(i2, i0);
      }
      if (!cuts.size) break;
      /* бюджет: если рёбер больше, чем осталось, берём самые короткие ребра
         в полосе (они держат кромку) — остальное подождёт до следующего захода */
      const keys = [...cuts.keys()];
      if (keys.length > room - total) break;   // целиком проход или никак
      const mid = new Map();                // key → файловый индекс новой вершины
      const out = [];
      for (const k of keys) { mid.set(k, count + out.length); out.push(cuts.get(k)); }
      if (!out.length) break;

      /* 2) новые вершины и индекс — одним пакетом */
      const kNew = out.length;
      const newSrc = new Float32Array((count + kNew) * 3);
      newSrc.set(src);
      const newCol = new Float32Array((count + kNew) * 3).fill(1);
      newCol.set(col);
      const node0 = vk;
      resizeNodeArrays(vk + kNew);
      const nc2 = new Int32Array(count + kNew);
      nc2.set(canon);
      const nr2 = new Int32Array(vk + kNew);
      nr2.set(topo.rep);
      for (let i = 0; i < kNew; i++) {
        const file = count + i, node = node0 + i;
        newSrc[file * 3] = out[i][0]; newSrc[file * 3 + 1] = out[i][1]; newSrc[file * 3 + 2] = out[i][2];
        nc2[file] = node; nr2[node] = file;
        pos[node * 3] = pos0[node * 3] = out[i][0];
        pos[node * 3 + 1] = pos0[node * 3 + 1] = out[i][1];
        pos[node * 3 + 2] = pos0[node * 3 + 2] = out[i][2];
        added[node] = 1;
      }
      count += kNew; vk += kNew;
      col = newCol;
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));

      const ni = new Int32Array(liveIndex.length + 3 * kNew * 2);
      let w = 0;
      const push = (a, b, c) => { ni[w++] = a; ni[w++] = b; ni[w++] = c; };
      for (let f = 0; f < nf0; f++) {
        const b3 = f * 3;
        const i0 = liveIndex[b3], i1 = liveIndex[b3 + 1], i2 = liveIndex[b3 + 2];
        const m01 = mid.get(keyOf(i0, i1)), m12 = mid.get(keyOf(i1, i2)), m20 = mid.get(keyOf(i2, i0));
        const n = (m01 !== undefined ? 1 : 0) + (m12 !== undefined ? 1 : 0) + (m20 !== undefined ? 1 : 0);
        if (n === 0) { push(i0, i1, i2); continue; }
        if (n === 3) {
          push(i0, m01, m20); push(i1, m12, m01); push(i2, m20, m12); push(m01, m12, m20);
          continue;
        }
        if (n === 1) {
          if (m01 !== undefined) { push(i0, m01, i2); push(m01, i1, i2); }
          else if (m12 !== undefined) { push(i1, m12, i0); push(m12, i2, i0); }
          else { push(i2, m20, i1); push(m20, i0, i1); }
          continue;
        }
        /* два ребра: отрезаем угол у общей вершины */
        if (m01 !== undefined && m12 !== undefined) {           // общая i1
          push(i1, m12, m01); push(i0, m01, m12); push(i0, m12, i2);
        } else if (m12 !== undefined && m20 !== undefined) {     // общая i2
          push(i2, m20, m12); push(i1, m12, m20); push(i1, m20, i0);
        } else {                                                 // общая i0
          push(i0, m01, m20); push(i2, m20, m01); push(i2, m01, i1);
        }
      }
      total += kNew;
      topo.canon = nc2; topo.rep = nr2; topo.vk = vk; topo.count = count; topo.pos0 = pos0;
      canon = nc2;
      applySrc(newSrc);
      setIndex(ni.subarray(0, w));
      topoDirty = true; nrm = null;
    }
    return total;
  }


  /* ---------- сгущение под кистью (dyntopo «по-простому») ----------
     Узкое место всех кистей: кисть меньше грани двигает одну точку и получается
     «шип» вместо купола. Поэтому перед правкой пятно кисти сгущается делением
     самых ДЛИННЫХ рёбер треугольника (бисекция Ривары): режем не короткое ребро,
     пока рядом длинное, иначе рядом навсегда остаются тонкие треугольники.
     Позиции и индексы вставляются пакетом, как в полосе борозды. */
  function refineSphere(c, R, target, budget = 4000, inRegion = null) {
    const R2 = R * R, s2 = target * target;
    const distC = (p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2]);
    const pt = (i) => [src[i * 3], src[i * 3 + 1], src[i * 3 + 2]];
    const len2 = (A, B) => (B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2 + (B[2] - A[2]) ** 2;
    let total = 0;
    /* Каждый проход — полный обход граней (122 000 на кузове). Восемь проходов
       давали 700 мс; бисекция Ривары делит ребро вдвое, так что трёх проходов
       хватает (4 мм → 0.5 мм). */
    for (let pass = 0; pass < 3 && total < budget; pass++) {
      const cuts = new Map();
      /* ключ ребра — по КАНОНИЧЕСКИМ узлам: в 3MF/GLB вершины на швах
         размножены, и одно и то же ребро иначе выглядит как два разных —
         сетка рвётся трещиной (T-стык) и грани «выворачиваются». */
      const keyOf = (a, b) => { const ca = canon[a], cb = canon[b]; return ca < cb ? ca * 1e6 + cb : cb * 1e6 + ca; };
      const cellC = 0.25 * Math.max(1, target / 1);
      const cGrid = new Map();
      const cKey = (p) => `${Math.round(p[0] / cellC)}|${Math.round(p[1] / cellC)}|${Math.round(p[2] / cellC)}`;
      const clashC = (p) => {
        const cx = Math.round(p[0] / cellC), cy = Math.round(p[1] / cellC), cz = Math.round(p[2] / cellC);
        for (let d = -1; d <= 1; d++) for (let e = -1; e <= 1; e++) for (let f2 = -1; f2 <= 1; f2++) {
          const e2 = cGrid.get(`${cx + d}|${cy + e}|${cz + f2}`);
          if (!e2) continue;
          for (const q of e2) if (Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) < target * 0.35) return true;
        }
        return false;
      };
      const takeC = (p) => { const k = cKey(p); const e = cGrid.get(k); if (e) e.push(p); else cGrid.set(k, [p]); };
      const regCut = new Set();
      const mark = (ia, ib) => {
        const k = keyOf(ia, ib);
        if (cuts.has(k)) return;
        const A = pt(ia), B = pt(ib);
        /* ребро берём, если оно задевает пятно кисти */
        const dA = distC(A), dB = distC(B);
        if (dA > R && dB > R) return;
        /* Шаг РАСТЁТ к краю пятна (graded refinement из Blender dyntopo):
           в центре режем до target, у края допускаем до 3×target. Резкая
           граница «мелко | крупно» давала длинные тонкие треугольники —
           минимальный угол падал до 2.7°. */
        const dm = Math.min(dA, dB);
        const grow = 1 + 2 * Math.min(1, dm / R);
        if (len2(A, B) <= s2 * grow * grow) return;   // уже мелко
        if (dead[canon[ia]] || dead[canon[ib]]) return; // ластик уже пометил
        const p = [(A[0] + B[0]) / 2, (A[1] + B[1]) / 2, (A[2] + B[2]) / 2];
        if (clashC(p)) return;
        takeC(p); cuts.set(k, p);
        /* узел попадёт в закраску, только если оба конца ребра были в ней */
        if (inRegion && inRegion(ia, ib)) regCut.add(k);
      };
      const nf0 = liveIndex.length / 3;
      for (let f = 0; f < nf0; f++) {
        const b3 = f * 3;
        const i0 = liveIndex[b3], i1 = liveIndex[b3 + 1], i2 = liveIndex[b3 + 2];
        const P0 = pt(i0), P1 = pt(i1), P2 = pt(i2);
        /* грань не задевает пятно — пропускаем */
        const dmin = Math.min(distC(P0), distC(P1), distC(P2));
        if (dmin > R + Math.sqrt(Math.max(len2(P0, P1), len2(P1, P2), len2(P2, P0)))) continue;
        /* режем САМОЕ ДЛИННОЕ ребро грани (правило Ривары) */
        const e01 = len2(P0, P1), e12 = len2(P1, P2), e20 = len2(P2, P0);
        if (e01 >= e12 && e01 >= e20) mark(i0, i1);
        else if (e12 >= e20) mark(i1, i2);
        else mark(i2, i0);
      }
      if (!cuts.size) break;
      const keys = [...cuts.keys()];
      if (keys.length > budget - total) break;
      const mid = new Map();
      const out = [];
      for (const k of keys) { mid.set(k, count + out.length); out.push(cuts.get(k)); }
      const kNew = out.length;
      const newSrc = new Float32Array((count + kNew) * 3);
      newSrc.set(src);
      const newCol = new Float32Array((count + kNew) * 3).fill(1);
      newCol.set(col);
      const node0 = vk;
      resizeNodeArrays(vk + kNew);
      const nc2 = new Int32Array(count + kNew); nc2.set(canon);
      const nr2 = new Int32Array(vk + kNew); nr2.set(topo.rep);
      for (let i = 0; i < kNew; i++) {
        const file = count + i, node = node0 + i;
        newSrc[file * 3] = out[i][0]; newSrc[file * 3 + 1] = out[i][1]; newSrc[file * 3 + 2] = out[i][2];
        nc2[file] = node; nr2[node] = file;
        pos[node * 3] = pos0[node * 3] = out[i][0];
        pos[node * 3 + 1] = pos0[node * 3 + 1] = out[i][1];
        pos[node * 3 + 2] = pos0[node * 3 + 2] = out[i][2];
        added[node] = 1;
        if (inRegion && regCut.has(keys[i])) { reg[node] = 1; regionN++; }
      }
      count += kNew; vk += kNew;
      col = newCol;
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const ni = new Int32Array(liveIndex.length + 3 * kNew * 2);
      let w = 0;
      const push = (a, b, c2) => { ni[w++] = a; ni[w++] = b; ni[w++] = c2; };
      for (let f = 0; f < nf0; f++) {
        const b3 = f * 3;
        const i0 = liveIndex[b3], i1 = liveIndex[b3 + 1], i2 = liveIndex[b3 + 2];
        const m01 = mid.get(keyOf(i0, i1)), m12 = mid.get(keyOf(i1, i2)), m20 = mid.get(keyOf(i2, i0));
        const n = (m01 !== undefined ? 1 : 0) + (m12 !== undefined ? 1 : 0) + (m20 !== undefined ? 1 : 0);
        if (n === 0) { push(i0, i1, i2); continue; }
        if (n === 3) { push(i0, m01, m20); push(i1, m12, m01); push(i2, m20, m12); push(m01, m12, m20); continue; }
        if (n === 1) {
          if (m01 !== undefined) { push(i0, m01, i2); push(m01, i1, i2); }
          else if (m12 !== undefined) { push(i1, m12, i0); push(m12, i2, i0); }
          else { push(i2, m20, i1); push(m20, i0, i1); }
          continue;
        }
        if (m01 !== undefined && m12 !== undefined) { push(i1, m12, m01); push(i0, m01, m12); push(i0, m12, i2); }
        else if (m12 !== undefined && m20 !== undefined) { push(i2, m20, m12); push(i1, m12, m20); push(i1, m20, i0); }
        else { push(i0, m01, m20); push(i2, m20, m01); push(i2, m01, i1); }
      }
      total += kNew;
      topo.canon = nc2; topo.rep = nr2; topo.vk = vk; topo.count = count; topo.pos0 = pos0;
      canon = nc2;
      applySrc(newSrc);
      setIndex(ni.subarray(0, w));
      topoDirty = true; nrm = null;
    }
    return total;
  }

  /** Прорезать канал вдоль пути: сгустить полосу и вдавить профиль.
      Один и тот же резец нужен «Борозде» (путь — прямая между нажатием и
      отпусканием) и «Канату» (путь произвольной формы, привязанный к граням).
      path.nrms — нормали поверхности вдоль пути: всё сечение едет в одну сторону,
      как настоящий резец. hw — полуширина, depth — глубина в центре. */
  function carvePath(path, hw, depth, opts = {}) {
    const t0 = performance.now();
    /* 1) новые точки: густота по шагу, мелкому относительно ширины желоба */
    const sp = Math.max(0.16 * U, hw / (opts.spd || 3.5));   // шаг сгущения
    const added = refineBand(path, hw, sp, opts.limit || 14000, opts);
    ensureTopo();
    paintMarks();
    flush();
    /* «мелкие» узлы: у всех граней вокруг ребро не длиннее 2.2 шага. Вдавливаем
       только их — тогда на грубой сетке не появляются складки из больших граней. */
    const fineEdge2 = (2.2 * sp) * (2.2 * sp);
    const fine = new Uint8Array(vk);
    for (let i = 0; i < liveIndex.length; i += 3) {
      const a = liveIndex[i], b = liveIndex[i + 1], c = liveIndex[i + 2];
      const e = (i1, i2) => (src[i1 * 3] - src[i2 * 3]) ** 2 + (src[i1 * 3 + 1] - src[i2 * 3 + 1]) ** 2 +
        (src[i1 * 3 + 2] - src[i2 * 3 + 2]) ** 2;
      if (Math.max(e(a, b), e(b, c), e(c, a)) > fineEdge2) continue;
      fine[a] = 1; fine[b] = 1; fine[c] = 1;
    }
    const tRefine = performance.now();

    /* 2) вдавливание ровно по полосе. Маска «лицевых» граней здесь вредна:
       полоса уже приклеена к поверхности лучом снаружи, а из-за наклона борта
       половина желоба оказывалась «отвёрнутой» и не вдавливалась. */
    const nv = getNormals();
    const pn = path.nrms || null;      // нормали поверхности вдоль пути
    const pre = Float32Array.from(pos);
    const pressed = [];
    let moved = 0, maxd = 0, skipped = 0;
    /* Отступ бокса — от полосы, а не «2 единицы»: при широком канале (канат
       ⌀5 мм) узлы у самой кромки оказывались за боксом и оставались на месте —
       стенка канала получалась обрывом. */
    const pad = hw * 1.2 + 2 * sp;
    const bx0 = Math.min(...path.map(q => q[0])) - pad, bx1 = Math.max(...path.map(q => q[0])) + pad;
    const by0 = Math.min(...path.map(q => q[1])) - pad, by1 = Math.max(...path.map(q => q[1])) + pad;
    const bz0 = Math.min(...path.map(q => q[2])) - pad, bz1 = Math.max(...path.map(q => q[2])) + pad;
    for (let v = 0; v < vk; v++) {
      const o = v * 3;
      const px = pos[o], py = pos[o + 1], pz = pos[o + 2];
      if (px < bx0 || px > bx1 || py < by0 || py > by1 || pz < bz0 || pz > bz1) continue;
      const pr = pathProj(path, [px, py, pz]);
      if (pr.beyond !== 0) continue;        // торцы желоба плоские: за торцом не режем
      const d2 = pr.lat * pr.lat;
      if (pr.lat > hw) continue;
      /* Кто вдавливается: глубоко внутри полосы — всегда (там сгущение уже
         довело сетку), у самой кромки — только мелкие грани. Раньше требовали
         «мелких» ВЕЗДЕ, а грань на кромке всегда имеет ребро наружу — и на
         грубой сетке не вдавливалось ни одного узла (глубина 0). */
      if (!fine[v] && pr.lat > hw * 0.94) continue;
      /* Профиль — ПОЛОВИНА ЦИЛИНДРА: сечение канавки это полукруг радиуса hw,
         вдавленный на глубину depth (depth = hw — ровно половина цилиндра).
         У кромки глубина плавно уходит в ноль, край получается чистым. */
      const u = Math.sqrt(d2) / hw;
      let f = Math.sqrt(Math.max(0, 1 - u * u));
      /* последние 18 % радиуса — мягкий вход: стенка полуцилиндра у кромки
         вертикальная, и на гранях 0.2 мм кромка выглядела рваной */
      if (u > 0.93) { const q = (u - 0.93) / 0.07; f *= 1 - q * q * (3 - 2 * q); }
      const s = -depth * f;
      if (s === 0) continue;
      /* Вдавливаем не по «своей» нормали узла, а по нормали поверхности в этой
         точке пути: всё сечение едет в одну сторону, как настоящий резец —
         иначе соседние узлы идут в разные стороны и появляется «рябь». */
      let ex = nv[o], ey = nv[o + 1], ez = nv[o + 2];
      if (pn && pn[pr.seg] && pn[pr.seg + 1]) {
        const A2 = pn[pr.seg], B2 = pn[pr.seg + 1], t2 = pr.t || 0;
        ex = A2[0] + (B2[0] - A2[0]) * t2;
        ey = A2[1] + (B2[1] - A2[1]) * t2;
        ez = A2[2] + (B2[2] - A2[2]) * t2;
        const le = Math.hypot(ex, ey, ez) || 1;
        ex /= le; ey /= le; ez /= le;
      }
      /* Только «своя» сторона: узел, у которого нормаль смотрит против нормали
         пути, — это обратная стенка панели. Раньше её тоже вдавливало, и на
         тонком кузове она протыкала наружную стенку: на месте канала получался
         кратер с видимой изнанкой (борозда «насквозь»). */
      const nl = Math.hypot(nv[o], nv[o + 1], nv[o + 2]) || 1;
      if ((nv[o] * ex + nv[o + 1] * ey + nv[o + 2] * ez) / nl < 0.15) { skipped++; continue; }
      pos[o] += ex * s;
      pos[o + 1] += ey * s;
      pos[o + 2] += ez * s;
      if (opts.snap) pressed.push({ v, s, ex, ey, ez });
      moved++; if (Math.abs(s) > maxd) maxd = Math.abs(s);
    }
    /* 2b) страховка от складок: если грань вывернулась или сжалась в иголку —
       подтягиваем её узлы назад (на грубой сетке это единственное, что мешает
       «сложить» панель вокруг желоба) */
    if (moved) clampFlips(pos, pre, topo.tris, 0.05, 6, 0.25);
    /* 3) разгладить поясок вокруг желоба: мелкие грани на неровной панели дают
       «рябь» в затенении. Двигаем только узлы у кромки (дна не касаемся) и только
       в касательной плоскости — глубина и форма желоба не меняются. */
    {
      const neigh = new Map();
      for (let i = 0; i < liveIndex.length; i += 3) {
        const a = liveIndex[i], b = liveIndex[i + 1], c = liveIndex[i + 2];
        const add = (x, y) => { const e = neigh.get(x); if (e) e.push(y); else neigh.set(x, [y]); };
        add(a, b); add(a, c); add(b, a); add(b, c); add(c, a); add(c, b);
      }
      const ring = [];
      const lim = 2.5 * sp;                        // сосед не дальше этого — иначе это край
      for (const [v, ns] of neigh) {
        if (!fine[v]) continue;
        const o = v * 3;
        const d = Math.sqrt(distToPath(path, [pos[o], pos[o + 1], pos[o + 2]]));
        if (d < hw * 0.55 || d > hw * 1.5) continue;      // дно и внешняя полоса
        if (ns.length < 3 || ns.length > 12) continue;
        let ok = true;
        for (const u of ns) {
          if (!fine[u] || Math.hypot(pos[u * 3] - pos[o], pos[u * 3 + 1] - pos[o + 1],
                                     pos[u * 3 + 2] - pos[o + 2]) > lim) { ok = false; break; }
        }
        if (!ok) continue;
        ring.push([v, ns]);
      }
      for (let it = 0; it < (opts.ringIters || 2); it++) {
        for (const [v, ns] of ring) {
          const o = v * 3;
          let cx = 0, cy = 0, cz = 0;
          for (const u of ns) { cx += pos[u * 3]; cy += pos[u * 3 + 1]; cz += pos[u * 3 + 2]; }
          const k = ns.length || 1;
          const dx = cx / k - pos[o], dy = cy / k - pos[o + 1], dz = cz / k - pos[o + 2];
          const nn = nv[o] * dx + nv[o + 1] * dy + nv[o + 2] * dz;   // нормальную часть убираем
          /* сдвиг не больше четверти шага — иначе сетку «поведёт» */
          const mvx = dx - nv[o] * nn, mvy = dy - nv[o + 1] * nn, mvz = dz - nv[o + 2] * nn;
          const ml = Math.hypot(mvx, mvy, mvz);
          const sc = ml > 1e-9 ? Math.min(0.5, 0.15 * sp / ml) : 0;
          pos[o] += sc * mvx; pos[o + 1] += sc * mvy; pos[o + 2] += sc * mvz;
        }
      }
    }
    /* 3b) посадка глубины: каждый вдавленный узел ставится ровно на профиль
       (s вдоль нормали пути от своего исходного места). Сетка после сгущения
       неравномерная — без этой посадки дно канала «гуляет» на доли шага и
       выглядит заломами. */
    if (opts.snap && pressed.length) {
      for (const q of pressed) {
        const o = q.v * 3;
        const meas = (pos[o] - pre[o]) * q.ex + (pos[o + 1] - pre[o + 1]) * q.ey +
          (pos[o + 2] - pre[o + 2]) * q.ez;
        const corr = q.s - meas;
        if (corr === 0) continue;
        pos[o] += q.ex * corr; pos[o + 1] += q.ey * corr; pos[o + 2] += q.ez * corr;
      }
    }

    /* 4) страховка: если какая-то грань всё же выродилась в ноль (площадь ~0),
       сдвигаем одну её вершину на 0.002 единицы вдоль нормали. Топология не
       меняется, вода-тайт не ломается, а нулевых граней в отчёте не остаётся. */
    for (let i = 0; i < liveIndex.length; i += 3) {
      const a = liveIndex[i], b2 = liveIndex[i + 1], c2 = liveIndex[i + 2];
      if (!fine[a] && !fine[b2] && !fine[c2]) continue;
      const ax = pos[a * 3], ay = pos[a * 3 + 1], az = pos[a * 3 + 2];
      const bx = pos[b2 * 3], by = pos[b2 * 3 + 1], bz = pos[b2 * 3 + 2];
      const cx = pos[c2 * 3], cy = pos[c2 * 3 + 1], cz = pos[c2 * 3 + 2];
      const nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
      const ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
      const nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (Math.hypot(nx, ny, nz) * 0.5 >= 1e-9) continue;
      pos[a * 3] += nv[a * 3] * 0.002;
      pos[a * 3 + 1] += nv[a * 3 + 1] * 0.002;
      pos[a * 3 + 2] += nv[a * 3 + 2] * 0.002;
    }
    nrm = null;
    flush();
    return {
      added, moved, maxd, skipped, pre, sp, hw, depth,
      secs: (performance.now() - t0) / 1000,
      tRefine: (tRefine - t0) / 1000,
      tPress: (performance.now() - tRefine) / 1000
    };
  }

  /* Сама борозда: путь по коже a→b → новые точки в полосе → вдавливание
     профилем «полуцилиндр вдоль пути». Ширина = ползунок «Ширина», глубина =
     «Глубина». Кузов вне полосы не двигается вовсе. */
  function makeGroove(a, b) {
    const hw = Math.max(brushR() * 0.5, 0.30 * U);          // полуширина желоба
    const t0 = performance.now();
    const path = surfacePath(a, b);
    const tPath = performance.now();
    if (path.length < 3) {
      syncInfo('борозда не сделана: линия не легла на кузов — протяните по поверхности');
      return null;
    }
    const L = pathLen(path);
    if (L < hw * 6) {
      syncInfo(`борозда короткая (${(L / U).toFixed(1)} мм) — протяните линию подлиннее`);
      return null;
    }
    /* Глубина в половину цилиндра: depth = hw даёт ровно половину окружности.
       Если ползунок больше полуширины — берём полуширину, иначе срез «съест» бока. */
    const depth = Math.min(depthD(), hw);
    const st = carvePath(path, hw, depth, { limit: 14000 });
    const secs = (performance.now() - t0) / 1000;
    grooveInfo = {
      a: a.slice(), b: b.slice(), path, len: L, width: hw * 2, depth: st.maxd,
      added: st.added, moved: st.moved, skipped: st.skipped, pre: st.pre,
      vertsAfter: vk, secs,
      tPath: (tPath - t0) / 1000, tRefine: st.tRefine, tPress: st.tPress
    };
    const half = Math.abs(depth - hw) < 1e-6;
    syncInfo(`борозда ${(L / U).toFixed(1)} мм: ширина ${(hw * 2 / U).toFixed(2)} мм, ` +
      `глубина ${(st.maxd / U).toFixed(2)} мм${half ? ' (полуцилиндр)' : ''} · ` +
      `новых точек ${st.added}, вдавлено ${st.moved} · ${secs.toFixed(2)} с`);
    return grooveInfo;
  }

  let grooveInfo = null;
  let grooveA = null, grooveB = null, grooveSnap = null;

  /* ---------- 8. Канат: привязка к граням → отдельный объект → прожиг ----------
     Канат прокладывают по поверхности: каждая точка пути — попадание луча в
     грань кузова, поэтому канат лежит точно по граням и повторяет все изгибы
     (нормаль грани задаёт «верх» сечения — канат не закручивается).

     По отпусканию кнопки канат становится ОТДЕЛЬНЫМ объектом сцены: у него своя
     геометрия-трубка, свой материал, своё имя («Канат 1»). Кузов при этом не
     меняется: канат лежит сверху и его можно убрать, не тронув модель.

     «🔥 Прожечь» прорезает в кузове канал ровно под форму каната: полуэллипс
     шириной 2R + 6 % и глубиной R (R — радиус каната). Канал получается
     полукруглым, как от раскалённого ножа, а канат садится в него вровень —
     верхняя половина остаётся снаружи. Кузов вне полосы не двигается. */
  let ropeWMM = 2.5;                      // радиус каната, мм
  const ROPE_SIDES = 10;                  // граней в сечении трубки
  const ropes = [];                       // готовые канаты (каждый — отдельный объект)
  let ropeSeq = 0;                        // нумерация канатов
  let ropeSel = -1;                       // выбранный канат
  let ropePts = null, ropeNrms = null;    // путь, который ведётся сейчас
  let ropePrev = null;                    // линия-превью под курсором
  let ropeQual = 2;                       // 1 черновая / 2 точная / 3 максимум полигонов
  let ropeMoveMode = false;               // «↔ Двигать»: тянуть канат по кузову
  let ropeDrag = null;                    // { rp, hit0, path0, nrms0, off0, y0, dist }
  const ropeR = () => ropeWMM * U;

  /** нормаль поверхности в точке попадания — «верх» сечения каната */
  function normalAt(p) {
    const v = nearestNode(p);
    if (v >= 0) {
      const nv = getNormals();
      const l = Math.hypot(nv[v * 3], nv[v * 3 + 1], nv[v * 3 + 2]);
      if (l > 1e-9) return [nv[v * 3] / l, nv[v * 3 + 1] / l, nv[v * 3 + 2] / l];
    }
    return [0, 1, 0];
  }

  /* объект каната живёт в той же системе координат, что и кузов: путь снят с
     его граней, поэтому копируем матрицу кузова, а не считаем заново */
  function attachToScene(obj) {
    const parent = mesh.parent || mesh;
    if (parent && parent.add) parent.add(obj);
    if (obj && obj.matrix && mesh.matrix && obj.matrix.copy) {
      obj.matrix.copy(mesh.matrix);
      obj.matrixAutoUpdate = false;
    }
    return obj;
  }
  function detachFromScene(obj) {
    if (!obj) return;
    const parent = mesh.parent || mesh;
    if (parent && parent.remove) parent.remove(obj);
    else if (parent && parent.children) {
      const i = parent.children.indexOf(obj);
      if (i >= 0) parent.children.splice(i, 1);
    }
    if (obj.geometry && obj.geometry.dispose) obj.geometry.dispose();
    if (obj.material && obj.material.dispose) obj.material.dispose();
  }

  /* --- превью: тонкая линия по центру будущего каната --- */
  function ropePreviewLine() {
    if (ropePrev) return ropePrev;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * 6000), 3));
    if (g.setDrawRange) g.setDrawRange(0, 0);
    const m = new THREE.LineBasicMaterial({ color: 0xff8a1e });
    ropePrev = new THREE.Line(g, m);
    ropePrev.frustumCulled = false;
    ropePrev.name = 'канат (превью)';
    attachToScene(ropePrev);
    return ropePrev;
  }
  function ropePreviewDraw() {
    if (!ropePts) return;
    const ln = ropePreviewLine();
    const arr = ln.geometry.attributes.position.array;
    const n = Math.min(ropePts.length, arr.length / 3);
    const off = ropeR();
    for (let i = 0; i < n; i++) {
      arr[i * 3] = ropePts[i][0] + ropeNrms[i][0] * off;
      arr[i * 3 + 1] = ropePts[i][1] + ropeNrms[i][1] * off;
      arr[i * 3 + 2] = ropePts[i][2] + ropeNrms[i][2] * off;
    }
    ln.geometry.attributes.position.needsUpdate = true;
    if (ln.geometry.setDrawRange) ln.geometry.setDrawRange(0, n);
  }

  function ropeStart(clientX, clientY) {
    const p = hitPoint(clientX, clientY);
    if (!p) { syncInfo('нажмите на кузов — канат начинается с грани под курсором'); return false; }
    ropePts = [p];
    ropeNrms = [normalAt(p)];
    ropePreviewDraw();
    syncInfo(`канат: ведите по кузову, отпустите кнопку · радиус ${ropeWMM.toFixed(1)} мм`);
    return true;
  }
  function ropeAdd(clientX, clientY) {
    if (!ropePts) return;
    const p = hitPoint(clientX, clientY);
    if (!p) return;
    const last = ropePts[ropePts.length - 1];
    const step = Math.max(0.45 * ropeR(), 0.30 * U);
    if (Math.hypot(p[0] - last[0], p[1] - last[1], p[2] - last[2]) < step) return;
    if (ropePts.length >= 5900) return;                  // превью-буфер на 6000 точек
    ropePts.push(p);
    ropeNrms.push(normalAt(p));
    ropePreviewDraw();
    syncInfo(`канат ${(pathLen(ropePts) / U).toFixed(1)} мм · точек ${ropePts.length} · ` +
      `радиус ${ropeWMM.toFixed(1)} мм — отпустите кнопку`);
  }
  /** отпускание: путь сглаживается, пересаживается на ровный шаг и становится объектом */
  function ropeFinish() {
    const pts = ropePts, nrs = ropeNrms;
    ropePts = null; ropeNrms = null;
    if (ropePrev) { detachFromScene(ropePrev); ropePrev = null; }
    if (!pts || pts.length < 3) {
      syncInfo('канат не сделан: протяните линию по кузову и отпустите кнопку');
      return null;
    }
    const R = ropeR();
    const step = Math.max(0.45 * R, 0.25 * U);
    const sm = smoothPath(pts, nrs, 2);                  // убираем дрожание луча
    /* и сразу снимаем излом по граням: на крупной сетке штрих идёт по хордам,
       и без этого канат выглядит «гранёным» ещё до «Выравнять».
       Окно 1.6 радиуса — «мягко»: замысел пользователя не съедаем,
       «⌐ Выровнять» потом может снять сильнее (окно 2 радиуса, 5 кругов) */
    let cur0 = { pts: sm.pts, nrms: sm.nrms }, w0 = 1.6 * R;
    for (let round = 0; round < 3; round++) {
      const pr0 = smoothProject(cur0.pts, cur0.nrms, w0);
      cur0 = resamplePath(pr0.pts, pr0.nrms, step);
      w0 *= 0.8;
    }
    const rs = { pts: cur0.pts, nrms: cur0.nrms };
    const L = pathLen(rs.pts);
    if (L < 4 * R) {
      syncInfo(`канат короткий (${(L / U).toFixed(1)} мм при радиусе ${ropeWMM.toFixed(1)} мм) — протяните подлиннее`);
      return null;
    }
    const rp = newRope(rs.pts, rs.nrms, R);
    syncInfo(`канат «${rp.name}»: ${(L / U).toFixed(1)} мм, радиус ${ropeWMM.toFixed(1)} мм, ` +
      `точек ${rs.pts.length} — теперь это отдельный объект · «🔥 Прожечь» прорежет канал под него`);
    return rp;
  }

  function ropeMaterial() {
    const M = THREE.MeshStandardMaterial || THREE.MeshBasicMaterial;
    return new M({ color: 0x2b2c31, roughness: 0.72, metalness: 0.08 });
  }
  function buildRopeMesh(rp) {
    const t = tubeArrays(rp.path, rp.nrms, rp.radius, ROPE_SIDES, rp.off);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(t.positions, 3));
    g.setIndex(new THREE.BufferAttribute(t.index, 1));
    if (g.computeVertexNormals) g.computeVertexNormals();
    if (g.computeBoundingSphere) g.computeBoundingSphere();
    const obj = new THREE.Mesh(g, ropeMaterial());
    obj.name = rp.name;
    obj.castShadow = true;
    obj.userData = { rope: rp.name, radiusMM: rp.radius / U, rings: t.rings, tris: t.index.length / 3 };
    rp.tube = t;
    attachToScene(obj);
    return obj;
  }
  function tintRopes() {
    ropes.forEach((rp, i) => {
      const m = rp.mesh && rp.mesh.material;
      if (!m) return;
      const hex = i === ropeSel ? 0xff7a18 : 0x2b2c31;
      if (m.color && typeof m.color.setHex === 'function') m.color.setHex(hex);
      else m.color = hex;
      if (m.emissive && typeof m.emissive.setHex === 'function') m.emissive.setHex(i === ropeSel ? 0x2a1100 : 0x000000);
    });
  }
  function newRope(pts, nrms, R) {
    const rp = { name: `Канат ${++ropeSeq}`, path: pts, nrms, radius: R, off: R, mesh: null, tube: null, burn: null,
      path0: pts.map(q => q.slice()), nrms0: nrms ? nrms.map(q => q.slice()) : null };
    rp.mesh = buildRopeMesh(rp);
    ropes.push(rp);
    ropeSel = ropes.length - 1;
    tintRopes();
    syncRopeRow();
    return rp;
  }
  function rebuildRope(rp) {
    const old = rp.mesh;
    rp.mesh = buildRopeMesh(rp);
    detachFromScene(old);
    tintRopes();
  }
  function removeRope(i = ropeSel) {
    const rp = ropes[i];
    if (!rp) { syncInfo('канатов нет — сначала проведите канат по кузову'); return 0; }
    detachFromScene(rp.mesh);
    ropes.splice(i, 1);
    ropeSel = ropes.length ? Math.min(i, ropes.length - 1) : -1;
    tintRopes();
    syncRopeRow();
    syncInfo(`канат «${rp.name}» убран из сцены — кузов не тронут`);
    return ropes.length;
  }
  function syncRopeRow() {
    const el = $('sc-ropeinfo');
    if (!el) return;
    if (!ropes.length) { el.textContent = 'канатов нет — проведите канат по кузову'; return; }
    const rp = ropes[ropeSel] || ropes[ropes.length - 1];
    const L = pathLen(rp.path);
    el.textContent = `канатов ${ropes.length} · «${rp.name}»: ${(L / U).toFixed(1)} мм, ` +
      `⌀${(rp.radius * 2 / U).toFixed(1)} мм · излом ${pathTurn(rp.path).toFixed(0)}°` +
      ` (макс ${pathKink(rp.path).toFixed(0)}°)` +
      (rp.burn ? ` · канал ${(rp.burn.depth / U).toFixed(2)} мм` : ' · канал не прожжён');
  }

  /* --- выравнивание и перемещение каната ---
     Канат ложится по граням, и если грани крупные, путь получается изломанным
     (по хордам граней). «Выровнять» сглаживает путь и снова привязывает его к
     кузову, «По прямой» ставит его по хорде между концами, «Двигать» тянет
     канат по поверхности мышью (Shift — приподнять/утопить). */

  /** привязать точки пути обратно к граням: луч вдоль нормали поверхности */
  /* Треугольники в окрестности пути: 110 тыс. граней кузова × сотни лучей —
     это секунды, поэтому сначала отбираем полосу вокруг каната (десятки
     миллисекунд), а лучи пускаем только по ней. */
  function trisNear(pts, pad) {
    let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
    for (const p of pts) {
      if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0];
      if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1];
      if (p[2] < z0) z0 = p[2]; if (p[2] > z1) z1 = p[2];
    }
    x0 -= pad; y0 -= pad; z0 -= pad; x1 += pad; y1 += pad; z1 += pad;
    const pos = mesh.geometry.attributes.position.array, idx = mesh.geometry.index.array;
    const out = [];
    const inBox = (i) => pos[i] >= x0 && pos[i] <= x1 && pos[i + 1] >= y0 && pos[i + 1] <= y1 && pos[i + 2] >= z0 && pos[i + 2] <= z1;
    for (let f = 0; f < idx.length; f += 3) {
      const a = idx[f] * 3, b = idx[f + 1] * 3, c = idx[f + 2] * 3;
      if (!inBox(a) || !inBox(b) || !inBox(c)) continue;
      out.push(pos[a], pos[a + 1], pos[a + 2], pos[b], pos[b + 1], pos[b + 2], pos[c], pos[c + 1], pos[c + 2]);
    }
    return Float64Array.from(out);
  }

  /* ближайшее попадание луча в набор треугольников (Мёллер–Трумберг) */
  function rayTris(T, f0, f1, f2, d0, d1, d2, maxT) {
    let best = -1;
    for (let i = 0; i < T.length; i += 9) {
      const ax = T[i], ay = T[i + 1], az = T[i + 2];
      const e1x = T[i + 3] - ax, e1y = T[i + 4] - ay, e1z = T[i + 5] - az;
      const e2x = T[i + 6] - ax, e2y = T[i + 7] - ay, e2z = T[i + 8] - az;
      const px = d1 * e2z - d2 * e2y, py = d2 * e2x - d0 * e2z, pz = d0 * e2y - d1 * e2x;
      const det = e1x * px + e1y * py + e1z * pz;
      if (det > -1e-12 && det < 1e-12) continue;
      const inv = 1 / det;
      const tx = f0 - ax, ty = f1 - ay, tz = f2 - az;
      const u = (tx * px + ty * py + tz * pz) * inv;
      if (u < 0 || u > 1) continue;
      const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
      const v = (d0 * qx + d1 * qy + d2 * qz) * inv;
      if (v < 0 || u + v > 1) continue;
      const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
      if (t > 1e-6 && t < maxT && (best < 0 || t < best)) best = t;
    }
    return best;
  }

  function projectOnSurface(pts, nrms) {
    const outP = [], outN = [];
    if (!pts.length) return { pts: outP, nrms: outN };
    const R = ropeR();
    const reach = Math.max(6 * R, 3 * U);
    const T = trisNear(pts, reach * 1.5);
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const n = (nrms && nrms[i]) ? nrms[i] : [0, 1, 0];
      const l = Math.hypot(n[0], n[1], n[2]) || 1;
      const nx = n[0] / l, ny = n[1] / l, nz = n[2] / l;
      const from = [p[0] + nx * reach, p[1] + ny * reach, p[2] + nz * reach];
      const t = rayTris(T, from[0], from[1], from[2], -nx, -ny, -nz, 2 * reach);
      if (t < 0) { outP.push(p.slice()); outN.push(n.slice()); continue; }
      outP.push([from[0] - nx * t, from[1] - ny * t, from[2] - nz * t]);
      outN.push(n.slice());
    }
    return { pts: outP, nrms: outN };
  }

  /** выровнять выбранный канат: 'smooth' — сгладить изломы, 'line' — по прямой */
  /* Проекция на грани со сглаживанием поправки: сначала путь сглаживается,
     потом считается, куда его тянет поверхность, и эта поправка тоже
     сглаживается. Так канат не «прилипает» к каждой грани (иначе на крупной
     сетке он идёт по хордам и ломается), но и не уходит от кузова. */
  function smoothProject(pts, nrms, halfW) {
    const sm = smoothWindow(pts, nrms, halfW, 2);
    const pr = projectOnSurface(sm.pts, sm.nrms);
    const disp = pr.pts.map((q, i) => [q[0] - sm.pts[i][0], q[1] - sm.pts[i][1], q[2] - sm.pts[i][2]]);
    const ds = smoothWindow(disp, null, Math.max(halfW * 0.7, U), 2);
    return {
      pts: sm.pts.map((p, i) => [p[0] + ds.pts[i][0], p[1] + ds.pts[i][1], p[2] + ds.pts[i][2]]),
      nrms: pr.nrms
    };
  }

  function ropeAlign(kind = 'smooth') {
    const rp = ropes[ropeSel];
    if (!rp) { syncInfo('каната нет — сначала проведите канат по кузову'); return null; }
    const turn0 = pathTurn(rp.path), kink0 = pathKink(rp.path);
    const step = Math.max(0.45 * rp.radius, 0.25 * U);
    let pts, nrms;
    if (kind === 'line') {
      const st = straightPath(rp.path, rp.nrms, Math.max(8, Math.round(pathLen(rp.path) / step)));
      pts = st.pts; nrms = st.nrms;
    } else {
      /* натянутая нить: круг «сгладить окном → привязать к граням → ровный
         шаг», окно сужается; 5 кругов сходятся к ровной линии на кузове.
         Окно задаётся в длине дуги (2 радиуса каната), а не числом итераций:
         ступеньки граней короче, поэтому и снимаются именно они. */
      let w = Math.max(2 * rp.radius, 3 * step);
      let cur = { pts: rp.path, nrms: rp.nrms };
      for (let round = 0; round < 5; round++) {
        const pr = smoothProject(cur.pts, cur.nrms, w);
        cur = resamplePath(pr.pts, pr.nrms, step);
        w *= 0.72;
      }
      pts = cur.pts; nrms = cur.nrms;
    }
    const pr = projectOnSurface(pts, nrms);                  // и снова на грани кузова
    const rs2 = resamplePath(pr.pts, pr.nrms, step);
    rp.path = projectOnSurface(rs2.pts, rs2.nrms).pts;       // точно на гранях
    rp.nrms = rs2.nrms || rp.nrms;
    rebuildRope(rp);
    syncRopeRow();
    const turn1 = pathTurn(rp.path), kink1 = pathKink(rp.path);
    syncInfo(`${kind === 'line' ? '➖ канат поставлен по прямой' : '⌐ канат выровнен'}: ` +
      `излом ${turn0.toFixed(0)}° → ${turn1.toFixed(0)}°, макс. залом ${kink0.toFixed(0)}° → ${kink1.toFixed(0)}° · ` +
      `длина ${(pathLen(rp.path) / U).toFixed(1)} мм, точек ${rp.path.length}`);
    return { turn0, turn1, kink0, kink1, len: pathLen(rp.path), points: rp.path.length };
  }

  /** приподнять/утопить канат над поверхностью (off в единицах модели) */
  function ropeLift(dOff) {
    const rp = ropes[ropeSel];
    if (!rp) return null;
    rp.off = Math.max(0, Math.min(rp.radius * 3, rp.off + dOff));
    rebuildRope(rp);
    syncRopeRow();
    return rp.off;
  }

  /** вернуть канат в исходный вид (как провели) */
  function ropeReset() {
    const rp = ropes[ropeSel];
    if (!rp || !rp.path0) { syncInfo('вернуть нечего — канат уже как провели'); return null; }
    rp.path = rp.path0.map(q => q.slice());
    rp.nrms = rp.nrms0 ? rp.nrms0.map(q => q.slice()) : rp.nrms;
    rp.off = rp.radius;
    rebuildRope(rp);
    syncRopeRow();
    syncInfo(`канат «${rp.name}» возвращён к исходной линии (${(pathLen(rp.path) / U).toFixed(1)} мм)`);
    return rp.path;
  }

  /* --- перемещение: тянем канат по поверхности --- */
  function ropeMoveStart(clientX, clientY) {
    const rp = ropes[ropeSel];
    if (!rp) { syncInfo('каната нет — проведите его по кузову'); return false; }
    const p = hitPoint(clientX, clientY);
    if (!p) { syncInfo('нажмите на кузов рядом с канатом — и тяните'); return false; }
    const w = mesh.localToWorld(new THREE.Vector3(p[0], p[1], p[2]));
    const cp = camera.position;
    const dist = Math.hypot(cp.x - w.x, cp.y - w.y, cp.z - w.z);   // без Vector3.distanceTo (стаб)
    ropeDrag = {
      rp, hit0: p, path0: rp.path.map(q => q.slice()), nrms0: rp.nrms.map(q => q.slice()),
      off0: rp.off, y0: clientY, dist: Math.max(1, dist)
    };
    syncInfo(`канат «${rp.name}»: тяните по кузову (Shift — приподнять/утопить)`);
    return true;
  }
  function ropeMoveAt(clientX, clientY, shift) {
    const d = ropeDrag;
    if (!d) return;
    if (shift) {
      /* Shift — по нормали: перевод пикселей в единицы модели по расстоянию до камеры */
      const fovY = (camera.fov || 36) * Math.PI / 180;
      const k = (2 * Math.tan(fovY / 2) / Math.max(1, innerHeight)) * d.dist;
      d.rp.off = Math.max(0, Math.min(d.rp.radius * 3, d.off0 - (clientY - d.y0) * k));
      rebuildRope(d.rp);
      syncInfo(`канат «${d.rp.name}»: подъём ${(d.rp.off / U).toFixed(2)} мм над поверхностью`);
      return;
    }
    const p = hitPoint(clientX, clientY);
    if (!p) return;
    const dx = p[0] - d.hit0[0], dy = p[1] - d.hit0[1], dz = p[2] - d.hit0[2];
    const moved = d.path0.map((q, i) => [q[0] + dx, q[1] + dy, q[2] + dz]);
    const pr = projectOnSurface(moved, d.nrms0);            // канат остаётся на гранях
    d.rp.path = pr.pts;
    d.rp.nrms = pr.nrms;
    rebuildRope(d.rp);
    syncInfo(`канат «${d.rp.name}»: сдвинут на ${(Math.hypot(dx, dy, dz) / U).toFixed(1)} мм ` +
      `(длина ${(pathLen(d.rp.path) / U).toFixed(1)} мм)`);
  }
  function ropeMoveEnd() {
    const d = ropeDrag;
    ropeDrag = null;
    if (!d) return null;
    syncRopeRow();
    syncInfo(`канат «${d.rp.name}» на новом месте: ${(pathLen(d.rp.path) / U).toFixed(1)} мм, ` +
      `точек ${d.rp.path.length} · «🔥 Прожечь» режет канал по новой линии`);
    return d.rp;
  }

  /** 🔥 Раскалённый нож: канал в кузове ровно под форму каната, канат садится в него */
  /* Сглаживание НОРМАЛЕЙ в полосе канала: геометрию не трогает, убирает
     «мятую бумагу» в затенении на сгущённой сетке. */
  function smoothNormalsNear(path, rad, iters = 6) {
    const nAttr = geom.attributes.normal;
    const idx = geom.index && geom.index.array;
    if (!nAttr || !idx || !topo || !topo.rep) return 0;
    const n = nAttr.array, rep = topo.rep;
    const map = new Map();
    for (let i = 0; i < idx.length; i += 3) {
      const f0 = rep[idx[i]], f1 = rep[idx[i + 1]], f2 = rep[idx[i + 2]];
      const add = (a, b) => { const l = map.get(a); if (l) l.push(b); else map.set(a, [b]); };
      add(f0, f1); add(f1, f2); add(f2, f0); add(f1, f0); add(f2, f1); add(f0, f2);
    }
    const r2 = rad * rad;
    const touch = [];
    for (let v = 0; v < vk; v++) {
      const o = v * 3;
      if (distToPath(path, [pos[o], pos[o + 1], pos[o + 2]]) <= r2) touch.push(rep[v]);
    }
    if (!touch.length) return 0;
    for (let it = 0; it < iters; it++) {
      const upd = [];
      for (const f of touch) {
        const ns = map.get(f);
        if (!ns || !ns.length) continue;
        let x = n[f * 3] * 2, y = n[f * 3 + 1] * 2, z = n[f * 3 + 2] * 2, c = 2;
        for (const g of ns) { x += n[g * 3]; y += n[g * 3 + 1]; z += n[g * 3 + 2]; c++; }
        const l = Math.hypot(x, y, z) || 1;
        upd.push([f, x / l, y / l, z / l]);
      }
      for (const [f, x, y, z] of upd) { n[f * 3] = x; n[f * 3 + 1] = y; n[f * 3 + 2] = z; }
    }
    nAttr.needsUpdate = true;
    return touch.length;
  }

  function burnRope(i = ropeSel) {
    const rp = ropes[i];
    if (!rp) { syncInfo('прожигать нечего: сначала проведите канат по кузову'); return null; }
    const t0 = performance.now();
    const R = rp.radius;
    const hw = R * 1.06;                                  // желоб на 6 % шире каната
    /* глубина: по умолчанию ровно радиус (канат ложится вровень), ползунок
       «Глубина» может сделать канал глубже — до двух радиусов */
    const depth = Math.min(2 * R, Math.max(R, depthD()));
    const path = rp.path.slice();
    path.nrms = rp.nrms;                                  // всё сечение едет в одну сторону
    pushHist();                                           // отмена вернёт кузов, канат остаётся
    /* «Плотность»: 1 — как борозда, 2 — точная, 3 — максимум полигонов.
       Шаг сгущения = hw/spd, поэтому канал получается гладким: чем мельче
       сетка в полосе, тем ровнее стенка и кромка. snap — после вдавливания
       каждый узел канала ставится точно на аналитический профиль
       (полуэллипс), из-за чего пропадают «заломы» от неровного шага сетки. */
    /* extraPass оставлен выключенным: повторные проходы bandPass на тонкой
       панели (коробка 3 мм, канат 0.6 мм) плодили вырожденные грани и дыры —
       проверено стендом dbg_box. Плотность даёт шаг сгущения spd. */
    const Q = { 1: { spd: 3.5, limit: 20000, snap: false, extra: 0 },
                2: { spd: 5, limit: 80000, snap: true, extra: 0 },
                3: { spd: 6.5, limit: 160000, snap: true, extra: 0 } }[ropeQual] ||
              { spd: 5, limit: 80000, snap: true, extra: 0 };
    const st = carvePath(path, hw, depth, {
      limit: Q.limit, spd: Q.spd, ringIters: 3, snap: Q.snap, extraPass: Q.extra, red: true
    });
    rp.off = 0;                                           // канат садится в канал
    /* Сгущение полосы даёт много узких граней; их нормали «гуляют», и в
       затенении канал выглядел мятой бумагой (геометрия при этом ровная —
       проверено лучами по профилю). Сглаживаем нормали только у полосы. */
    smoothNormalsNear(path, 4 * hw, 6);
    rebuildRope(rp);
    const L = pathLen(rp.path);
    const secs = (performance.now() - t0) / 1000;
    const prev = rp.burn;
    rp.burn = {
      width: hw * 2, depth: st.maxd, len: L, moved: st.moved, added: st.added, secs,
      vertsAfter: vk, n: (prev ? prev.n : 0) + 1
    };
    syncRopeRow();
    syncInfo(`🔥 канал по канату «${rp.name}»: ширина ${(hw * 2 / U).toFixed(2)} мм, ` +
      `глубина ${(st.maxd / U).toFixed(2)} мм, длина ${(L / U).toFixed(1)} мм` +
      (rp.burn.n > 1 ? ` (прожиг №${rp.burn.n} — канал глубже)` : '') +
      ` · новых точек ${st.added}, вдавлено ${st.moved} · ${secs.toFixed(2)} с`);
    return rp.burn;
  }

  /** сохранить выбранный канат отдельным файлом — он и правда отдельный объект */
  function saveRope(kind = '3mf') {
    const rp = ropes[ropeSel];
    if (!rp) { syncInfo('канатов нет — сохранять нечего'); return null; }
    const t = rp.tube || tubeArrays(rp.path, rp.nrms, rp.radius, ROPE_SIDES, rp.off);
    let bytes, name;
    if (kind === 'stl') {
      bytes = meshToSTL(t.positions, t.index, 'канат');
      name = `${rp.name}.stl`;
    } else {
      bytes = meshTo3mf(t.positions, t.index, {
        name: rp.name,
        meta: `Канат: ${(pathLen(rp.path) / U).toFixed(1)} мм, радиус ${(rp.radius / U).toFixed(2)} мм`
      });
      name = `${rp.name}.3mf`;
    }
    const res = downloadBytes(bytes, name, kind === 'stl' ? 'model/stl' : undefined);
    if (info) {
      info.innerHTML = `канат «${rp.name}» → <a href="${res.url}" download="${name}" ` +
        `style="color:var(--acc)">скачать ${name}</a> · ${(bytes.length / 1024).toFixed(0)} КБ`;
    }
    return { name, size: bytes.length, url: res.url };
  }

  /* ---------- 7. рамка: прямоугольник по осям экрана → зеркальная копия ----------
     Рамка рисуется на экране, поэтому её стороны всегда параллельны осям.
     Внутрь попадают только видимые узлы (маска лицевых граней) и только та
     сторона кузова, которой в рамке больше: иначе зеркалить нечего.
     «⇄ Зеркально» переносит выделение на другую сторону: каждая вершина там
     встаёт ровно в зеркальную точку выделения (y → −y), если зеркальная
     поверхность ближе «Радиуса». Топология не меняется — сетка остаётся
     замкнутой. */
  let marqueeEl = null;
  const marquee = () => (marqueeEl || (marqueeEl = $('sc-marquee')));
  const projV = new THREE.Vector3();
  function showMarquee(x0, y0, x1, y1) {
    const m = marquee();
    if (!m) return;
    m.style.display = 'block';
    m.style.left = Math.min(x0, x1) + 'px';
    m.style.top = Math.min(y0, y1) + 'px';
    m.style.width = Math.abs(x1 - x0) + 'px';
    m.style.height = Math.abs(y1 - y0) + 'px';
  }
  function hideMarquee() { const m = marquee(); if (m) m.style.display = 'none'; }
  /** точка модели → экран (пиксели), out = [x, y] */
  function screenOfPoint(x, y, z, out) {
    projV.x = x; projV.y = y; projV.z = z;
    if (mesh.localToWorld) mesh.localToWorld(projV);
    projV.project(camera);
    out[0] = (projV.x * 0.5 + 0.5) * innerWidth;
    out[1] = (-projV.y * 0.5 + 0.5) * innerHeight;
    return out;
  }
  function screenOf(v, out) {
    const o = v * 3;
    return screenOfPoint(pos[o], pos[o + 1], pos[o + 2], out);
  }
  function inRectScreen(x, y, z) {
    if (!rectSel) return true;
    const sc = screenOfPoint(x, y, z, [0, 0]);
    return sc[0] >= rectSel.x0 - 1 && sc[0] <= rectSel.x1 + 1 &&
      sc[1] >= rectSel.y0 - 1 && sc[1] <= rectSel.y1 + 1;
  }
  function rectSelect(x0, y0, x1, y1) {
    const xa = Math.min(x0, x1), xb = Math.max(x0, x1);
    const ya = Math.min(y0, y1), yb = Math.max(y0, y1);
    if (xb - xa < 5 || yb - ya < 5) { syncInfo('рамка слишком маленькая — растяните её по кузову'); return 0; }
    const c = hit((xa + xb) / 2, (ya + yb) / 2) || [0, 0, 0];
    computeFront(c);
    const sc = [0, 0];
    const picked = [];
    let sumY = 0;
    for (let v = 0; v < vk; v++) {
      if (dead[v] || !front[v]) continue;
      screenOf(v, sc);
      if (sc[0] < xa || sc[0] > xb || sc[1] < ya || sc[1] > yb) continue;
      picked.push(v); sumY += pos[v * 3 + 1];
    }
    if (!picked.length) { syncInfo('в рамке нет видимых узлов — растяните её по кузову'); return 0; }
    const meanY = sumY / picked.length;
    let half = 0;
    for (const v of picked) half = Math.max(half, Math.abs(pos[v * 3 + 1]));
    if (Math.abs(meanY) < half * 0.12) {
      reg.fill(0); regionN = 0; regionFront = null; paintMarks();
      syncInfo('рамка захватила обе стороны кузова — выделите часть одной стороны');
      return 0;
    }
    const side = meanY > 0 ? 1 : -1;
    reg.fill(0); regionN = 0; regionFront = null;
    let n = 0;
    for (const v of picked) {
      if ((pos[v * 3 + 1] > 0 ? 1 : -1) !== side) continue;
      reg[v] = 1; n++;
    }
    regionN = n;
    rectSel = { x0: xa, y0: ya, x1: xb, y1: yb };
    regFromRect = true;
    paintMarks();
    syncInfo(`рамка: ${n} вершин, ${side > 0 ? 'левая' : 'правая'} сторона · ` +
      `«⇄ Зеркально» скопирует на ${side > 0 ? 'правую' : 'левую'}`);
    return n;
  }

  /** Убрать грани с нулевой площадью и с двойным направленным ребром, дырку
      зашить: после вставки копии на кромке остаётся десяток таких граней. */
  function repairSeam(rounds = 6) {
    let fixed = 0;
    for (let r = 0; r < rounds; r++) {
      const dir = new Map();
      const dupe = new Set();
      for (let i = 0; i < liveIndex.length; i += 3) {
        const a = canon[liveIndex[i]], b = canon[liveIndex[i + 1]], c = canon[liveIndex[i + 2]];
        for (const [p, q] of [[a, b], [b, c], [c, a]]) {
          const k = p * 1000000 + q;
          const was = dir.get(k);
          if (was) dupe.add(k);
          dir.set(k, 1);
        }
      }
      const bad = new Uint8Array(vk);
      let nbad = 0;
      for (let i = 0; i < liveIndex.length; i += 3) {
        const a = canon[liveIndex[i]], b = canon[liveIndex[i + 1]], c = canon[liveIndex[i + 2]];
        const ao = a * 3, bo = b * 3, co = c * 3;
        const e1x = pos[bo] - pos[ao], e1y = pos[bo + 1] - pos[ao + 1], e1z = pos[bo + 2] - pos[ao + 2];
        const e2x = pos[co] - pos[ao], e2y = pos[co + 1] - pos[ao + 1], e2z = pos[co + 2] - pos[ao + 2];
        let badTri = Math.hypot(e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x) * 0.5 < 1e-9;
        if (!badTri) {
          for (const [p, q] of [[a, b], [b, c], [c, a]]) {
            if (dupe.has(p * 1000000 + q)) { badTri = true; break; }
          }
        }
        if (!badTri) continue;
        for (const v of [a, b, c]) if (!bad[v]) { bad[v] = 1; nbad++; }
      }
      if (!nbad) break;
      const res = deleteAndFill(topo, liveIndex, pos, bad);
      setIndex(res.index);
      fixed += nbad;
    }
    return fixed;
  }

  /* ---------- зеркальная копия: вырезать другую сторону и вставить копию ----------
     Это не подтягивание соседних вершин к зеркалу, а замена: часть другой стороны
     внутри зеркальной рамки удаляется, на её место встаёт полная копия выделенной
     части (свои вершины и треугольники), а кромка копии привязывается к ближайшим
     оставшимся вершинам — сетка остаётся замкнутой. */
  function mirrorCopy() {
    if (!regionN) { syncInfo('сначала выделите рамкой часть одной стороны кузова'); return 0; }
    let sumY = 0, n = 0;
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let v = 0; v < vk; v++) {
      if (!reg[v] || dead[v]) continue;
      const o = v * 3;
      sumY += pos[o + 1]; n++;
      if (pos[o] < minX) minX = pos[o];
      if (pos[o] > maxX) maxX = pos[o];
      if (pos[o + 2] < minZ) minZ = pos[o + 2];
      if (pos[o + 2] > maxZ) maxZ = pos[o + 2];
    }
    if (!n) { syncInfo('в рамке нет живых вершин'); return 0; }
    const side = sumY / n > 0 ? 1 : -1;

    /* 1. что копируем — все грани этой стороны, попавшие в рамку (видимые и
          скрытые: копия должна закрыть рамку целиком, а не только то, что
          видно с одной точки). Ближе 1 ед. к плоскости симметрии не берём —
          там зеркало накладывается само на себя. */
    const patch = [];
    const inPatch = new Uint8Array(count);
    const inArea = (nd) => {
      const o = nd * 3;
      if (((pos[o + 1] > 0) ? 1 : -1) !== side) return false;
      if (Math.abs(pos[o + 1]) < 1) return false;          // сама плоскость симметрии
      if (!regFromRect) return !!reg[nd];
      return inRectScreen(pos[o], pos[o + 1], pos[o + 2]);
    };
    for (let f = 0; f < liveIndex.length; f += 3) {
      const a = liveIndex[f], b = liveIndex[f + 1], c = liveIndex[f + 2];
      const na = canon[a], nb = canon[b], nc = canon[c];
      if (dead[na] || dead[nb] || dead[nc]) continue;
      if (!inArea(na) || !inArea(nb) || !inArea(nc)) continue;
      patch.push(a, b, c);
      inPatch[a] = 1; inPatch[b] = 1; inPatch[c] = 1;
    }
    if (!patch.length) { syncInfo('в выделении нет целых граней — растяните рамку'); return 0; }

    /* 2. что вырезаем — ровно «след» копии на экране: силуэт выделенного участка
          проецируется в сетку пикселей, и вершина другой стороны вырезается,
          если её зеркало попадает в этот силуэт. Тогда кромка дыры совпадает с
          кромкой копии, и щель на стыке выходит узкой. */
    const pad = Math.max(brushR(), 12);
    const CELL = 2;                       // размер клетки силуэта, px
    let sMinX = Infinity, sMaxX = -Infinity, sMinY = Infinity, sMaxY = -Infinity;
    const sp = [0, 0];
    const patchScreen = new Map();        // «клетка» → 1
    const cellOf = (x, y) => Math.floor(x / CELL) + ':' + Math.floor(y / CELL);
    for (let i = 0; i < patch.length; i += 3) {
      const A = patch[i] * 3, B = patch[i + 1] * 3, C = patch[i + 2] * 3;
      const pa = screenOfPoint(src[A], -src[A + 1], src[A + 2], [0, 0]);
      const ax = pa[0], ay = pa[1];
      const pb = screenOfPoint(src[B], -src[B + 1], src[B + 2], [0, 0]);
      const bx = pb[0], by = pb[1];
      const pc = screenOfPoint(src[C], -src[C + 1], src[C + 2], [0, 0]);
      const cx = pc[0], cy = pc[1];
      if (!isFinite(ax + ay + bx + by + cx + cy)) continue;
      if (ax < sMinX) sMinX = ax; if (ax > sMaxX) sMaxX = ax;
      if (bx < sMinX) sMinX = bx; if (bx > sMaxX) sMaxX = bx;
      if (cx < sMinX) sMinX = cx; if (cx > sMaxX) sMaxX = cx;
      if (ay < sMinY) sMinY = ay; if (ay > sMaxY) sMaxY = ay;
      if (by < sMinY) sMinY = by; if (by > sMaxY) sMaxY = by;
      if (cy < sMinY) sMinY = cy; if (cy > sMaxY) sMaxY = cy;
      const area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
      if (Math.abs(area) < 1e-9) { patchScreen.set(cellOf(ax, ay), 1); continue; }
      const inv = 1 / area;
      const gx0 = Math.floor(Math.min(ax, bx, cx) / CELL) - 1, gx1 = Math.ceil(Math.max(ax, bx, cx) / CELL) + 1;
      const gy0 = Math.floor(Math.min(ay, by, cy) / CELL) - 1, gy1 = Math.ceil(Math.max(ay, by, cy) / CELL) + 1;
      for (let gy = gy0; gy <= gy1; gy++) {
        for (let gx = gx0; gx <= gx1; gx++) {
          const px = gx * CELL + CELL / 2, py = gy * CELL + CELL / 2;
          const w0 = ((bx - px) * (cy - py) - (cx - px) * (by - py)) * inv;
          const w1 = ((cx - px) * (ay - py) - (ax - px) * (cy - py)) * inv;
          const w2 = 1 - w0 - w1;
          if (w0 < -0.02 || w1 < -0.02 || w2 < -0.02) continue;
          patchScreen.set(gx + ':' + gy, 1);
        }
      }
    }
    const del = new Uint8Array(vk);
    let delN = 0;
    for (let v = 0; v < vk; v++) {
      if (dead[v]) continue;
      const o = v * 3;
      if (((pos[o + 1] > 0) ? 1 : -1) === side) continue;
      if (pos[o] < minX - pad || pos[o] > maxX + pad) continue;
      if (pos[o + 2] < minZ - pad || pos[o + 2] > maxZ + pad) continue;
      screenOfPoint(pos[o], pos[o + 1], pos[o + 2], sp);
      if (!patchScreen.has(cellOf(sp[0], sp[1]))) continue;
      /* нормали не фильтруем: под копией должно исчезнуть ВСЁ, иначе копия
         ложится поверх старой поверхности и на стыке остаётся сетка заплаток */
      del[v] = 1; delN++;
    }
    if (!delN) { syncInfo('на другой стороне в этой рамке нечего заменять — сдвиньте рамку'); return 0; }

    /* 3. новые вершины — зеркало вершин копируемого участка */
    const newOf = new Int32Array(count).fill(-1);
    const addPos = [];
    let nv = count;
    for (let i = 0; i < count; i++) {
      if (!inPatch[i]) continue;
      newOf[i] = nv++;
      addPos.push(src[i * 3], -src[i * 3 + 1], src[i * 3 + 2]);
    }

    /* 4. индекс: что осталось + грани копии (обход в обратную сторону — зеркало
          переворачивает ориентацию) */
    const keep = [];
    let removedFaces = 0;
    const edgeUse = new Map();        // рёбра оставшейся части: сколько раз и куда
    for (let f = 0; f < liveIndex.length; f += 3) {
      const a = liveIndex[f], b = liveIndex[f + 1], c = liveIndex[f + 2];
      if (del[canon[a]] || del[canon[b]] || del[canon[c]]) { removedFaces++; continue; }
      keep.push(a, b, c);
    }
    if (!keep.length) { syncInfo('после вырезания ничего не осталось — уменьшите рамку'); return 0; }
    const vk2 = count + 8;
    for (let i = 0; i < keep.length; i += 3) {
      const a = keep[i], b = keep[i + 1], c = keep[i + 2];
      for (const [p, q] of [[a, b], [b, c], [c, a]]) {
        const k = p * vk2 + q, kr = q * vk2 + p;
        if (edgeUse.has(kr)) edgeUse.delete(kr); else edgeUse.set(k, 1);
      }
    }
    /* кромка дыры: рёбра, оставшиеся без пары */
    const hole = [];
    const holeSeen = new Set();
    for (const k of edgeUse.keys()) {
      const p = Math.floor(k / vk2), q = k - p * vk2;
      for (const v of [p, q]) if (!holeSeen.has(v)) { holeSeen.add(v); hole.push(v); }
    }

    /* граничные вершины копии: рёбра, встречающиеся в ней один раз */
    const pEdge = new Map();
    for (let i = 0; i < patch.length; i += 3) {
      const a = patch[i], b = patch[i + 1], c = patch[i + 2];
      for (const [p, q] of [[a, b], [b, c], [c, a]]) {
        const k = p * vk2 + q, kr = q * vk2 + p;
        if (pEdge.has(kr)) pEdge.delete(kr); else pEdge.set(k, 1);
      }
    }
    const pBound = [];
    const pSeen = new Set();
    for (const k of pEdge.keys()) {
      const p = Math.floor(k / vk2), q = k - p * vk2;
      for (const v of [p, q]) if (!pSeen.has(v)) { pSeen.add(v); pBound.push(v); }
    }

    /* 5. привязка кромки копии к ближайшим вершинам кромки дыры (1:1) */
    const cell = Math.max(pad, 12);
    const grid = new Map();
    const key = (x, y, z) => (Math.floor(x / cell) + 500) * 1000000 +
      (Math.floor(y / cell) + 500) * 1000 + Math.floor(z / cell) + 500;
    for (const v of hole) {
      const o = canon[v] * 3;
      const k = key(pos[o], pos[o + 1], pos[o + 2]);
      const a = grid.get(k); if (a) a.push(v); else grid.set(k, [v]);
    }
    /* медиана длины ребра — насколько близко должна лежать пара, чтобы её
       можно было сварить: далёкую сварку не делаем, иначе копия растягивается */
    const es = [];
    for (let f = 0; f < liveIndex.length; f += 3) {
      const a = canon[liveIndex[f]] * 3, b = canon[liveIndex[f + 1]] * 3;
      es.push(Math.hypot(pos[a] - pos[b], pos[a + 1] - pos[b + 1], pos[a + 2] - pos[b + 2]));
    }
    es.sort((a, b) => a - b);
    const edgeMed = es.length ? es[es.length >> 1] : 2;
    const weldMax = Math.max(edgeMed * 1.5, 1);
    const pairs = [];
    for (const v of pBound) {
      const i = newOf[v];
      if (i < 0) continue;
      const o = (i - count) * 3;
      const px = addPos[o], py = addPos[o + 1], pz = addPos[o + 2];
      const ix = Math.floor(px / cell), iy = Math.floor(py / cell), iz = Math.floor(pz / cell);
      let bd = Infinity, bv = -1;
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
        const a = grid.get((ix + dx + 500) * 1000000 + (iy + dy + 500) * 1000 + (iz + dz + 500));
        if (!a) continue;
        for (const h of a) {
          const q = canon[h] * 3;
          const d = (pos[q] - px) ** 2 + (pos[q + 1] - py) ** 2 + (pos[q + 2] - pz) ** 2;
          if (d < bd) { bd = d; bv = h; }
        }
      }
      if (bv >= 0 && bd <= weldMax * weldMax) pairs.push([bd, i, bv]);
    }
    pairs.sort((a, b) => a[0] - b[0]);
    const weldOf = new Map();       // новый индекс → вершина оставшейся части
    const usedHole = new Set();
    for (const [d, i, h] of pairs) {
      if (usedHole.has(h) || weldOf.has(i)) continue;
      weldOf.set(i, h); usedHole.add(h);
    }

    /* 6. грани копии */
    for (let i = 0; i < patch.length; i += 3) {
      const a = newOf[patch[i]], b = newOf[patch[i + 1]], c = newOf[patch[i + 2]];
      if (a < 0 || b < 0 || c < 0) continue;
      const A = weldOf.has(a) ? weldOf.get(a) : a;
      const B = weldOf.has(c) ? weldOf.get(c) : c;
      const C = weldOf.has(b) ? weldOf.get(b) : b;
      if (A === B || B === C || C === A) continue;
      keep.push(A, B, C);
    }

    /* 7. применить: координаты, индекс, топология заново */
    pushHist();
    const nextPos = new Float32Array((count + addPos.length / 3) * 3);
    nextPos.set(src);
    nextPos.set(addPos, count * 3);
    count += addPos.length / 3;
    applySrc(nextPos);
    col = new Float32Array(count * 3).fill(1);
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    setIndex(Int32Array.from(keep));
    topo = buildTopology(src, liveIndex);
    vk = topo.vk; canon = topo.canon; pos0 = topo.pos0;
    topo.vk = vk; topo.count = count; topo.pos0 = pos0; topoDirty = false;
    resizeNodeArrays(Math.max(vk, reg.length, dead.length, added.length));
    pos = Float32Array.from(pos0);
    reg.fill(0); dead.fill(0); added.fill(0); pending.fill(0); front.fill(1);
    regionN = 0; delTotal = 0; regionFront = null; block = null; allow = null; nrm = null;

    /* 8. остатки кромки (где привязка не нашлась) зашиваем треугольниками */
    const res = deleteAndFill(topo, liveIndex, pos, new Uint8Array(vk));
    setIndex(res.index);
    const fixedV = repairSeam();

    free = freeMask(topo, reg);
    clearDots(); paintMarks(); updateAddedDots();
    flush();
    const m = checkMesh(topo, liveIndex, pos, U);
    const bad = m.openEdges + m.flipEdges + m.nonManifold + m.degenerate;
    const sideName = side > 0 ? ['левая', 'правая'] : ['правая', 'левая'];
    if (info) {
      info.innerHTML = `скопировано с ${sideName[0]} на ${sideName[1]}: <b>${removedFaces}</b> граней ` +
        `(${delN} вершин) заменено копией из <b>${patch.length / 3}</b> граней, ` +
        `новых вершин ${addPos.length / 3}, привязано к ближайшим ${weldOf.size}` +
        (res.addedFaces ? `, зашито кромкой ${res.addedFaces}` : '') +
        (fixedV ? `, подчищено ${fixedV}` : '') + ' · сетка ' +
        (bad ? `<span style="color:var(--warn)">⚠ дыр ${m.openEdges}, перевёрнутых ${m.flipEdges}, ` +
          `вырожденных ${m.degenerate}</span>` : 'цела (0 дыр, 0 перевёрнутых, 0 вырожденных)');
    }
    return patch.length / 3;
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
    /* кромка, которая была ДО ластика: её нельзя зашивать (край открытой части) */
    ensureTopo();
    const preBoundary = boundaryEdgeKeys(topo, liveIndex);
    const res = deleteAndFill(topo, liveIndex, pos, dead, null, preBoundary);
    setIndex(res.index);
    clearDots();
    let regNow = 0, delNow = 0;
    for (let v = 0; v < vk; v++) { if (reg[v]) regNow++; if (dead[v]) delNow++; }
    regionN = regNow; delTotal = delNow; regionFront = null;
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

  /* ---------- снять выделение ---------- */
  function deselect() {
    if (!regionN) { syncInfo('выделения нет — закрасьте участок кистью'); return 0; }
    const n = regionN;
    pushHist();                       // выделение можно вернуть через Ctrl+Z
    reg.fill(0); regionN = 0; regionFront = null; rectSel = null; regFromRect = false;
    paintMarks();
    syncInfo(`выделение снято: ${n} вершин`);
    return n;
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

  /** Центр и радиус закрашенной области. */
  function regionBall() {
    let cx = 0, cy = 0, cz = 0, n = 0;
    for (let v = 0; v < vk; v++) if (reg[v]) { cx += pos[v * 3]; cy += pos[v * 3 + 1]; cz += pos[v * 3 + 2]; n++; }
    if (!n) return null;
    cx /= n; cy /= n; cz /= n;
    let R = 0;
    for (let v = 0; v < vk; v++) {
      if (!reg[v]) continue;
      const d = Math.hypot(pos[v * 3] - cx, pos[v * 3 + 1] - cy, pos[v * 3 + 2] - cz);
      if (d > R) R = d;
    }
    return { c: [cx, cy, cz], R, n };
  }
  /** Сгустить закрашенную область перед действием: иначе «Выдавить» на грубой
      сетке поднимает 3 узла — получается шип, а не плоскость/купол. */
  function refineRegion(extra = 0) {
    const rb = regionBall();
    if (!rb) return 0;
    const r = rb.R + extra;
    const target = Math.max(r / 4, detailFor(r) * 0.6, 0.1 * U);
    const inRegion = (ia, ib) => reg[canon[ia]] === 1 && reg[canon[ib]] === 1;
    const n = refineSphere(rb.c, r * 1.15, target, 4000, inRegion);
    if (n) { ensureTopo(); regionFront = null; nrm = null; }
    return n;
  }
  function smoothOnce() {
    if (!needRegion()) return;
    const steps = Math.max(4, Math.round(amount / 100 * MAXSTEPS));
    pushHist();
    refineRegion();
    const before = sym ? Float32Array.from(pos) : null;
    const before0 = Float32Array.from(pos);
    free = freeMask(topo, reg);
    const rg = frontMaskedRegion();
    const w = boundaryFalloff(topo, rg, 4);
    const p = smoothDiffuse(topo, freeMask(topo, rg), steps, LAMBDA, pos);
    for (let v = 0; v < vk; v++) {
      if (w[v] >= 1) continue;
      const o = v * 3;
      for (let c = 0; c < 3; c++) p[o + c] = pos[o + c] + (p[o + c] - pos[o + c]) * w[v];
    }
    clampFlips(p, before0, topo.tris);
    pos.set(p);
    const mirrored = symAfter(before);
    nrm = null;
    flush();
    let dmax = 0, moved = 0;
    for (let v = 0; v < vk; v++) {
      if (!rg[v]) continue;
      const o = v * 3;
      const d = Math.hypot(pos[o] - before0[o], pos[o + 1] - before0[o + 1], pos[o + 2] - before0[o + 2]);
      if (d > 1e-9) moved++;
      if (d > dmax) dmax = d;
    }
    if (info) {
      info.textContent = `сглажено: ${moved} вершин из ${regionN}, макс. сдвиг ${(dmax / U).toFixed(3)} мм` +
        (mirrored ? ` · зеркально повторено на ${mirrored}` : '');
    }
  }

  function flattenOnce() {
    if (!needRegion()) return;
    pushHist();
    refineRegion();
    const before = sym ? Float32Array.from(pos) : null;
    const soft = Math.max(1, Math.min(3, Math.round(Math.sqrt(regionN) / 3)));
    const preF = Float32Array.from(pos);
    const res = flattenRegion(topo, pos, frontMaskedRegion(), liveIndex, 1, soft);
    clampFlips(pos, preF, topo.tris, 0.05, 6, 0.15);
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
    refineRegion();
    const before = sym ? Float32Array.from(pos) : null;
    const soft = Math.max(1, Math.min(3, Math.round(Math.sqrt(regionN) / 3)));
    const pre = Float32Array.from(pos);
    const res = sharpenRegion(topo, pos, frontMaskedRegion(), liveIndex, 0.5, 3, soft);
    clampFlips(pos, pre, topo.tris);
    symAfter(before);
    nrm = null;
    flush();
    if (info) info.textContent = `обострено: ${res.moved} вершин (макс. ${(res.maxd / U).toFixed(3)} мм)`;
  }

  /** ⇢ Расслабить: узлы ровнее, форма та же. */
  function relaxOnce() {
    if (!needRegion()) return;
    pushHist();
    const before = sym ? Float32Array.from(pos) : null;
    const pre = Float32Array.from(pos);
    const steps = Math.max(2, Math.round(amount / 100 * 12));
    const res = relaxRegion(topo, pos, frontMaskedRegion(), steps, 0.5);
    clampFlips(pos, pre, topo.tris, 0.05, 6, 0.15);
    symAfter(before);
    nrm = null;
    flush();
    if (info) info.textContent = `расслаблено: ${res.moved} вершин, макс. сдвиг ${(res.maxd / U).toFixed(3)} мм (форма не меняется)`;
  }

  /** ⇤ Защемить: стянуть область к её центру вдоль поверхности. */
  function pinchOnce() {
    if (!needRegion()) return;
    pushHist();
    const before = sym ? Float32Array.from(pos) : null;
    const pre = Float32Array.from(pos);
    const amt = Math.max(0.05, Math.min(0.6, amount / 200));
    const res = pinchRegion(topo, pos, frontMaskedRegion(), amt, 2);
    clampFlips(pos, pre, topo.tris, 0.05, 6, 0.15);
    symAfter(before);
    nrm = null;
    flush();
    if (info) info.textContent = `защемлено: ${res.moved} вершин, к центру на ${(res.maxd / U).toFixed(2)} мм`;
  }

  /** ＋ Шире / － Уже: выделение растёт или сжимается по кольцам рёбер. */
  function growOnce(k) {
    if (!needRegion()) return;
    pushHist();
    ensureTopo();
    const n = k > 0 ? growRegion(topo, reg, k) : shrinkRegion(topo, reg, -k);
    regionN = 0;
    for (let v = 0; v < vk; v++) if (reg[v]) regionN++;
    regionFront = null;
    paintMarks();
    flush();
    if (info) info.textContent = `${k > 0 ? 'выделение шире' : 'выделение уже'} на ${Math.abs(n)} вершин · всего ${regionN}`;
  }

  /** ⇋ Инверсия: всё видимое, кроме закрашенного. */
  function invertRegion() {
    pushHist();
    ensureTopo();
    /* центр — по текущей закраске (или по габариту сетки, если пусто): по нему
       строится маска «видимых граней», чтобы инверсия не залезла на изнанку */
    let cx = 0, cy = 0, cz = 0, n = 0;
    for (let v = 0; v < vk; v++) if (reg[v]) { cx += pos[v * 3]; cy += pos[v * 3 + 1]; cz += pos[v * 3 + 2]; n++; }
    if (n) { cx /= n; cy /= n; cz /= n; }
    else {
      let mnx = Infinity, mny = Infinity, mnz = Infinity, mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
      for (let v = 0; v < vk; v++) {
        mnx = Math.min(mnx, pos[v * 3]); mxx = Math.max(mxx, pos[v * 3]);
        mny = Math.min(mny, pos[v * 3 + 1]); mxy = Math.max(mxy, pos[v * 3 + 1]);
        mnz = Math.min(mnz, pos[v * 3 + 2]); mxz = Math.max(mxz, pos[v * 3 + 2]);
      }
      cx = (mnx + mxx) / 2; cy = (mny + mxy) / 2; cz = (mnz + mxz) / 2;
    }
    if (frontOnly) computeFront([cx, cy, cz]);
    let k = 0;
    const nr = new Uint8Array(vk);
    for (let v = 0; v < vk; v++) if ((!frontOnly || front[v]) && !reg[v]) { nr[v] = 1; k++; }
    reg = nr; regionN = k; regionFront = null;
    paintMarks();
    flush();
    if (info) info.textContent = `инверсия: выделено ${k} вершин`;
  }

  function extrude(sign) {
    if (!needRegion()) return;
    pushHist();
    refineRegion();
    const before = sym ? Float32Array.from(pos) : null;
    const soft = Math.max(1, Math.min(4, Math.round(Math.sqrt(regionN) / 3)));
    const pre = Float32Array.from(pos);
    const res = offsetRegion(topo, pos, frontMaskedRegion(), sign * depthD(), getNormals(), soft);
    clampFlips(pos, pre, topo.tris);
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
  const btnRect  = $('sc-rect');
  const btnGroove = $('sc-groove');
  const btnRope   = $('sc-rope');
  const ropeW     = $('sc-ropew');
  const btnPoint = $('sc-point');
  const btnSym   = $('sc-sym');
  const rows     = Array.from(document.querySelectorAll('.sc-row'));
  const hint     = $('sc-hint');
  const info     = $('sc-info');
  const chip     = $('sc-toolchip');
  const TOOL_NAMES = { select: 'участок', erase: 'ластик', pen: 'перо', shape: 'фигура',
    point: 'точка', measure: 'измерение', rect: 'рамка', groove: 'борозда', rope: 'канат' };
  function syncChip() {
    if (!chip) return;
    if (!tool) { chip.textContent = 'просмотр'; chip.classList.add('idle'); return; }
    chip.classList.remove('idle');
    chip.textContent = TOOL_NAMES[tool] || tool;
  }
  const HINTS = {
    select: hint ? hint.innerHTML : '',
    erase: 'Левой кнопкой <b>сотрите</b> лишние точки: они подсвечиваются красными ' +
      'точками и уходят по отпусканию кнопки, а дырка зашивается треугольниками по ' +
      'своей кромке — поверхность сама соединится с ближайшими оставшимися точками. ' +
      'С зажатым <b>Shift</b> — точечно: уходит один ближайший узел, кисть не работает.',
    pen: 'Левой кнопкой <b>ведите по поверхности</b>: под пером остаётся валик ' +
      'высотой «Глубина» и шириной «Кисть» — высота не копится за штрих, сколько ' +
      'бы вы ни водили мышью. С зажатым <b>Shift</b> — канавка (внутрь).',
    shape: 'Левой кнопкой <b>поставьте фигуру</b> на кузов (● сфера / ▮ цилиндр / ' +
      '■ куб, размер — «Кисть», глубина — «Глубина»). В пятне касания сетка сначала ' +
      'сгущается — добавляются точки по размеру кисти, — а потом эти точки ' +
      'вдавливаются на полную глубину: штамп получается жёстким, с плоским дном и ' +
      'резкой кромкой. Сгущение работает и на грубой сетке (куб с рёбрами 20 мм), ' +
      'и на мелочи: грань отбирается по расстоянию до треугольника, а не до его ' +
      'центра. С зажатым <b>Shift</b> — купол наружу.',
    measure: 'Два клика по поверхности — расстояние в мм (и разница по осям). ' +
      'Третий клик начинает новую пару.',
    rect: '<b>Растяните рамку</b> по кузову: стороны строго по осям экрана, выделяются ' +
      'только видимые грани и только та сторона, которой больше в рамке. Дальше ' +
      '«<b>⇄ Зеркально</b>» — часть другой стороны внутри той же рамки <b>вырезается</b>, ' +
      'а на её место встаёт <b>полная копия</b> выделенного (свои вершины и грани, ' +
      'зеркало y → −y); кромка копии привязывается к ближайшим оставшимся вершинам, ' +
      'щели зашиваются треугольниками — сетка остаётся замкнутой. «Радиус» — в каких ' +
      'пределах искать эти ближайшие вершины. Копия отменяется Ctrl+Z.',
    groove: '<b>Протяните линию по кузову</b> — вдоль неё, на всю длину протяжки, ' +
      'встанет <b>канавка сечением в половину цилиндра</b> (ширина = ползунок «Ширина», ' +
      'глубина = «Глубина»; глубина, равная половине ширины, даёт ровный полукруг). ' +
      'Торцы плоские, кромка плавная. ' +
      'Сначала в полосе появляется <b>новый набор точек</b> (сетка сгущается), и только ' +
      'эти точки вдавливаются: дно плоское, глубина одинаковая по всей длине, стенки ' +
      'резкие. Кузов вне полосы <b>не двигается</b> — скелет модели остаётся на месте. ' +
      '<b>Кисть</b> задаёт ширину желоба, <b>Глубина</b> — глубину. Протяжка всегда ' +
      'прямая: линия проводится между точкой нажатия и точкой отпускания — так контуры ' +
      'дверей получаются ровными.',
    rope: '<b>Проведите по кузову</b> — канат ложится ровно по граням (каждая его ' +
      'точка это попадание в треугольник), радиус — ползунок «Толщина». По отпусканию ' +
      'кнопки канат становится <b>отдельным объектом</b> сцены: своя трубка, свой ' +
      'материал, своё имя «Канат N»; кузов при этом не меняется. Кнопка ' +
      '«<b>🔥 Прожечь</b>» прорезает в кузове канал ровно под форму каната — ' +
      'полукруглый, на 6 % шире каната и глубиной в его радиус: канат садится в ' +
      'канал вровень, верхняя половина остаётся снаружи. «Глубина» делает канал ' +
      'глубже (до двух радиусов). Канал отменяется Ctrl+Z, канат остаётся; ' +
      '«⌫ Убрать» убирает канат, не трогая кузов.',
    point: '<b>Клик или протяжка по кузову</b> — новая точка ровно на существующей ' +
      'грани (грань делится на три, новой поверхности не появляется). Сразу после ' +
      'вставки ближайшие узлы притягиваются к новой точке (перевороты рёбер по ' +
      'Делоне), поэтому «иголок» не остаётся. Появляются зелёные метки. Ближе 15 % ' +
      'к вершинам грани точка не ставится — иначе получились бы вырожденные ' +
      'треугольники.'
  };

  function syncInfo(note) {
    syncChip();
    const regRow = $('sc-regrow');
    if (regRow) {
      regRow.classList.toggle('hd', !regionN);
      const btn = $('sc-deselect');
      if (btn && regionN) btn.textContent = `✕ Снять выделение · ${regionN}`;
    }
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
    } else if (tool === 'groove') {
      info.textContent = `борозда: ширина ${brushMM.toFixed(2)} мм, глубина ` +
        `${depthMM.toFixed(2)} мм · протяните линию по кузову` + (grooveInfo
          ? ` · прошлая: ${(grooveInfo.len / U).toFixed(0)} мм` : '');
    } else if (tool === 'measure') {
      info.textContent = 'измерение: два клика по поверхности';
    } else if (regionN) {
      const pct = (100 * regionN / vk).toFixed(0);
      info.textContent = `выбрано ${regionN} вершин (${pct} %), ` +
        `за нажатие ${Math.max(4, Math.round(amount / 100 * MAXSTEPS))} итераций`;
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
    show('sc-actions', tool === 'select' || tool === 'rect');
    show('sc-extrude-row', tool === 'select' || tool === 'rect');
    show('sc-pinchrow', tool === 'select' || tool === 'rect');
    show('sc-shaperow', tool === 'shape');
    show('sc-roperow', tool === 'rope' || ropes.length > 0);
    show('sc-fallrow', tool !== 'measure' && tool !== 'point' && tool !== 'erase' && tool !== 'rope');
    const depthRow = $('sc-depthrow');
    if (depthRow) depthRow.classList.toggle('hd',
      tool === 'erase' || tool === 'measure' || tool === 'point');
    /* у рамки ползунок — не кисть, а радиус поиска зеркальной пары */
    const brushRow = $('sc-brushrow');
    if (brushRow) brushRow.classList.toggle('hd', tool === 'measure' || tool === 'point' || tool === 'rope');
    const btag = $('sc-brushtag');
    if (btag) btag.textContent = tool === 'rect' ? 'Радиус' : tool === 'groove' ? 'Ширина' : 'Кисть';
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
      : tool === 'measure' ? 'crosshair' : tool === 'rope' ? 'copy' : 'crosshair';
  }

  /* подписи кнопок кистей: цифра = горячая клавиша */
  function syncToolButtons() {
    const tag = (btn, v, onText, offText) => {
      if (!btn) return;
      btn.classList.toggle('on', v);
      btn.textContent = v ? onText : offText;
    };
    tag(btnMode, tool === 'select', '■ Готово (участок)', '1 ✎ Участок');
    tag(btnDel, tool === 'erase', '■ Готово (ластик)', '2 ⌫ Ластик');
    tag(btnPen, tool === 'pen', '■ Готово (перо)', '3 ✏ Рисовать');
    tag(btnShape, tool === 'shape', '■ Готово (фигура)', '4 ◆ Фигура');
    tag(btnPoint, tool === 'point', '■ Готово (точка)', '5 ✚ Точка');
    tag(btnMeas, tool === 'measure', '■ Готово (измерение)', '6 📏 Измерить');
    tag(btnRect, tool === 'rect', '■ Готово (рамка)', '7 ▭ Рамка');
    tag(btnGroove, tool === 'groove', '■ Готово (борозда)', '8 ╱ Борозда');
    tag(btnRope, tool === 'rope', '■ Готово (канат)', '9 ╱ Канат');
  }

  function setTool(t) {
    if (t !== tool) hideCursor();
    const wasSelect = tool === 'select' && regionN > 0;
    tool = (tool === t) ? null : t;
    const keepRegion = wasSelect && tool && tool !== 'select';
    syncToolButtons();
    if (hint) hint.innerHTML = HINTS[tool] || HINTS.select;
    if (tool !== 'measure') { measPts = []; measLine.visible = false; }
    if (!tool) hideCursor();
    if (tool !== 'rect') { rectStart = null; rectLast = null; hideMarquee(); }
    block = null; allow = null;   // маска лицевых граней считается заново на каждом штрихе
    if (tool !== 'point') { ptSnap = null; lastPt = null; }
    syncRows();
    clearDots();
    syncControls();
    paintMarks();
    if (keepRegion) {
      syncInfo(`выделение сохранено: ${regionN} вершин — клавиша 1 вернёт к нему, 0 снимет`);
    } else {
      syncInfo();
    }
  }

  const cvs = renderer.domElement;
  cvs.addEventListener('contextmenu', e => { if (tool) e.preventDefault(); });
  cvs.addEventListener('pointerdown', e => {
    if (!tool || e.button !== 0) return;
    updateCursor(e);
    if (tool === 'measure') { addMeasurePoint(e.clientX, e.clientY); return; }
    painting = true;
    invert = e.shiftKey;
    lastW = null;                            // новый штрих — новый отсчёт шага
    try { cvs.setPointerCapture(e.pointerId); } catch (_) {}
    if (tool === 'rect') {
      rectStart = [e.clientX, e.clientY];
      rectLast = [e.clientX, e.clientY];
      showMarquee(e.clientX, e.clientY, e.clientX, e.clientY);
    } else if (tool === 'point') {
      lastPt = null;
      addPointsAt(e.clientX, e.clientY);
    } else if (tool === 'erase') {
      strokeDel = 0; clearDots(); strokeSnap = snapshot();
      const c0 = hitPoint(e.clientX, e.clientY);
      if (c0) computeFront(c0);
      makeBlock();
      eraseStamp(e.clientX, e.clientY);
    } else if (tool === 'pen') {
      pushHist(); nrm = null; penMoved = 0; penBase = Float32Array.from(pos);
      const c0 = hitPoint(e.clientX, e.clientY);
      if (c0) computeFront(c0);
      makeBlock();
      const r0 = brushR();
      if (c0) ensureDetail(c0, r0, 3000);
      penBase = Float32Array.from(pos);       // отсчёт — уже по сгущённой сетке
      ensureTopo(); nrm = null;
      penStamp(e.clientX, e.clientY, invert ? -1 : 1);
      flush();
    } else if (tool === 'groove') {
      grooveSnap = snapshot();
      grooveA = hitPoint(e.clientX, e.clientY);
      grooveB = null;
      if (grooveA) syncInfo('борозда: ведите до конца контура и отпустите кнопку');
      else syncInfo('нажмите на кузов — с поверхности начинается борозда');
    } else if (tool === 'shape') {
      pushHist(); nrm = null;
      shapeCount = 0; shapeLast = null;
      const c0 = hitPoint(e.clientX, e.clientY);
      if (c0) computeFront(c0);
      makeBlock();
      shapeDrag(e.clientX, e.clientY, true);
    } else if (tool === 'rope') {
      if (ropeMoveMode && ropes[ropeSel]) ropeMoveStart(e.clientX, e.clientY);
      else ropeStart(e.clientX, e.clientY);
    } else {
      const c0 = hitPoint(e.clientX, e.clientY);
      if (c0) computeFront(c0);
      makeBlock();
      stamp(e.clientX, e.clientY);
    }
  });
  cvs.addEventListener('pointermove', e => {
    updateCursor(e);                         // кольцо кисти видно до нажатия
    if (!painting || !tool) return;
    if (tool === 'rect') {
      if (rectStart) { rectLast = [e.clientX, e.clientY]; showMarquee(rectStart[0], rectStart[1], e.clientX, e.clientY); }
    }
    else if (tool === 'point') addPointsAt(e.clientX, e.clientY);
    else if (tool === 'erase') eraseStamp(e.clientX, e.clientY);
    /* Во время штриха нормали не пересчитываем: computeVertexNormals на
       100 000 граней — это ~200 мс на каждое движение мыши. Пересчёт один раз,
       по отпусканию кнопки (см. stop). */
    else if (tool === 'pen') { penStamp(e.clientX, e.clientY, invert ? -1 : 1); flush(false); }
    else if (tool === 'groove') {
      if (!grooveA) return;
      const p = hitPoint(e.clientX, e.clientY);
      if (!p) return;
      grooveB = p;
      const L = Math.hypot(p[0] - grooveA[0], p[1] - grooveA[1], p[2] - grooveA[2]);
      syncInfo(`борозда ${(L / U).toFixed(1)} мм · ширина ${brushMM.toFixed(2)} мм, ` +
        `глубина ${depthMM.toFixed(2)} мм — отпустите кнопку`);
    }
    else if (tool === 'shape') shapeDrag(e.clientX, e.clientY);
    else if (tool === 'rope') {
      if (ropeDrag) ropeMoveAt(e.clientX, e.clientY, e.shiftKey);
      else ropeAdd(e.clientX, e.clientY);
    }
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
      if (penBase) clampFlips(pos, penBase, topo.tris, 0.05, 6, 0.15);   // страховка от выворотов — один раз на штрих
      nrm = null;
      penBase = null;
      flush();                             // нормали и габарит — один раз на штрих
      paintMarks();
      if (info) info.textContent = `перо: сдвинуто ${penMoved} вершин, высота до ${depthMM.toFixed(2)} мм` +
        (invert ? ' (канавка)' : '');
    } else if (tool === 'point') {
      ensureTopo(); nrm = null;
      if (ptSnap) { hist.push(ptSnap); if (hist.length > 15) hist.shift(); redo.length = 0; ptSnap = null; }
      flush();
      updateAddedDots();
      if (ptMsg) { const m = ptMsg; ptMsg = null; syncInfo(m); }
      else syncInfo();
    }
  };
  cvs.addEventListener('pointerup', (e) => {
    if (tool === 'rect' && rectStart) {
      const [x0, y0] = rectStart;
      const [x1, y1] = rectLast || [e.clientX, e.clientY];
      rectStart = null; rectLast = null; painting = false;
      hideMarquee();
      rectSelect(x0, y0, x1, y1);
      return;
    }
    if (tool === 'groove' && grooveA) {
      const b = hitPoint(e.clientX, e.clientY) || grooveB;
      const a = grooveA;
      const snap = grooveSnap;
      grooveA = null; grooveB = null; grooveSnap = null; painting = false;
      if (b) {
        if (snap) pushHist(snap);
        makeGroove(a, b);
      } else syncInfo('борозда не сделана: отпустите кнопку на кузове');
      return;
    }
    if (tool === 'rope' && ropeDrag) {
      ropeMoveAt(e.clientX, e.clientY, e.shiftKey);
      ropeMoveEnd();
      painting = false;
      return;
    }
    if (tool === 'rope' && ropePts) {
      ropeAdd(e.clientX, e.clientY);
      painting = false;
      try { cvs.releasePointerCapture(e.pointerId); } catch (_) {}
      ropeFinish();
      return;
    }
    const nothing = tool === 'erase' && painting && !strokeDel;
    stop(e);
    if (nothing) syncInfo('здесь нечего стирать — под кистью нет точек');
  });
  cvs.addEventListener('pointercancel', stop);
  cvs.addEventListener('pointerleave', (e) => {
    /* при захвате указателя уход за холст не должен обрывать штрих */
    if (cvs.hasPointerCapture && cvs.hasPointerCapture(e.pointerId)) return;
    hideCursor();
    stop(e);
  });

  if (btnMode) btnMode.addEventListener('click', () => setTool('select'));
  if (btnDel) btnDel.addEventListener('click', () => setTool('erase'));
  if (btnPen) btnPen.addEventListener('click', () => setTool('pen'));
  if (btnShape) btnShape.addEventListener('click', () => setTool('shape'));
  if (btnMeas) btnMeas.addEventListener('click', () => setTool('measure'));
  if (btnPoint) btnPoint.addEventListener('click', () => setTool('point'));
  if (btnRect) btnRect.addEventListener('click', () => setTool('rect'));
  if (btnGroove) btnGroove.addEventListener('click', () => setTool('groove'));
  if (btnRope) btnRope.addEventListener('click', () => setTool('rope'));
  if (ropeW) {
    const ropeWv = $('sc-ropewv');
    const showW = () => {
      ropeWMM = Math.min(12, Math.max(0.4, +ropeW.value || 2.5));
      if (ropeWv) ropeWv.textContent = `${ropeWMM.toFixed(1)} мм`;
      syncRopeRow();
      if (ropePrev) ropePreviewDraw();
    };
    ropeW.value = String(ropeWMM);
    showW();
    ropeW.addEventListener('input', showW);
  }
  const btnAlign = $('sc-rope-align');
  if (btnAlign) btnAlign.addEventListener('click', () => ropeAlign('smooth'));
  const btnLine = $('sc-rope-line');
  if (btnLine) btnLine.addEventListener('click', () => ropeAlign('line'));
  const btnMove = $('sc-rope-move');
  if (btnMove) btnMove.addEventListener('click', () => {
    ropeMoveMode = !ropeMoveMode;
    btnMove.classList.toggle('on', ropeMoveMode);
    btnMove.textContent = ropeMoveMode ? '■ Двигать' : '↔ Двигать';
    syncInfo(ropeMoveMode
      ? 'канат: тяните по кузову мышью — он останется на гранях (Shift — приподнять/утопить)'
      : 'канат: снова прокладывается линией по кузову');
  });
  const btnReset = $('sc-rope-reset');
  if (btnReset) btnReset.addEventListener('click', () => ropeReset());
  const ropeQEl = $('sc-ropeq');
  if (ropeQEl) {
    const showQ = () => { ropeQual = Math.max(1, Math.min(3, +ropeQEl.value || 2)); syncRopeRow(); };
    showQ();
    ropeQEl.addEventListener('change', showQ);
  }
  const btnBurn = $('sc-burn');
  if (btnBurn) btnBurn.addEventListener('click', () => burnRope());
  const btnRopeDel = $('sc-rope-del');
  if (btnRopeDel) btnRopeDel.addEventListener('click', () => removeRope());
  const btnRopeSave = $('sc-rope-save');
  if (btnRopeSave) btnRopeSave.addEventListener('click', () => saveRope('3mf'));
  syncRopeRow();

  /* ограничение правок лицевыми гранями */
  const chkFront = $('sc-front');
  if (chkFront) {
    chkFront.checked = frontOnly;
    chkFront.addEventListener('change', () => {
      frontOnly = chkFront.checked;
      chkFront.classList.toggle('on', frontOnly);
      chkFront.textContent = frontOnly ? '◐ Видимые грани' : '◐ Все грани';
      block = null; allow = null; regionFront = null; front.fill(frontOnly ? 0 : 1);
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
  const fallEl = $('sc-fall');
  if (fallEl) fallEl.addEventListener('change', () => { setFalloff(fallEl.value); syncInfo(); });
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
  on('sc-relax', relaxOnce);
  on('sc-pinch', pinchOnce);
  on('sc-grow', () => growOnce(1));
  on('sc-shrink', () => growOnce(-1));
  on('sc-invert', invertRegion);

  /* сетка */
  const wireBtn = $('wire');
  if (wireBtn) wireBtn.addEventListener('click', () => {
    wireOn = !wireOn;
    wireBtn.classList.toggle('on', wireOn);
    wireBtn.textContent = wireOn ? '▦ Сетка вкл' : '▦ Сетка';
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
    secBtn.textContent = secOn ? '◧ Сечение вкл' : '◧ Сечение';
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
  on('sc-deselect', deselect);
  on('sc-mirror', mirrorCopy);
  on('sc-proj-save', saveProject);
  on('sc-proj-load', () => { const f = $('sc-file'); if (f) f.click(); });
  on('sc-file', (e) => {
    const file = e && e.target && e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { loadProject(String(reader.result)); if (e.target) e.target.value = ''; };
    reader.readAsText(file);
  }, 'change');
  on('sc-clear', () => {
    pushHist();
    if (count !== origCount || liveIndex.length !== origIndex.length) {
      /* точки добавлялись — возвращаем исходную геометрию целиком */
      count = origCount;
      applySrc(Float32Array.from(origSrc));
      col = new Float32Array(count * 3).fill(1);
      geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
      setIndex(origIndex.slice());
      topo = buildTopology(src, liveIndex);
      canon = topo.canon; vk = Math.min(topo.vk, reg.length);
      topo.vk = vk; topo.count = count; topo.pos0 = pos0; topoDirty = false;
    }
    pos0.fill(0); pos0.set(posInit.subarray(0, Math.min(posInit.length, pos0.length)));
    pos.fill(0); pos.set(posInit.subarray(0, Math.min(posInit.length, pos.length)));
    reg.fill(0); regionN = 0; regionFront = null;
    dead.fill(0); delTotal = 0;
    added.fill(0); addedTotal = 0; front.fill(1); block = null; allow = null;
    setIndex(origIndex.slice());
    nrm = null;
    dirty = false;
    paintMarks(); updateAddedDots(); flush(); syncInfo();
  });
  /* ---------- сохранение и загрузка правки ----------
     В 3MF/STL уходит готовая печать, а этот файл — чтобы продолжить работу:
     смещения вершин от исходного кузова, маски (выделение, удалённые,
     добавленные узлы) и индексы, если топология менялась. Файл маленький:
     пишутся только те вершины, которые действительно сдвинуты. */
  const r4 = (x) => Math.round(x * 1e4) / 1e4;
  /* Индексы после правки топологии отличаются от исходных локально: храним
     только середину между общим началом и общим концом. Один вставленный узел
     даёт вместо 2 МБ чисел несколько десятков. */
  function indexPacked() {
    const a = liveIndex, b = origIndex;
    const n = Math.min(a.length, b.length);
    let p = 0;
    while (p < n && a[p] === b[p]) p++;
    let s = 0;
    while (s < n - p && a[a.length - 1 - s] === b[b.length - 1 - s]) s++;
    return { base: b.length, p, s, ins: Array.from(a.subarray(p, a.length - s)) };
  }
  function indexUnpacked(pack) {
    if (!pack || !(pack.base === origIndex.length) || pack.p < 0 || pack.s < 0 ||
        pack.p + pack.s > origIndex.length || !Array.isArray(pack.ins)) return null;
    const out = new Int32Array(pack.p + pack.ins.length + pack.s);
    out.set(origIndex.subarray(0, pack.p), 0);
    out.set(Int32Array.from(pack.ins), pack.p);
    out.set(origIndex.subarray(origIndex.length - pack.s), pack.p + pack.ins.length);
    return out;
  }
  function projectData() {
    flush();                                  // pos → src, чтобы сохранить актуальное
    const sameIndex = liveIndex.length === origIndex.length &&
      liveIndex.every((x, i) => x === origIndex[i]);
    const delta = [];
    for (let i = 0; i < count; i++) {
      const o = i * 3;
      if (i < origCount) {
        if (src[o] === origSrc[o] && src[o + 1] === origSrc[o + 1] && src[o + 2] === origSrc[o + 2]) continue;
      }
      delta.push(i, r4(src[o]), r4(src[o + 1]), r4(src[o + 2]));
    }
    const list = (m) => { const a = []; for (let v = 0; v < vk; v++) if (m[v]) a.push(v); return a; };
    return {
      app: 'tayron-sculpt', v: 1, when: new Date().toISOString(), U,
      count, vk, origCount, sameIndex,
      index: sameIndex ? null : indexPacked(),
      verts: delta, reg: list(reg), dead: list(dead), added: list(added),
      regionN, delTotal, addedTotal
    };
  }
  function saveProject() {
    const data = projectData();
    const json = JSON.stringify(data);
    const name = 'Tayron_body_edit.json';
    const res = downloadBytes(new TextEncoder().encode(json), name, 'application/json');
    dirty = false;
    if (info) {
      info.innerHTML = `правка сохранена: <a href="${res.url}" download="${name}" style="color:var(--acc)">` +
        `скачать ${name}</a> · ${(json.length / 1024).toFixed(1)} КБ · сдвинуто вершин ` +
        `${data.verts.length / 4}` + (data.sameIndex ? '' : ' · с новой топологией');
    }
    return data;
  }
  function loadProject(text) {
    let d = null;
    try { d = JSON.parse(text); } catch (_) { d = null; }
    if (!d || d.app !== 'tayron-sculpt') {
      syncInfo('это не правка Tayron: нужен файл, сохранённый кнопкой «Сохранить»');
      return false;
    }
    if (!Array.isArray(d.verts) || !(d.count >= origCount) || (d.index && !d.index.ins)) {
      syncInfo('файл неполный или сохранён от другой модели');
      return false;
    }
    if (d.U && Math.abs(d.U - U) > 1e-6) {
      syncInfo(`файл от модели с другим масштабом (${d.U} против ${U})`);
      return false;
    }
    const useIndex = d.index ? indexUnpacked(d.index) : origIndex.slice();
    if (!useIndex) { syncInfo('файл правки от другой топологии — не могу восстановить индексы'); return false; }
    pushHist();
    count = d.count | 0;
    const next = new Float32Array(count * 3);
    next.set(origSrc.subarray(0, Math.min(origSrc.length, next.length)));
    for (let k = 0; k + 3 < d.verts.length; k += 4) {
      const i = d.verts[k] | 0, o = i * 3;
      if (i < 0 || o + 2 >= next.length) continue;
      next[o] = d.verts[k + 1]; next[o + 1] = d.verts[k + 2]; next[o + 2] = d.verts[k + 3];
    }
    applySrc(next);
    col = new Float32Array(count * 3).fill(1);
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    setIndex(useIndex);
    topo = buildTopology(src, liveIndex);
    vk = topo.vk; canon = topo.canon; pos0 = topo.pos0;
    topo.vk = vk; topo.count = count; topo.pos0 = pos0; topoDirty = false;
    resizeNodeArrays(Math.max(vk, reg.length, dead.length, added.length));
    pos = Float32Array.from(pos0);
    reg.fill(0); dead.fill(0); added.fill(0); pending.fill(0); front.fill(1);
    const mark = (m, arr) => { let n = 0; if (Array.isArray(arr)) for (const v of arr) { if (v >= 0 && v < vk) { m[v] = 1; n++; } } return n; };
    regionN = mark(reg, d.reg);
    delTotal = mark(dead, d.dead);
    addedTotal = mark(added, d.added);
    block = null; allow = null; regionFront = null; nrm = null;
    free = freeMask(topo, reg);
    clearDots(); paintMarks(); updateAddedDots(); flush();
    dirty = false;
    if (d.vk && d.vk !== vk) {
      syncInfo(`правка загружена, но узлов стало ${vk} вместо ${d.vk} — маски пересчитаны`);
    } else {
      syncInfo(`правка загружена: ${count} верт. · сдвинуто ${d.verts.length / 4} · ` +
        `удалено ${delTotal} · добавлено ${addedTotal}`);
    }
    return true;
  }

  /* ---------- горячие клавиши: 1…6 — кисти, Esc — выход, [ ] — размер кисти ---------- */
  const TOOL_KEYS = { '1': 'select', '2': 'erase', '3': 'pen', '4': 'shape', '5': 'point',
    '6': 'measure', '7': 'rect', '8': 'groove', '9': 'rope' };
  function stepBrush(d) {
    brushMM = Math.min(20, Math.max(0.1, Math.round((brushMM + d) * 10) / 10));
    if (brushEl) brushEl.value = String(brushMM);
    const l = $('sc-brushv');
    if (l) l.textContent = brushMM.toFixed(1) + ' мм';
    syncInfo();
  }
  globalThis.addEventListener && globalThis.addEventListener('keydown', (e) => {
    const tn = e.target && e.target.tagName;
    if (tn === 'INPUT' || tn === 'TEXTAREA' || tn === 'SELECT') return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;       // Ctrl+Z — ниже
    if (TOOL_KEYS[e.key]) { setTool(TOOL_KEYS[e.key]); e.preventDefault(); return; }
    if (e.key === 'Escape' && tool) { setTool(tool); e.preventDefault(); return; }
    if (e.key === '[') { stepBrush(-0.5); e.preventDefault(); return; }
    if (e.key === ']') { stepBrush(0.5); e.preventDefault(); return; }
    if (e.key === '0') { deselect(); e.preventDefault(); }
  });
  /* закрывать вкладку с несохранённой правкой без предупреждения нельзя */
  globalThis.addEventListener && globalThis.addEventListener('beforeunload', (e) => {
    if (!dirty) return;
    e.preventDefault();
    e.returnValue = 'в правке есть несохранённые изменения — сохраните её (💾)';
    return e.returnValue;
  });
  globalThis.addEventListener && globalThis.addEventListener('keydown', (e) => {
    if (!e.ctrlKey && !e.metaKey) return;
    const k = (e.key || '').toLowerCase();
    if (k === 's') { e.preventDefault(); saveProject(); return; }   // Ctrl+S — сохранить правку
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

  syncToolButtons();   // подписи кнопок не ждут первого клика
  syncInfo();

  return {
    get mode() { return !!tool; },
    get tool() { return tool; },
    get erasing() { return tool === 'erase'; },
    setTool,
    deselect,
    mirrorCopy,
    makeGroove,
    get groove() { return grooveInfo; },
    grooveAt(clientX, clientY) { const p = hitPoint(clientX, clientY); return p; },
    rectSelect,
    saveProject,
    loadProject,
    get dirty() { return dirty; },
    projectData,
    get projectJSON() { return JSON.stringify(projectData()); },
    get regionCount() { return regionN; },
    get deletedCount() { return delTotal; },
    get symmetrical() { return sym; },
    /* канат: отдельные объекты сцены и прожиг канала под их форму */
    get ropeCount() { return ropes.length; },
    get ropeSelected() { return ropeSel; },
    get ropeDrawing() { return !!ropePts; },
    ropeList: () => ropes.map((rp, i) => ({
      i, name: rp.name, len: pathLen(rp.path), radius: rp.radius, off: rp.off, burn: rp.burn,
      verts: rp.tube ? rp.tube.verts : 0, tris: rp.tube ? rp.tube.index.length / 3 : 0,
      inScene: (() => {
        const par = mesh.parent || mesh;
        return !!(rp.mesh && par && par.children && par.children.includes(rp.mesh));
      })()
    })),
    ropeMesh: (i = ropeSel) => (ropes[i] ? ropes[i].mesh : null),
    ropePath: (i = ropeSel) => (ropes[i] ? ropes[i].path : null),
    addRope(pts, nrms, radiusMM) {
      if (!pts || pts.length < 2) return null;
      const R = Math.max(0.2, (radiusMM || ropeWMM)) * U;
      const sm = smoothPath(pts.map(q => q.slice()), nrms ? nrms.map(q => q.slice()) : null, 1);
      const rs = resamplePath(sm.pts, sm.nrms, Math.max(0.45 * R, 0.25 * U));
      if (pathLen(rs.pts) < 4 * R) return null;
      return newRope(rs.pts, rs.nrms, R);
    },
    burnRope: (i = ropeSel) => burnRope(i),
    ropeAlign: (kind) => ropeAlign(kind),
    ropeLift: (d) => ropeLift(d),
    ropeReset: () => ropeReset(),
    ropeHealth: (i = ropeSel) => {
      const rp = ropes[i];
      if (!rp) return null;
      return { turn: pathTurn(rp.path), kink: pathKink(rp.path), len: pathLen(rp.path),
        points: rp.path.length, off: rp.off, radius: rp.radius };
    },
    get ropeMoveMode() { return ropeMoveMode; },
    setRopeMoveMode(v) { ropeMoveMode = !!v; if (btnMove) { btnMove.classList.toggle('on', ropeMoveMode); btnMove.textContent = ropeMoveMode ? '■ Двигать' : '↔ Двигать'; } return ropeMoveMode; },
    setRopeQuality(q) { ropeQual = Math.max(1, Math.min(3, +q || 2)); if (ropeQEl) ropeQEl.value = String(ropeQual); syncRopeRow(); return ropeQual; },
    get ropeQuality() { return ropeQual; },
    removeRope: (i = ropeSel) => removeRope(i),
    saveRope: (kind) => saveRope(kind),
    setRopeThickness(mm) {
      ropeWMM = Math.min(12, Math.max(0.4, +mm || 2.5));
      if (ropeW) { ropeW.value = String(ropeWMM); }
      const v = $('sc-ropewv');
      if (v) v.textContent = `${ropeWMM.toFixed(1)} мм`;
      syncRopeRow();
      return ropeWMM;
    },
    selectRope(i) {
      if (i < 0 || i >= ropes.length) return -1;
      ropeSel = i; tintRopes(); syncRopeRow();
      return ropeSel;
    },
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
