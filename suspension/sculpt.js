/* ============================================================
   «Сглаживание кузова» на сайте — с лепкой.

   Инструменты (все двигают только существующие вершины):
   1. ✎ Участок — кисть закрашивает область (бирюзовым), дальше:
        «≡ Сгладить» (диффузия по рёбрам), «⬆ Выдавить» / «⬇ Вдавить»
        (область уезжает вдоль нормалей, у кромки шаг гасится).
   2. ✏ Рисовать — перо: ведёте по поверхности, под ним остаётся валик
      (Shift — канавка). Ширина — «Кисть», высота — «Глубина».
   3. ◆ Фигура — ставит выбранную фигуру (сфера / цилиндр / куб) на
      поверхность: точки внутри фигуры вдавливаются в кузов по её форме.
      Shift — наоборот, фигура вытягивает купол.
   4. ⌫ Ластик — точки под кистью подсвечиваются красными точками и уходят по
      отпусканию кнопки, дырка зашивается треугольниками по своей кромке
      (meshops.deleteAndFill). Один штрих = один шаг «Отменить».

   Кнопка «Сетка» (в общих пресетах) показывает триангуляцию тонкими чёрными
   линиями. «⬇ Скачать 3MF» отдаёт кузов файлом прямо из браузера.

   Всё в единицах модели (1 ед. = 0.17 мм печати, U = 5.8824 ед./мм).
   ============================================================ */
import * as THREE from 'three';
import { meshTo3mf, downloadBytes } from './export3mf.js';
import {
  buildTopology, freeMask, smoothDiffuse, boundaryFalloff, paintBall,
  expandToVertices, deleteAndFill, compactMesh, vertexNormals,
  pushBall, offsetRegion, imprintShape
} from './meshops.js';

const MAXSTEPS = 600;   // итераций за одно нажатие при «Степень» 100 %
const LAMBDA   = 0.5;   // доля шага диффузии

export function initSculpt(opts) {
  const { mesh, material, camera, renderer, controls, U } = opts;
  const geom = mesh.geometry;
  const attr = geom.attributes.position;
  const count = attr.count;
  if (!geom.index) {
    const idx = new Uint32Array(count);
    for (let i = 0; i < count; i++) idx[i] = i;
    geom.setIndex(new THREE.BufferAttribute(idx, 1));
  }
  const src = attr.array;                 // координаты вершин геометрии (как в файле)
  const origIndex = Int32Array.from(geom.index.array);  // индексы исходной сетки
  let liveIndex = origIndex.slice();                    // текущие индексы (для экспорта)
  const topo = buildTopology(src, liveIndex);
  const { vk, pos0, canon } = topo;

  const pos = Float32Array.from(pos0);
  const reg = new Uint8Array(vk);
  const dead = new Uint8Array(vk);        // узлы, убранные ластиком
  const pending = new Uint8Array(vk);     // узлы, помеченные в текущем штрихе (для точек)
  let delTotal = 0;                       // сколько узлов удалено всего
  let free = new Uint8Array(vk);
  let regionN = 0;
  let nrm = null;                         // нормали узлов, считаются по требованию

  /* ---------- подсветка области ---------- */
  const col = new Float32Array(count * 3).fill(1);
  geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
  material.vertexColors = true;
  material.needsUpdate = true;
  const CANON_COL = new Float32Array(vk * 3).fill(1);
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
  const dotPos = new Float32Array(vk * 3);
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
  const hist = [];
  function snapshot() {
    return {
      pos: Float32Array.from(pos), reg: reg.slice(), dead: dead.slice(),
      n: regionN, del: delTotal, index: liveIndex
    };
  }
  function pushHist(snap) {
    hist.push(snap || snapshot());
    if (hist.length > 15) hist.shift();
  }
  function setIndex(ix) {
    liveIndex = ix;
    geom.setIndex(new THREE.BufferAttribute(new Uint32Array(ix), 1));
    if (wireOn) rebuildWire();
  }

  /* ---------- кисть ---------- */
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let brushMM = 0.5;        // радиус кисти в мм печати (0.1 … 20)
  let depthMM = 0.5;        // глубина лепки в мм печати
  let amount = 25;          // «степень» в процентах: сколько итераций за нажатие
  let painting = false;
  let invert = false;       // Shift: рисовать внутрь / вытягивать фигурой
  let strokeSnap = null;    // снимок для «Отменить» — до начала штриха
  let strokeDel = 0;        // сколько удалено в текущем штрихе
  let tool = null;          // 'select' | 'erase' | 'pen' | 'shape'
  let shapeKind = 'sphere';

  const brushR = () => brushMM * U;
  const depthD = () => depthMM * U;

  function getNormals() {
    if (!nrm) nrm = vertexNormals(topo, liveIndex, pos);
    return nrm;
  }

  /** точка под курсором в локальных координатах кузова */
  function hit(clientX, clientY) {
    ndc.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObject(mesh, false);
    if (!hits.length) return null;
    return mesh.worldToLocal(hits[0].point.clone());
  }

  /** ближайший живой узел — чтобы кисть меньше шага сетки всё-таки работала */
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
  function stamp(clientX, clientY) {
    const p = hit(clientX, clientY);
    if (!p) return 0;
    const c = [p.x, p.y, p.z];
    let added = paintBall(topo, reg, c, brushR(), pos, dead);
    if (!added) {                         // кисть мельче сетки — берём ближайшую точку
      const v = nearestNode(c, dead);
      if (v >= 0 && !reg[v] && !dead[v]) { reg[v] = 1; added = 1; }
    }
    if (added) { regionN += added; paintMarks(); syncInfo(); }
    return added;
  }

  /* ---------- 2. ластик ---------- */
  function eraseStamp(clientX, clientY) {
    const p = hit(clientX, clientY);
    if (!p) return 0;
    const c = [p.x, p.y, p.z];
    const r2 = brushR() * brushR();
    let n = 0;
    for (let v = 0; v < vk; v++) {
      if (dead[v]) continue;
      const o = v * 3;
      const dx = pos[o] - c[0], dy = pos[o + 1] - c[1], dz = pos[o + 2] - c[2];
      if (dx * dx + dy * dy + dz * dz > r2) continue;
      dead[v] = 1; pending[v] = 1; delTotal++; n++;
      if (reg[v]) { reg[v] = 0; regionN--; }
    }
    if (!n) {                             // кисть мельче сетки — берём ближайшую точку
      const v = nearestNode(c, dead);
      if (v >= 0) {
        dead[v] = 1; pending[v] = 1; delTotal++; n = 1;
        if (reg[v]) { reg[v] = 0; regionN--; }
      }
    }
    if (n) { strokeDel += n; paintMarks(); updateDots(); syncInfo(); }
    return n;
  }

  /* ---------- 3. перо: валик вдоль штриха ---------- */
  let penTicks = 0;
  function penStamp(clientX, clientY, sign) {
    const p = hit(clientX, clientY);
    if (!p) return 0;
    if (++penTicks % 8 === 0) nrm = null;  // поверхность уехала — обновляем нормали
    const c = [p.x, p.y, p.z];
    const res = pushBall(topo, pos, c, brushR(), sign * depthD(), getNormals(), dead);
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
    return res.moved;
  }

  /* ---------- 4. фигура: вдавливается в кузов ---------- */
  function shapeStamp(clientX, clientY, sign) {
    const p = hit(clientX, clientY);
    if (!p) return null;
    const c = [p.x, p.y, p.z];
    // нормаль поверхности в точке — усредняем по ближайшим узлам
    const n = getNormals();
    let nx = 0, ny = 0, nz = 0, wsum = 0;
    const r2 = brushR() * brushR();
    for (let v = 0; v < vk; v++) {
      const o = v * 3;
      const dx = pos[o] - c[0], dy = pos[o + 1] - c[1], dz = pos[o + 2] - c[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 > r2) continue;
      const w = 1 - Math.sqrt(d2) / brushR() + 1e-6;
      nx += n[o] * w; ny += n[o + 1] * w; nz += n[o + 2] * w; wsum += w;
    }
    if (wsum > 0) { nx /= wsum; ny /= wsum; nz /= wsum; }
    else { nx = 0; ny = 0; nz = 1; }
    const shape = {
      kind: shapeKind, center: c, n: [nx, ny, nz],
      size: { r: brushR(), h: Math.max(brushR() * 0.8, depthD()), a: brushR() * 0.8, b: brushR() * 0.8 }
    };
    const res = imprintShape(topo, pos, shape, sign * depthD(), getNormals());
    return { ...res, kind: shapeKind };
  }

  /* ---------- зашивка: убрать точки и закрыть дырку ---------- */
  function applyErase(n) {
    if (!n) return;
    const res = deleteAndFill(topo, liveIndex, pos, dead);
    setIndex(res.index);
    clearDots();
    // зашивка могла убрать ещё несколько точек (кромка касалась себя)
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

  /* ---------- «Сгладить»: мягкая диффузия по рёбрам ---------- */
  function smoothOnce() {
    if (!regionN) { syncInfo('сначала закрасьте участок кистью'); return; }
    const steps = Math.max(20, Math.round(amount / 100 * MAXSTEPS));
    pushHist();
    free = freeMask(topo, reg);
    const w = boundaryFalloff(topo, reg, 4);
    const p = smoothDiffuse(topo, free, steps, LAMBDA, pos);
    for (let v = 0; v < vk; v++) {          // плавный вход в область
      if (w[v] >= 1) continue;
      const o = v * 3;
      for (let c = 0; c < 3; c++) p[o + c] = pos[o + c] + (p[o + c] - pos[o + c]) * w[v];
    }
    pos.set(p);
    nrm = null;
    flush();
    syncInfo();
  }

  /* ---------- «Выдавить» / «Вдавить»: область вдоль нормалей ---------- */
  function extrude(sign) {
    if (!regionN) { syncInfo('сначала закрасьте участок кистью'); return; }
    pushHist();
    // ширина плавного входа зависит от размера области: на маленьком пятне
    // иначе «съедается» весь ход
    const soft = Math.max(1, Math.min(4, Math.round(Math.sqrt(regionN) / 3)));
    const res = offsetRegion(topo, pos, reg, sign * depthD(), getNormals(), soft);
    nrm = null;
    flush();
    if (info) {
      info.textContent = res.moved
        ? `${sign > 0 ? 'выдавлено' : 'вдавлено'} на ${depthMM.toFixed(2)} мм · ` +
          `сдвинуто ${res.moved} из ${regionN} вершин области ` +
          `(макс. ${(res.maxd / U).toFixed(2)} мм)`
        : 'область слишком узкая — закрасьте участок пошире';
    }
  }

  /* ---------- интерфейс ---------- */
  const $ = (id) => document.getElementById(id);
  const btnMode  = $('sc-mode');
  const btnDel   = $('sc-del');
  const btnPen   = $('sc-pen');
  const btnShape = $('sc-shape');
  const rows     = Array.from(document.querySelectorAll('.sc-row'));
  const hint     = $('sc-hint');
  const info     = $('sc-info');
  const HINTS = {
    select: hint ? hint.innerHTML : '',
    erase: 'Левой кнопкой <b>сотрите</b> лишние точки: они подсвечиваются красными ' +
      'точками и уходят по отпусканию кнопки, а дырка зашивается треугольниками по ' +
      'своей кромке — поверхность сама соединится с ближайшими оставшимися точками. ' +
      'Правой — вращать камеру, колесом — приблизить.',
    pen: 'Левой кнопкой <b>ведите по поверхности</b>: под пером остаётся валик ' +
      'высотой «Глубина» и шириной «Кисть». С зажатым <b>Shift</b> — канавка ' +
      '(внутрь). Двигаются только существующие вершины, сетка не меняется.',
    shape: 'Левой кнопкой <b>поставьте фигуру</b> на кузов — она вдавится в ' +
      'поверхность (сфера / цилиндр / куб выбираются ниже, размер — «Кисть», ' +
      'глубина — «Глубина»). С зажатым <b>Shift</b> фигура, наоборот, вытянет ' +
      'купол наружу.'
  };

  function syncInfo(note) {
    if (!info) return;
    if (note) { info.textContent = note; return; }
    const faces = liveIndex.length / 3;
    if (tool === 'erase') {
      info.textContent = `ластик: за один штрих удаляется всё под кистью ` +
        `(${brushMM.toFixed(1)} мм) · в сетке ${faces} граней`;
    } else if (tool === 'pen') {
      info.textContent = `перо ${brushMM.toFixed(1)} мм, высота ${depthMM.toFixed(2)} мм` +
        ` · в сетке ${faces} граней`;
    } else if (tool === 'shape') {
      info.textContent = `фигура «${SHAPE_NAMES[shapeKind]}»: радиус ${brushMM.toFixed(1)} мм, ` +
        `глубина ${depthMM.toFixed(2)} мм`;
    } else if (regionN) {
      const pct = (100 * regionN / vk).toFixed(0);
      info.textContent = `выбрано ${regionN} вершин (${pct} %), ` +
        `за нажатие ${Math.max(20, Math.round(amount / 100 * MAXSTEPS))} итераций`;
    } else {
      info.textContent = 'область не выбрана — закрасьте участок кистью';
    }
  }

  const SHAPE_NAMES = { sphere: 'сфера', cylinder: 'цилиндр', box: 'куб' };

  /** Показ/скрытие строк под текущий инструмент */
  function syncRows() {
    const on = !!tool;
    rows.forEach(r => r.classList.toggle('hd', !on));
    if (hint) hint.classList.toggle('hd', !on);
    if (!on) return;
    const show = (id, v) => { const e = $(id); if (e) e.classList.toggle('hd', !v); };
    show('sc-amtrow', tool === 'select');
    show('sc-go', tool === 'select');
    show('sc-extrude-row', tool === 'select');
    show('sc-shaperow', tool === 'shape');
    const depthRow = $('sc-depthrow');
    if (depthRow) depthRow.classList.toggle('hd', tool === 'erase');
  }

  function setTool(t) {
    tool = (tool === t) ? null : t;
    const on = !!tool;
    const tag = (btn, v, onText, offText) => {
      if (!btn) return;
      btn.classList.toggle('on', v);
      btn.textContent = v ? onText : offText;
    };
    tag(btnMode, tool === 'select', '■ Закончить участок', '✎ Участок');
    tag(btnDel, tool === 'erase', '■ Закончить удаление', '⌫ Ластик');
    tag(btnPen, tool === 'pen', '■ Закончить рисование', '✏ Рисовать');
    tag(btnShape, tool === 'shape', '■ Закончить фигуры', '◆ Фигура');
    if (hint) hint.innerHTML = HINTS[tool] || HINTS.select;
    syncRows();
    clearDots();
    renderer.domElement.style.cursor = !on ? ''
      : tool === 'erase' ? 'cell' : tool === 'shape' ? 'copy' : 'crosshair';
    const MB = THREE.MOUSE;
    if (on) {
      controls.mouseButtons = { LEFT: null, MIDDLE: MB.DOLLY, RIGHT: MB.ROTATE };
      controls.touches = { ONE: null, TWO: MB.DOLLY_ROTATE };
    } else {
      controls.mouseButtons = { LEFT: MB.ROTATE, MIDDLE: MB.DOLLY, RIGHT: MB.PAN };
      controls.touches = { ONE: MB.ROTATE, TWO: MB.DOLLY_PAN };
    }
    paintMarks();
    syncInfo();
  }

  const cvs = renderer.domElement;
  cvs.addEventListener('contextmenu', e => { if (tool) e.preventDefault(); });
  cvs.addEventListener('pointerdown', e => {
    if (!tool || e.button !== 0) return;
    painting = true;
    invert = e.shiftKey;
    try { cvs.setPointerCapture(e.pointerId); } catch (_) {}
    if (tool === 'erase') {
      strokeDel = 0; clearDots(); strokeSnap = snapshot();
      eraseStamp(e.clientX, e.clientY);
    } else if (tool === 'pen') {
      pushHist(); nrm = null;
      penStamp(e.clientX, e.clientY, invert ? -1 : 1);
      flush();
    } else if (tool === 'shape') {
      pushHist();
      nrm = null;
      const res = shapeStamp(e.clientX, e.clientY, invert ? -1 : 1);
      nrm = null;
      flush();
      if (res && info) {
        info.textContent = `фигура «${SHAPE_NAMES[res.kind]}» ${invert ? 'вытянула' : 'вдавила'} ` +
          `поверхность: ${res.moved} вершин, макс. ${(res.maxd / U).toFixed(2)} мм`;
      }
      painting = false;
    } else {
      stamp(e.clientX, e.clientY);
    }
  });
  cvs.addEventListener('pointermove', e => {
    if (!painting || !tool) return;
    if (tool === 'erase') eraseStamp(e.clientX, e.clientY);
    else if (tool === 'pen') { penStamp(e.clientX, e.clientY, invert ? -1 : 1); flush(); }
    else if (tool === 'select') stamp(e.clientX, e.clientY);
  });
  const stop = () => {
    if (!painting) return;
    painting = false;
    if (tool === 'erase' && strokeDel) {
      const n = strokeDel;
      strokeDel = 0;
      pushHist(strokeSnap);           // «Отменить» вернёт сетку и все точки до штриха
      strokeSnap = null;
      applyErase(n);
    }
  };
  cvs.addEventListener('pointerup', stop);
  cvs.addEventListener('pointercancel', stop);
  cvs.addEventListener('pointerleave', stop);

  if (btnMode) btnMode.addEventListener('click', () => setTool('select'));
  if (btnDel) btnDel.addEventListener('click', () => setTool('erase'));
  if (btnPen) btnPen.addEventListener('click', () => setTool('pen'));
  if (btnShape) btnShape.addEventListener('click', () => setTool('shape'));

  // выбор фигуры
  const shapeBtns = ['sphere', 'cylinder', 'box'];
  shapeBtns.forEach((k) => {
    const b = $('sc-fig-' + k);
    if (b) b.addEventListener('click', () => {
      shapeKind = k;
      shapeBtns.forEach((k2) => {
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
  const goBtn = $('sc-go');
  if (goBtn) goBtn.addEventListener('click', smoothOnce);
  const outBtn = $('sc-out');
  if (outBtn) outBtn.addEventListener('click', () => extrude(1));
  const inBtn = $('sc-in');
  if (inBtn) inBtn.addEventListener('click', () => extrude(-1));

  const wireBtn = $('wire');
  if (wireBtn) wireBtn.addEventListener('click', () => {
    wireOn = !wireOn;
    wireBtn.classList.toggle('on', wireOn);
    wireBtn.textContent = wireOn ? 'Сетка вкл' : 'Сетка';
    ensureWire();
    if (wireOn && wire) rebuildWire();
  });

  const undoBtn = $('sc-undo');
  if (undoBtn) undoBtn.addEventListener('click', () => {
    const h = hist.pop();
    if (!h) { syncInfo('отменять нечего'); return; }
    pos.set(h.pos); reg.set(h.reg); regionN = h.n;
    dead.set(h.dead); delTotal = h.del;
    if (h.index !== liveIndex) setIndex(h.index);
    nrm = null;
    paintMarks(); free = freeMask(topo, reg);
    flush(); syncInfo();
  });

  const clearBtn = $('sc-clear');
  if (clearBtn) clearBtn.addEventListener('click', () => {
    pushHist();
    reg.fill(0); regionN = 0;
    pos.set(pos0); dead.fill(0); delTotal = 0;
    setIndex(origIndex.slice());
    nrm = null;
    paintMarks(); flush(); syncInfo();
  });

  const saveBtn = $('sc-save');
  if (saveBtn) saveBtn.addEventListener('click', () => {
    // экспорт: если точки удаляли — сначала выбрасываем осиротевшие вершины,
    // чтобы в файле не осталось ни одной неиспользованной точки
    const cmp = compactMesh(src, liveIndex);
    const bytes = meshTo3mf(cmp.positions, cmp.index, {
      name: 'Кузов',
      meta: 'VW Tayron: кузов со сглаженным участком (печать 17 %)'
    });
    const name = 'Tayron_body_smooth.3mf';
    const res = downloadBytes(bytes, name);
    if (info) {
      info.innerHTML = `готово: <a href="${res.url}" download="${name}" style="color:var(--acc)">` +
        `скачать ${name}</a> · ${(bytes.length / 1048576).toFixed(1)} МБ` +
        (cmp.dropped ? ` · ${cmp.dropped} вершин без граней выкинуто` : '') +
        '. Если браузер не начал скачивание сам — нажмите ссылку.';
    }
  });

  syncInfo();

  return {
    get mode() { return !!tool; },
    get tool() { return tool; },
    get erasing() { return tool === 'erase'; },
    setTool,
    get regionCount() { return regionN; },
    get deletedCount() { return delTotal; },
    toggleWire() { wireOn = !wireOn; ensureWire(); },
    stats: () => ({
      vertices: vk, faces: liveIndex.length / 3,
      sourceVertices: count, welded: topo.welded,
      deleted: delTotal, facesNow: liveIndex.length / 3
    })
  };
}
