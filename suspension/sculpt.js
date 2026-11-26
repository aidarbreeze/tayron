/* ============================================================
   «Сглаживание участка» на сайте.

   Как это работает:
   1. Кистью (левая кнопка) закрашиваем участок на кузове — вершины попадают
      в область, на модели они подсвечиваются.
   2. Кнопка «Сгладить» прогоняет диффузию по рёбрам внутри области (кромка
      области и всё остальное не двигаются). Нажатий можно делать сколько
      угодно — каждое накапливается, «Отменить» откатывает шаг.
   3. Кнопка «Удалять точки» переключает кисть в ластик: закрашенные красным
      точки убираются, а на месте дырки поверхность зашивается треугольниками
      по её же кромке — плоскость сама соединяется с ближайшими оставшимися
      точками (см. meshops.deleteAndFill). Один штрих = один шаг «Отменить».
   4. «Скачать 3MF» отдаёт кузов файлом прямо из браузера.

   Сглаживание топологию не меняет: ни булевых операций, ни новых треугольников
   — только сдвиг вершин. Удаление убирает точки и перерезает грани, но новые
   точки не появляются: заплатка строится из уже существующих вершин, поэтому
   сетка остаётся замкнутой.

   Всё в единицах модели (1 ед. = 0.17 мм печати, U = 5.8824 ед./мм).
   ============================================================ */
import * as THREE from 'three';
import { meshTo3mf, downloadBytes } from './export3mf.js';
import {
  buildTopology, freeMask, smoothDiffuse, boundaryFalloff, paintBall,
  expandToVertices, deleteAndFill, compactMesh
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
  let liveIndex = Int32Array.from(geom.index.array);   // индексы для экспорта
  const topo = buildTopology(src, liveIndex);
  const { vk, pos0, canon } = topo;

  const pos = Float32Array.from(pos0);
  const reg = new Uint8Array(vk);
  const dead = new Uint8Array(vk);        // узлы, убранные ластиком
  let delTotal = 0;                       // сколько узлов удалено всего
  let free = new Uint8Array(vk);
  let regionN = 0;

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
      if (dead[v])     { CANON_COL[k] = 1.0;  CANON_COL[k + 1] = 0.28; CANON_COL[k + 2] = 0.30; }
      else if (reg[v]) { CANON_COL[k] = 0.34; CANON_COL[k + 1] = 1.0;  CANON_COL[k + 2] = 0.80; }
      else             { CANON_COL[k] = 1;    CANON_COL[k + 1] = 1;    CANON_COL[k + 2] = 1;    }
    }
    repaintColors();
  }

  /* ---------- координаты → геометрия ---------- */
  function flush() {
    expandToVertices(topo, pos, src);
    attr.needsUpdate = true;
    geom.computeVertexNormals();
    geom.computeBoundingSphere();
    geom.computeBoundingBox();
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
  }

  /* ---------- кисть ---------- */
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  let brushMM = 6;
  let amount = 25;          // «степень» в процентах: сколько итераций за нажатие
  let painting = false;
  let eraseOn = false;      // ластик вместо закраски
  let strokeSnap = null;    // снимок для «Отменить» — до начала штриха
  let strokeDel = 0;        // сколько удалено в текущем штрихе

  /** точка под курсором в локальных координатах кузова */
  function hit(clientX, clientY) {
    ndc.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObject(mesh, false);
    if (!hits.length) return null;
    return mesh.worldToLocal(hits[0].point.clone());
  }

  function stamp(clientX, clientY) {
    const p = hit(clientX, clientY);
    if (!p) return 0;
    const added = paintBall(topo, reg, [p.x, p.y, p.z], brushMM * U, pos, dead);
    if (added) { regionN += added; paintMarks(); syncInfo(); }
    return added;
  }

  /* ---------- ластик: пометить точки под кистью ---------- */
  function eraseStamp(clientX, clientY) {
    const p = hit(clientX, clientY);
    if (!p) return 0;
    const r2 = (brushMM * U) * (brushMM * U);
    let n = 0;
    for (let v = 0; v < vk; v++) {
      if (dead[v]) continue;
      const o = v * 3;
      const dx = pos[o] - p.x, dy = pos[o + 1] - p.y, dz = pos[o + 2] - p.z;
      if (dx * dx + dy * dy + dz * dz > r2) continue;
      dead[v] = 1; delTotal++; n++;
      if (reg[v]) { reg[v] = 0; regionN--; }
    }
    if (n) { strokeDel += n; paintMarks(); syncInfo(); }
    return n;
  }

  /* ---------- зашивка: убрать точки и закрыть дырку ---------- */
  function applyErase(n) {
    if (!n) return;
    const res = deleteAndFill(topo, liveIndex, pos, dead);
    setIndex(res.index);
    // зашивка могла убрать ещё несколько точек (кромка касалась себя) и снять
    // какие-то из закрашенных — пересчитываем по факту
    let regNow = 0, delNow = 0;
    for (let v = 0; v < vk; v++) { if (reg[v]) regNow++; if (dead[v]) delNow++; }
    regionN = regNow; delTotal = delNow;
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
    if (!regionN) { syncInfo('сначала закрасьте участок'); return; }
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
    flush();
    syncInfo();
  }

  /* ---------- интерфейс ---------- */
  const $ = (id) => document.getElementById(id);
  const btnMode  = $('sc-mode');
  const btnDel   = $('sc-del');
  const rows     = Array.from(document.querySelectorAll('.sc-row'));
  const hint     = $('sc-hint');
  const info     = $('sc-info');
  const HINT_SMOOTH = hint ? hint.innerHTML : '';
  const HINT_ERASE =
    'Левой кнопкой <b>сотрите</b> лишние точки — они подсветятся красным и уйдут по ' +
    'отпусканию кнопки, а дырка зашьётся треугольниками по своей кромке: поверхность ' +
    'сама соединится с ближайшими оставшимися точками. Правой — вращать камеру, ' +
    'колесом — приблизить. «Отменить» возвращает целый штрих.';
  let on = false;

  function syncInfo(note) {
    if (!info) return;
    if (note) { info.textContent = note; return; }
    const faces = liveIndex.length / 3;
    if (eraseOn) {
      info.textContent = `ластик: за один штрих удаляется всё под кистью ` +
        `(${brushMM} мм) · в сетке ${faces} граней`;
    } else if (regionN) {
      const pct = (100 * regionN / vk).toFixed(0);
      info.textContent = `выбрано ${regionN} вершин (${pct} %), ` +
        `за нажатие ${Math.max(20, Math.round(amount / 100 * MAXSTEPS))} итераций`;
    } else {
      info.textContent = 'область не выбрана';
    }
  }

  function setErase(v) {
    eraseOn = v;
    if (btnDel) {
      btnDel.classList.toggle('on', v);
      btnDel.textContent = v ? '■ Закончить удаление' : '⌫ Удалять точки';
    }
    const amtRow = $('sc-amtrow'), goBtn = $('sc-go');
    if (amtRow) amtRow.classList.toggle('hd', v);
    if (goBtn) goBtn.classList.toggle('hd', v);
    if (hint) hint.innerHTML = v ? HINT_ERASE : HINT_SMOOTH;
    if (v && !on) setMode(true);
    paintMarks();
    syncInfo();
  }

  function setMode(v) {
    on = v;
    if (btnMode) {
      btnMode.classList.toggle('on', on);
      btnMode.textContent = on ? '■ Закончить выбор' : '✎ Выбрать участок';
    }
    rows.forEach(r => r.classList.toggle('hd', !on));
    if (on) {
      const amtRow = $('sc-amtrow'), goBtn = $('sc-go');
      if (amtRow) amtRow.classList.toggle('hd', eraseOn);
      if (goBtn) goBtn.classList.toggle('hd', eraseOn);
    }
    if (hint) hint.classList.toggle('hd', !on);
    renderer.domElement.style.cursor = on ? (eraseOn ? 'cell' : 'crosshair') : '';
    const MB = THREE.MOUSE;
    if (on) {
      controls.mouseButtons = { LEFT: null, MIDDLE: MB.DOLLY, RIGHT: MB.ROTATE };
      controls.touches = { ONE: null, TWO: MB.DOLLY_ROTATE };
    } else {
      controls.mouseButtons = { LEFT: MB.ROTATE, MIDDLE: MB.DOLLY, RIGHT: MB.PAN };
      controls.touches = { ONE: MB.ROTATE, TWO: MB.DOLLY_PAN };
      if (eraseOn) setErase(false);
    }
  }

  const cvs = renderer.domElement;
  cvs.addEventListener('contextmenu', e => { if (on) e.preventDefault(); });
  cvs.addEventListener('pointerdown', e => {
    if (!on || e.button !== 0) return;
    painting = true;
    try { cvs.setPointerCapture(e.pointerId); } catch (_) {}
    if (eraseOn) { strokeDel = 0; strokeSnap = snapshot(); eraseStamp(e.clientX, e.clientY); }
    else stamp(e.clientX, e.clientY);
  });
  cvs.addEventListener('pointermove', e => {
    if (!on || !painting) return;
    if (eraseOn) eraseStamp(e.clientX, e.clientY);
    else stamp(e.clientX, e.clientY);
  });
  const stop = () => {
    if (!painting) return;
    painting = false;
    if (eraseOn && strokeDel) {
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

  if (btnMode) btnMode.addEventListener('click', () => setMode(!on));
  if (btnDel) btnDel.addEventListener('click', () => setErase(!eraseOn));

  const brushEl = $('sc-brush');
  if (brushEl) brushEl.addEventListener('input', () => {
    brushMM = +brushEl.value;
    $('sc-brushv').textContent = brushMM + ' мм';
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

  const undoBtn = $('sc-undo');
  if (undoBtn) undoBtn.addEventListener('click', () => {
    const h = hist.pop();
    if (!h) { syncInfo('отменять нечего'); return; }
    pos.set(h.pos); reg.set(h.reg); regionN = h.n;
    dead.set(h.dead); delTotal = h.del;
    if (h.index !== liveIndex) setIndex(h.index);
    paintMarks(); free = freeMask(topo, reg);
    flush(); syncInfo();
  });

  const clearBtn = $('sc-clear');
  if (clearBtn) clearBtn.addEventListener('click', () => {
    pushHist();
    reg.fill(0); regionN = 0;
    pos.set(pos0); paintMarks(); flush(); syncInfo();
  });

  const saveBtn = $('sc-save');
  if (saveBtn) saveBtn.addEventListener('click', () => {
    // экспорт: если точки удаляли — сначала выбрасываем осиротевшие вершины,
    // чтобы в файле не осталось ни одной неиспользованной точки
    const cmp = compactMesh(src, liveIndex);
    const positions = cmp.positions, index = cmp.index;
    const bytes = meshTo3mf(positions, index, {
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
    get mode() { return on; },
    get erasing() { return eraseOn; },
    setMode,
    setErase,
    get regionCount() { return regionN; },
    get deletedCount() { return delTotal; },
    stats: () => ({
      vertices: vk, faces: liveIndex.length / 3,
      sourceVertices: count, welded: topo.welded,
      deleted: delTotal, facesNow: liveIndex.length / 3
    })
  };
}
