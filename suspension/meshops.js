/* ============================================================
   Геометрия для «сглаживания участка» — без three.js, чистые массивы.

   Сетка из GLB приходит с разрезанными вершинами (одна точка может лежать в
   нескольких вершинах). Для сглаживания нужен граф соседей: склеиваем
   совпадающие позиции в один узел (canon) и строим связность по рёбрам.
   Склейка проверяется по кратности рёбер: если после неё сетка перестала быть
   замкнутой (где-то две поверхности соприкасаются в точке), склейка
   отменяется и работаем по вершинам файла как есть.

   Сглаживание — диффузия по рёбрам: p ← p + λ(среднее соседей − p), кромка
   области закреплена. Топология при этом не меняется вообще: ни одного нового
   треугольника, ни одной удалённой вершины.
   ============================================================ */

/** Ровно два треугольника на каждое ребро? (замкнутая сетка) */
function edgesBalanced(tris) {
  const keys = new Float64Array(tris.length);
  let k = 0;
  for (let i = 0; i < tris.length; i += 3) {
    const a = tris[i], b = tris[i + 1], c = tris[i + 2];
    keys[k++] = Math.min(a, b) * 1e7 + Math.max(a, b);
    keys[k++] = Math.min(b, c) * 1e7 + Math.max(b, c);
    keys[k++] = Math.min(c, a) * 1e7 + Math.max(c, a);
  }
  keys.sort();
  for (let i = 0; i < keys.length; i++) {
    let n = 1;
    while (i + 1 < keys.length && keys[i + 1] === keys[i]) { n++; i++; }
    if (n !== 2) return false;
  }
  return true;
}

/** Топология: узлы (canon), треугольники в узлах, соседи в виде CSR. */
export function buildTopology(position, index, q = 1e-3) {
  const count = position.length / 3;
  let canon = new Int32Array(count);
  const map = new Map();
  const vidx = [];
  let vk = 0;
  for (let i = 0; i < count; i++) {
    const x = position[i * 3], y = position[i * 3 + 1], z = position[i * 3 + 2];
    const key = Math.round(x / q) + '|' + Math.round(y / q) + '|' + Math.round(z / q);
    let c = map.get(key);
    if (c === undefined) { c = vk++; map.set(key, c); vidx.push(x, y, z); }
    canon[i] = c;
  }
  let tris = new Int32Array(index.length);
  for (let i = 0; i < index.length; i++) tris[i] = canon[index[i]];
  let pos0 = new Float32Array(vidx);
  let welded = vk < count;

  if (welded && !edgesBalanced(tris)) {          // склейка ломает замкнутость
    welded = false;
    vk = count;
    canon = new Int32Array(count);
    for (let i = 0; i < count; i++) canon[i] = i;
    tris = new Int32Array(index.length);
    for (let i = 0; i < index.length; i++) tris[i] = index[i];
    pos0 = Float32Array.from(position);
  }

  /* --- соседи: пары (a,b) → сортировка → CSR --- */
  const pairs = new Int32Array(tris.length * 4);   // 6 направленных рёбер на треугольник
  let pn = 0;
  const add = (a, b) => { if (a !== b) { pairs[pn++] = a; pairs[pn++] = b; } };
  for (let i = 0; i < tris.length; i += 3) {
    const a = tris[i], b = tris[i + 1], c = tris[i + 2];
    add(a, b); add(b, a);
    add(b, c); add(c, b);
    add(c, a); add(a, c);
  }
  const keys = new Float64Array(pn / 2);
  for (let i = 0, k = 0; i < pn; i += 2, k++) keys[k] = pairs[i] * vk + pairs[i + 1];
  keys.sort();
  const off = new Int32Array(vk + 1);
  const tos = [];
  for (let k = 0; k < keys.length; k++) {
    if (k && keys[k] === keys[k - 1]) continue;
    const from = Math.floor(keys[k] / vk);
    off[from + 1]++;
    tos.push(from, keys[k] % vk);
  }
  for (let i = 0; i < vk; i++) off[i + 1] += off[i];
  const nbr = new Int32Array(tos.length / 2);
  const fill = off.slice(0, vk);
  for (let i = 0; i < tos.length; i += 2) nbr[fill[tos[i]]++] = tos[i + 1];

  /* --- представитель узла: первая вершина файла с этой позицией ---
     нужен, чтобы вернуть треугольники в индексы файла (экспорт 3MF) */
  const rep = new Int32Array(vk).fill(-1);
  for (let i = 0; i < count; i++) {
    const c = canon[i];
    if (rep[c] < 0) rep[c] = i;
  }

  return { count, vk, pos0, canon, tris, off, nbr, welded, rep };
}

/** Внутренность области: узлы, у всех соседей которых та же область. */
export function freeMask(topo, reg) {
  const { vk, off, nbr } = topo;
  const free = new Uint8Array(vk);
  for (let v = 0; v < vk; v++) {
    if (!reg[v]) continue;
    let ok = 1;
    for (let k = off[v]; k < off[v + 1]; k++) if (!reg[nbr[k]]) { ok = 0; break; }
    free[v] = ok;
  }
  return free;
}

/**
 * Сглаживание области: steps шагов диффузии по рёбрам.
 * from — откуда считать (по умолчанию исходные координаты, pos0);
 * чтобы нажатия кнопки накапливались, передают текущие координаты.
 */
export function smoothDiffuse(topo, free, steps, lambda = 0.5, from = null) {
  const { vk, pos0, off, nbr } = topo;
  const pos = Float32Array.from(from || pos0);
  const buf = new Float32Array(vk * 3);
  for (let s = 0; s < steps; s++) {
    buf.set(pos);
    for (let v = 0; v < vk; v++) {
      if (!free[v]) continue;
      const a = off[v], b = off[v + 1], n = b - a;
      if (!n) continue;
      let sx = 0, sy = 0, sz = 0;
      for (let k = a; k < b; k++) {
        const w = nbr[k] * 3;
        sx += buf[w]; sy += buf[w + 1]; sz += buf[w + 2];
      }
      const o = v * 3;
      pos[o]     += lambda * (sx / n - buf[o]);
      pos[o + 1] += lambda * (sy / n - buf[o + 1]);
      pos[o + 2] += lambda * (sz / n - buf[o + 2]);
    }
  }
  return pos;
}

/**
 * Плавный вход в область: вес 0 на кромке области, 1 внутри (за ringMax шагов).
 * Без этого сглаженные вершины у самой кромки «отъезжают» на 2-3 мм от
 * неподвижных соседей и получаются длинные тонкие треугольники.
 */
export function boundaryFalloff(topo, reg, ringMax = 4) {
  const { vk, off, nbr } = topo;
  const w = new Float32Array(vk);
  const dist = new Int32Array(vk).fill(-1);
  const queue = new Int32Array(vk);
  let head = 0, tail = 0;
  for (let v = 0; v < vk; v++) {
    if (!reg[v]) continue;
    for (let k = off[v]; k < off[v + 1]; k++) if (!reg[nbr[k]]) { dist[v] = 0; queue[tail++] = v; break; }
  }
  while (head < tail) {
    const v = queue[head++];
    if (dist[v] >= ringMax) continue;
    for (let k = off[v]; k < off[v + 1]; k++) {
      const u = nbr[k];
      if (reg[u] && dist[u] < 0) { dist[u] = dist[v] + 1; queue[tail++] = u; }
    }
  }
  for (let v = 0; v < vk; v++) {
    if (!reg[v]) continue;
    const t = dist[v] < 0 ? 1 : Math.min(1, dist[v] / ringMax);
    w[v] = t * t * (3 - 2 * t);            // smoothstep
  }
  return w;
}

/** Заливка кистью: узлы в шаре радиуса r вокруг точки p. */
export function paintBall(topo, reg, p, r, pos, skip) {
  const { vk } = topo;
  const P = pos || topo.pos0;
  const r2 = r * r;
  let added = 0;
  for (let v = 0; v < vk; v++) {
    if (reg[v]) continue;
    if (skip && skip[v]) continue;          // удалённые ластиком точки не красим
    const o = v * 3;
    const dx = P[o] - p[0], dy = P[o + 1] - p[1], dz = P[o + 2] - p[2];
    if (dx * dx + dy * dy + dz * dz <= r2) { reg[v] = 1; added++; }
  }
  return added;
}

/** Перенести координаты узлов в массив вершин геометрии (по canon). */
export function expandToVertices(topo, pos, out) {
  const { count, canon } = topo;
  for (let i = 0; i < count; i++) {
    const c = canon[i] * 3, o = i * 3;
    out[o] = pos[c]; out[o + 1] = pos[c + 1]; out[o + 2] = pos[c + 2];
  }
  return out;
}

/* ============================================================
   Удаление вершин с автоматической зашивкой дыр.

   deleted[c] = 1 — узел c убираем. Грани, задевающие такие узлы, выбрасываются;
   на их месте остаются дырки. Каждую дырку зашиваем треугольниками по её же
   кромке: новые точки не появляются, поверхность сама соединяется с ближайшими
   оставшимися вершинами (ими и является кромка дырки).

   Заливка — «отрезание ушей» в плоскости кромки: дырка проецируется на свою
   плоскость, треугольники режутся по контуру, поэтому заплатка получается
   плоской, а не седлом.
   ============================================================ */

/** Кромка дыр: направленные рёбра без парного обратного + вершины-«пинчи». */
function boundaryEdges(keep, vk) {
  const key = (a, b) => a * vk + b;
  const dir = new Set();
  for (let i = 0; i < keep.length; i += 3) {
    const a = keep[i], b = keep[i + 1], c = keep[i + 2];
    dir.add(key(a, b)); dir.add(key(b, c)); dir.add(key(c, a));
  }
  const next = new Map();                 // a → b для кромочных рёбер
  const outDeg = new Map(), inDeg = new Map();
  for (const k of dir) {
    const a = Math.floor(k / vk), b = k - a * vk;
    if (dir.has(key(b, a))) continue;
    next.set(a, b);
    outDeg.set(a, (outDeg.get(a) || 0) + 1);
    inDeg.set(b, (inDeg.get(b) || 0) + 1);
  }
  /* вершина, через которую кромка проходит дважды: дыра касается себя.
     Такую дыру не зашить — вершину тоже придётся убрать. */
  const pinch = [];
  const verts = new Set([...outDeg.keys(), ...inDeg.keys()]);
  for (const v of verts) if (outDeg.get(v) !== 1 || inDeg.get(v) !== 1) pinch.push(v);
  return { next, pinch };
}

/** Замкнутые контуры кромки (когда все степени 1 — это гарантированно циклы). */
function boundaryLoops(next) {
  const loops = [];
  const seen = new Set();
  for (const start of next.keys()) {
    if (seen.has(start)) continue;
    const loop = [];
    let v = start, closed = false;
    while (v !== undefined && !seen.has(v)) {
      seen.add(v); loop.push(v);
      v = next.get(v);
      if (v === start) { closed = true; break; }
    }
    if (closed && loop.length >= 3) loops.push(loop);
  }
  return loops;
}

/** Проекция кромки на её плоскость: базис (u, v) и 2D-контур. */
function flattenLoop(loop, pos) {
  const n = loop.length;
  let cx = 0, cy = 0, cz = 0;
  for (const id of loop) { cx += pos[id * 3]; cy += pos[id * 3 + 1]; cz += pos[id * 3 + 2]; }
  cx /= n; cy /= n; cz /= n;
  // нормаль Ньюэлла
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < n; i++) {
    const a = loop[i] * 3, b = loop[(i + 1) % n] * 3;
    const ax = pos[a] - cx, ay = pos[a + 1] - cy, az = pos[a + 2] - cz;
    const bx = pos[b] - cx, by = pos[b + 1] - cy, bz = pos[b + 2] - cz;
    nx += ay * bz - az * by;
    ny += az * bx - ax * bz;
    nz += ax * by - ay * bx;
  }
  const ln = Math.hypot(nx, ny, nz) || 1;
  nx /= ln; ny /= ln; nz /= ln;
  // произвольный вектор, не параллельный нормали
  let ax = Math.abs(nx) < 0.9 ? 1 : 0, ay = Math.abs(nx) < 0.9 ? 0 : 1, az = 0;
  let ux = ay * nz - az * ny, uy = az * nx - ax * nz, uz = ax * ny - ay * nx;
  const lu = Math.hypot(ux, uy, uz) || 1; ux /= lu; uy /= lu; uz /= lu;
  const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
  const p2 = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    const o = loop[i] * 3;
    const px = pos[o] - cx, py = pos[o + 1] - cy, pz = pos[o + 2] - cz;
    p2[i * 2] = px * ux + py * uy + pz * uz;
    p2[i * 2 + 1] = px * vx + py * vy + pz * vz;
  }
  return { p2, normal: [nx, ny, nz] };
}

/** Площадь со знаком (2D). */
function signedArea(p2) {
  let s = 0;
  const n = p2.length / 2;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    s += p2[i * 2] * p2[j * 2 + 1] - p2[j * 2] * p2[i * 2 + 1];
  }
  return s * 0.5;
}

function pointInTri(px, py, ax, ay, bx, by, cx, cy) {
  const d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by);
  const d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy);
  const d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay);
  const neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
  const pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
  return !(neg && pos);
}

/**
 * Отрезание ушей: контур → треугольники (индексы внутри контура).
 * forbidden(i, j) — диагональ, которую нельзя ставить: такое ребро уже есть в
 * сетке (перемычка между точками кромки), и лишний треугольник на нём сделал бы
 * сетку незамкнутой. Такие уши пропускаем, пока есть другие.
 */
function earClip(p2, forbidden = null) {
  const n = p2.length / 2;
  if (n < 3) return [];
  const idx = [];
  for (let i = 0; i < n; i++) idx.push(i);
  const ccw = signedArea(p2) > 0;
  const tris = [];
  const cross3 = (i0, i1, i2) =>
    (p2[i1 * 2] - p2[i0 * 2]) * (p2[i2 * 2 + 1] - p2[i0 * 2 + 1]) -
    (p2[i1 * 2 + 1] - p2[i0 * 2 + 1]) * (p2[i2 * 2] - p2[i0 * 2]);
  let guard = 0;
  while (idx.length > 3 && guard++ < n * n + 64) {
    let good = -1, goodA = -1;             // ухо без запрещённой диагонали
    let bad = -1, badA = -1;               // ухо с запрещённой (крайний случай)
    for (let i = 0; i < idx.length; i++) {
      const i0 = idx[(i + idx.length - 1) % idx.length], i1 = idx[i], i2 = idx[(i + 1) % idx.length];
      const cr = cross3(i0, i1, i2);
      if (ccw ? cr <= 1e-12 : cr >= -1e-12) continue;      // не выпуклое ухо
      let inside = false;
      for (const j of idx) {
        if (j === i0 || j === i1 || j === i2) continue;
        if (pointInTri(p2[j * 2], p2[j * 2 + 1],
              p2[i0 * 2], p2[i0 * 2 + 1], p2[i1 * 2], p2[i1 * 2 + 1], p2[i2 * 2], p2[i2 * 2 + 1])) {
          inside = true; break;
        }
      }
      if (inside) continue;
      const area = Math.abs(cr);
      if (forbidden && forbidden(i0, i2)) {
        if (area > badA) { badA = area; bad = i; }
      } else if (area > goodA) {
        goodA = area; good = i;
      }
    }
    let pick = good >= 0 ? good : bad;
    if (pick < 0) {                        // вырожденный контур — режем самое «толстое» ухо
      let best = 0, bestA = -1;
      for (let i = 0; i < idx.length; i++) {
        const i0 = idx[(i + idx.length - 1) % idx.length], i1 = idx[i], i2 = idx[(i + 1) % idx.length];
        const a = Math.abs(cross3(i0, i1, i2));
        if (a > bestA) { bestA = a; best = i; }
      }
      pick = best;
    }
    const i0 = idx[(pick + idx.length - 1) % idx.length], i1 = idx[pick], i2 = idx[(pick + 1) % idx.length];
    tris.push(i0, i1, i2);
    idx.splice(pick, 1);
  }
  if (idx.length === 3) tris.push(idx[0], idx[1], idx[2]);
  return tris;
}

/**
 * Триангуляция контура минимальной суммой диагоналей (динамика по подотрезкам).
 * Работает прямо в 3D — проецировать контур не нужно, поэтому узкие «перешейки»,
 * на которых отрезание ушей застревает, разбираются корректно.
 * forbidden(i, j) — диагональ, которую ставить нельзя (ребро уже есть в сетке).
 * Возвращает индексы внутри контура или null, если контур нельзя закрыть.
 */
function dpTriangulate(loop, pos, forbidden) {
  const n = loop.length;
  if (n < 3) return null;
  if (n > 1000) return null;                    // совсем край: зовём отрезание ушей
  const d = (i, j) => {
    const a = loop[i] * 3, b = loop[j] * 3;
    return Math.hypot(pos[b] - pos[a], pos[b + 1] - pos[a + 1], pos[b + 2] - pos[a + 2]);
  };
  const INF = 1e12;
  const len = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    len[i * n + i] = 0;
    for (let j = i + 1; j < n; j++) {
      const edge = (j === i + 1) || (i === 0 && j === n - 1);   // сторона контура
      len[i * n + j] = len[j * n + i] =
        (!edge && forbidden && forbidden(i, j)) ? INF : d(i, j);
    }
  }
  const dp = new Float64Array(n * n).fill(INF);
  const cut = new Int32Array(n * n).fill(-1);
  for (let i = 0; i + 1 < n; i++) dp[i * n + i + 1] = 0;
  for (let span = 2; span < n; span++) {
    for (let i = 0; i + span < n; i++) {
      const j = i + span;
      let best = INF, bestK = -1;
      for (let k = i + 1; k < j; k++) {
        const v = dp[i * n + k] + dp[k * n + j] + len[i * n + k] + len[k * n + j];
        if (v < best) { best = v; bestK = k; }
      }
      dp[i * n + j] = best; cut[i * n + j] = bestK;
    }
  }
  if (!isFinite(dp[n - 1]) || dp[0 * n + (n - 1)] >= INF) return null;
  const tris = [];
  const stack = [[0, n - 1]];
  while (stack.length) {
    const [i, j] = stack.pop();
    const k = cut[i * n + j];
    if (k < 0) continue;
    tris.push(i, k, j);
    if (k - i > 1) stack.push([i, k]);
    if (j - k > 1) stack.push([k, j]);
  }
  return tris.length ? tris : null;
}

/**
 * Убрать помеченные узлы и зашить получившиеся дырки.
 * Возвращает новый список граней (индексы вершин файла), число дырок и
 * добавленных треугольников.
 */
export function deleteAndFill(topo, index, pos, deleted, report) {
  const { canon, vk, rep } = topo;

  /* Грани, которые остаются, и кромка. Если кромка проходит через вершину
     дважды (дыра касается себя — тонкая перемычка), такую вершину тоже убираем
     и считаем заново: иначе дыру не зашить. */
  let keep = [], removed = 0, edges = null, extra = 0;
  for (let pass = 0; pass < 200; pass++) {
    keep = []; removed = 0;
    for (let i = 0; i < index.length; i += 3) {
      const a = canon[index[i]], b = canon[index[i + 1]], c = canon[index[i + 2]];
      if (a === b || b === c || c === a) { removed++; continue; }
      if (deleted[a] || deleted[b] || deleted[c]) { removed++; continue; }
      keep.push(a, b, c);
    }
    edges = boundaryEdges(keep, vk);
    if (!edges.pinch.length) break;
    let grew = 0;
    for (const v of edges.pinch) if (!deleted[v]) { deleted[v] = 1; grew++; }
    extra += grew;
    if (!grew) break;                    // убирать больше нечего — оставляем как есть
  }
  const loops = boundaryLoops(edges.next);

  /* Ориентация заплатки: у замкнутой сетки кромка дыры направлена так, что
     заплатка обязана идти по ней в обратную сторону. Проверяем честно — по
     совпадению направленных рёбер с уже существующими (сетка может прийти
     с несогласованным обходом граней). */
  const vk2 = vk * 1e6;                       // vk ≤ 60 000: ключи точны в double
  const dirSet = new Set();
  for (let i = 0; i < keep.length; i += 3) {
    const a = keep[i], b = keep[i + 1], c = keep[i + 2];
    dirSet.add(a * vk2 + b); dirSet.add(b * vk2 + c); dirSet.add(c * vk2 + a);
  }
  const patch = [];
  let added = 0;
  for (const loop of loops) {
    const forb = (i, j) => dirSet.has(loop[i] * vk2 + loop[j]) ||
                           dirSet.has(loop[j] * vk2 + loop[i]);
    const dp = dpTriangulate(loop, pos, forb);
    const tri = dp || earClip(flattenLoop(loop, pos).p2, forb);
    if (report) report.push({ loop: loop.length, dp: dp ? dp.length / 3 : 0, tris: tri.length / 3 });
    if (!tri.length) continue;
    // два варианта обхода, берём тот, что не дублирует уже занятые рёбра
    const score = (flip) => {
      let dup = 0;
      for (let i = 0; i < tri.length; i += 3) {
        const a = loop[tri[i]], b = loop[tri[i + 1]], c = loop[tri[i + 2]];
        const t = flip ? [a, c, b] : [a, b, c];
        if (dirSet.has(t[0] * vk2 + t[1])) dup++;
        if (dirSet.has(t[1] * vk2 + t[2])) dup++;
        if (dirSet.has(t[2] * vk2 + t[0])) dup++;
      }
      return dup;
    };
    const flip = score(true) < score(false);
    for (let i = 0; i < tri.length; i += 3) {
      const a = loop[tri[i]], b = loop[tri[i + 1]], c = loop[tri[i + 2]];
      const t = flip ? [a, c, b] : [a, b, c];
      if (t[0] === t[1] || t[1] === t[2] || t[2] === t[0]) continue;
      patch.push(t[0], t[1], t[2]);
      dirSet.add(t[0] * vk2 + t[1]);
      dirSet.add(t[1] * vk2 + t[2]);
      dirSet.add(t[2] * vk2 + t[0]);
      added++;
    }
  }
  const out = new Int32Array((keep.length + patch.length));
  for (let i = 0; i < keep.length; i++) out[i] = rep[keep[i]];
  for (let i = 0; i < patch.length; i++) out[keep.length + i] = rep[patch[i]];
  return { index: out, removedFaces: removed, holes: loops.length, addedFaces: added, extraDeleted: extra };
}

/* ============================================================
   Лепка: нормали узлов и сдвиги вдоль них.

   Все инструменты работают одинаково — двигают существующие вершины вдоль их
   нормалей. Ни одна операция не добавляет и не удаляет геометрию, поэтому
   после любого сдвига сетка остаётся той же замкнутой сеткой.
   ============================================================ */

/** Нормали узлов: сумма нормалей граней (по площади), нормированная. */
export function vertexNormals(topo, index, pos) {
  const { canon, vk } = topo;
  const n = new Float32Array(vk * 3);
  for (let i = 0; i < index.length; i += 3) {
    const a = canon[index[i]] * 3, b = canon[index[i + 1]] * 3, c = canon[index[i + 2]] * 3;
    const e1x = pos[b] - pos[a], e1y = pos[b + 1] - pos[a + 1], e1z = pos[b + 2] - pos[a + 2];
    const e2x = pos[c] - pos[a], e2y = pos[c + 1] - pos[a + 1], e2z = pos[c + 2] - pos[a + 2];
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    n[a] += nx; n[a + 1] += ny; n[a + 2] += nz;
    n[b] += nx; n[b + 1] += ny; n[b + 2] += nz;
    n[c] += nx; n[c + 1] += ny; n[c + 2] += nz;
  }
  for (let v = 0; v < vk; v++) {
    const o = v * 3;
    const len = Math.hypot(n[o], n[o + 1], n[o + 2]);
    if (len > 1e-12) { n[o] /= len; n[o + 1] /= len; n[o + 2] /= len; }
    else { n[o] = 0; n[o + 1] = 0; n[o + 2] = 1; }
  }
  return n;
}

const smooth01 = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));

/**
 * Сдвиг шара: вершины в радиусе r уезжают вдоль своих нормалей на
 * amount * smoothstep(1 − d/r). Знак amount — вдавить (минус) или вытянуть.
 * only — маска узлов, которые можно двигать (например, без удалённых).
 */
export function pushBall(topo, pos, center, r, amount, normals, only) {
  const { vk } = topo;
  let moved = 0, maxd = 0;
  const r2 = r * r;
  for (let v = 0; v < vk; v++) {
    if (only && !only[v]) continue;
    const o = v * 3;
    const dx = pos[o] - center[0], dy = pos[o + 1] - center[1], dz = pos[o + 2] - center[2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 >= r2) continue;
    const w = smooth01(1 - Math.sqrt(d2) / r);
    const s = amount * w;
    if (s === 0) continue;
    pos[o] += normals[o] * s;
    pos[o + 1] += normals[o + 1] * s;
    pos[o + 2] += normals[o + 2] * s;
    moved++; if (Math.abs(s) > maxd) maxd = Math.abs(s);
  }
  return { moved, maxd };
}

/**
 * Выдавливание закрашенной области: все узлы области уезжают вдоль нормалей на
 * amount, у кромки шаг гасится плавным входом (ширина soft в узлах).
 */
export function offsetRegion(topo, pos, reg, amount, normals, soft = 4) {
  const { vk } = topo;
  const w = boundaryFalloff(topo, reg, soft);
  let moved = 0, maxd = 0;
  for (let v = 0; v < vk; v++) {
    if (!reg[v] || w[v] <= 0) continue;
    const o = v * 3;
    const s = amount * w[v];
    pos[o] += normals[o] * s;
    pos[o + 1] += normals[o + 1] * s;
    pos[o + 2] += normals[o + 2] * s;
    moved++; if (Math.abs(s) > maxd) maxd = Math.abs(s);
  }
  return { moved, maxd };
}

/**
 * Отпечаток фигуры в поверхности.
 * shape: { kind: 'sphere'|'box'|'cylinder', center, n (нормаль поверхности),
 *          size: {r, h, a, b} в единицах модели }
 * Вершины внутри фигуры вдавливаются вдоль своей нормали пропорционально
 * глубине погружения — поверхность принимает форму фигуры, оставаясь целой.
 * amount > 0 — вдавить, < 0 — вытянуть (фигура «вытаскивает» купол).
 */
export function imprintShape(topo, pos, shape, amount, normals) {
  const { vk } = topo;
  const c = shape.center;
  // локальная ось фигуры (вдоль нормали поверхности), и два вектора в плоскости
  let nx = shape.n[0], ny = shape.n[1], nz = shape.n[2];
  const ln = Math.hypot(nx, ny, nz) || 1; nx /= ln; ny /= ln; nz /= ln;
  let tx = Math.abs(nx) < 0.9 ? 1 : 0, ty = Math.abs(nx) < 0.9 ? 0 : 1, tz = 0;
  let ux = ty * nz - tz * ny, uy = tz * nx - tx * nz, uz = tx * ny - ty * nx;
  const lu = Math.hypot(ux, uy, uz) || 1; ux /= lu; uy /= lu; uz /= lu;
  const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;

  const R = shape.size.r || 1, H = shape.size.h || R;
  const A = shape.size.a || R, B = shape.size.b || R;
  let moved = 0, maxd = 0;
  for (let v = 0; v < vk; v++) {
    const o = v * 3;
    const px = pos[o] - c[0], py = pos[o + 1] - c[1], pz = pos[o + 2] - c[2];
    const u = px * ux + py * uy + pz * uz;
    const w = px * vx + py * vy + pz * vz;
    const h = px * nx + py * ny + pz * nz;          // вдоль нормали поверхности
    let sdf;
    if (shape.kind === 'box') {
      sdf = Math.max(Math.abs(u) - A, Math.abs(w) - B, Math.abs(h) - H);
    } else if (shape.kind === 'cylinder') {
      sdf = Math.max(Math.hypot(u, w) - R, Math.abs(h) - H);
    } else {                                         // сфера
      sdf = Math.hypot(u, w, h) - R;
    }
    if (sdf >= 0) continue;                          // вне фигуры
    const t = Math.min(1, -sdf / H);                 // чем глубже, тем сильнее
    const s = -amount * smooth01(t);                 // amount > 0 — вдавить
    if (s === 0) continue;
    pos[o] += normals[o] * s;
    pos[o + 1] += normals[o + 1] * s;
    pos[o + 2] += normals[o + 2] * s;
    moved++; if (Math.abs(s) > maxd) maxd = Math.abs(s);
  }
  return { moved, maxd };
}

/**
 * Выровнять закрашенную область в плоскость.
 * Плоскость — по самой области: центроид + средняя нормаль граней, целиком
 * лежащих в области (у решётки это даёт «панель» нужного наклона).
 * amount 0…1 — насколько вытянуть (1 = точно в плоскость), у кромки плавный вход.
 */
export function flattenRegion(topo, pos, reg, index, amount = 1, soft = 3) {
  const { vk, canon } = topo;
  let cx = 0, cy = 0, cz = 0, cnt = 0;
  for (let v = 0; v < vk; v++) {
    if (!reg[v]) continue;
    cx += pos[v * 3]; cy += pos[v * 3 + 1]; cz += pos[v * 3 + 2]; cnt++;
  }
  if (!cnt) return { moved: 0, maxd: 0, normal: null };
  cx /= cnt; cy /= cnt; cz /= cnt;

  // нормаль — по площади граней, у которых все три узла в области
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < index.length; i += 3) {
    const a = canon[index[i]], b = canon[index[i + 1]], c = canon[index[i + 2]];
    if (!reg[a] || !reg[b] || !reg[c]) continue;
    const ao = a * 3, bo = b * 3, co = c * 3;
    const e1x = pos[bo] - pos[ao], e1y = pos[bo + 1] - pos[ao + 1], e1z = pos[bo + 2] - pos[ao + 2];
    const e2x = pos[co] - pos[ao], e2y = pos[co + 1] - pos[ao + 1], e2z = pos[co + 2] - pos[ao + 2];
    nx += e1y * e2z - e1z * e2y;
    ny += e1z * e2x - e1x * e2z;
    nz += e1x * e2y - e1y * e2x;
  }
  const ln = Math.hypot(nx, ny, nz);
  if (ln < 1e-12) {                       // область слишком изрезана — по нормалям узлов
    const vn = vertexNormals(topo, index, pos);
    for (let v = 0; v < vk; v++) {
      if (!reg[v]) continue;
      nx += vn[v * 3]; ny += vn[v * 3 + 1]; nz += vn[v * 3 + 2];
    }
    const l2 = Math.hypot(nx, ny, nz);
    if (l2 < 1e-12) return { moved: 0, maxd: 0, normal: null };
    nx /= l2; ny /= l2; nz /= l2;
  } else { nx /= ln; ny /= ln; nz /= ln; }

  const w = boundaryFalloff(topo, reg, soft);
  let moved = 0, maxd = 0;
  for (let v = 0; v < vk; v++) {
    if (!reg[v] || w[v] <= 0) continue;
    const o = v * 3;
    const d = (pos[o] - cx) * nx + (pos[o + 1] - cy) * ny + (pos[o + 2] - cz) * nz;
    const s = -d * amount * w[v];
    if (s === 0) continue;
    pos[o] += nx * s; pos[o + 1] += ny * s; pos[o + 2] += nz * s;
    moved++; if (Math.abs(s) > maxd) maxd = Math.abs(s);
  }
  return { moved, maxd, normal: [nx, ny, nz] };
}

/**
 * Обострение (обратное сглаживанию): узел уходит от среднего своих соседей —
 * рельеф подчёркивается, края становятся резче. Несколько итераций подряд.
 */
export function sharpenRegion(topo, pos, reg, index, k = 0.5, iters = 3, soft = 3) {
  const { vk, off, nbr } = topo;
  const w = boundaryFalloff(topo, reg, soft);
  const buf = new Float32Array(vk * 3);
  let moved = 0, maxd = 0;
  for (let it = 0; it < iters; it++) {
    buf.set(pos);
    for (let v = 0; v < vk; v++) {
      if (!reg[v] || w[v] <= 0) continue;
      const a = off[v], b = off[v + 1], n = b - a;
      if (!n) continue;
      let sx = 0, sy = 0, sz = 0;
      for (let j = a; j < b; j++) {
        const u = nbr[j] * 3;
        sx += buf[u]; sy += buf[u + 1]; sz += buf[u + 2];
      }
      const o = v * 3;
      const dx = buf[o] - sx / n, dy = buf[o + 1] - sy / n, dz = buf[o + 2] - sz / n;
      const lim = 0.5;                    // не даём улететь
      const s = k * w[v];
      const mx = dx * s, my = dy * s, mz = dz * s;
      const len = Math.hypot(mx, my, mz);
      const f = len > lim ? lim / len : 1;
      pos[o] += mx * f; pos[o + 1] += my * f; pos[o + 2] += mz * f;
      moved++; const dd = len * f; if (dd > maxd) maxd = dd;
    }
  }
  return { moved, maxd };
}

/**
 * Симметрия: переносит смещения узлов на их отражения по плоскости y = 0.
 * before — координаты до правки. Пары ищутся по сетке координат, чтобы
 * работать и на неидеально симметричной сетке.
 */
export function mirrorDeltas(topo, pos, before, tol = 2.5) {
  const { vk } = topo;
  const q = Math.max(tol, 1.0);
  const key = (x, y, z) => Math.round(x / q) + '|' + Math.round(y / q) + '|' + Math.round(z / q);
  const grid = new Map();
  for (let v = 0; v < vk; v++) {
    const o = v * 3;
    const k = key(before[o], before[o + 1], before[o + 2]);
    let arr = grid.get(k);
    if (!arr) grid.set(k, arr = []);
    arr.push(v);
  }
  const deltas = new Float32Array(vk * 3);
  const touched = new Uint8Array(vk);
  let moved = 0;
  for (let v = 0; v < vk; v++) {
    const o = v * 3;
    const dx = pos[o] - before[o], dy = pos[o + 1] - before[o + 1], dz = pos[o + 2] - before[o + 2];
    if (Math.abs(dx) + Math.abs(dy) + Math.abs(dz) < 1e-6) continue;
    touched[v] = 1;
    deltas[o] = dx; deltas[o + 1] = -dy; deltas[o + 2] = dz;
  }
  for (let v = 0; v < vk; v++) {
    if (!touched[v]) continue;
    const o = v * 3;
    // отражение ищем вокруг (x, -y, z)
    const mx = before[o], my = -before[o + 1], mz = before[o + 2];
    let best = -1, bd = tol * tol;
    for (let ix = -1; ix <= 1; ix++) for (let iy = -1; iy <= 1; iy++) for (let iz = -1; iz <= 1; iz++) {
      const k = key(mx + ix * q, my + iy * q, mz + iz * q);
      const arr = grid.get(k);
      if (!arr) continue;
      for (const u of arr) {
        const uo = u * 3;
        const d2 = (before[uo] - mx) ** 2 + (before[uo + 1] - my) ** 2 + (before[uo + 2] - mz) ** 2;
        if (d2 < bd) { bd = d2; best = u; }
      }
    }
    if (best < 0 || best === v) continue;
    if (touched[best]) continue;          // эта точка уже двигалась сама — не дублируем
    const bo = best * 3;
    pos[bo] += deltas[o];
    pos[bo + 1] += deltas[o + 1];
    pos[bo + 2] += deltas[o + 2];
    touched[best] = 1;
    moved++;
  }
  return { moved };
}

/**
 * Проверка сетки — то, что смотрят перед печатью:
 * замкнутость (у каждого ребра ровно две грани), согласованность обхода,
 * вырожденные треугольники, объём, габариты, длины рёбер.
 */
export function checkMesh(topo, index, pos, U = 5.8824) {
  const nf = index.length / 3;
  const dir = new Map();
  let degen = 0, volume6 = 0;
  for (let i = 0; i < index.length; i += 3) {
    const a = index[i], b = index[i + 1], c = index[i + 2];
    const ao = a * 3, bo = b * 3, co = c * 3;
    const ax = pos[ao], ay = pos[ao + 1], az = pos[ao + 2];
    const bx = pos[bo], by = pos[bo + 1], bz = pos[bo + 2];
    const cx = pos[co], cy = pos[co + 1], cz = pos[co + 2];
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    if (Math.hypot(nx, ny, nz) * 0.5 < 1e-9) degen++;
    volume6 += (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx));
    for (const [p, q] of [[a, b], [b, c], [c, a]]) {
      const k = p * 1e6 + q;
      dir.set(k, (dir.get(k) || 0) + 1);
    }
  }
  let open = 0, over = 0, flip = 0;
  const seen = new Set();
  for (const [k, n] of dir) {
    if (n > 1) over++;                       // ребро в одну сторону дважды
    const a = Math.floor(k / 1e6), b = k - a * 1e6;
    const rev = b * 1e6 + a;
    if (seen.has(rev)) continue;
    seen.add(k);
    if (!dir.has(rev)) { open++; continue; } // ребра нет в обратную сторону
    if (dir.get(rev) !== n) flip++;
  }
  // габариты и длины рёбер
  let minx = Infinity, miny = Infinity, minz = Infinity;
  let maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
  for (let v = 0; v < topo.vk; v++) {
    const o = v * 3;
    if (pos[o] < minx) minx = pos[o]; if (pos[o] > maxx) maxx = pos[o];
    if (pos[o + 1] < miny) miny = pos[o + 1]; if (pos[o + 1] > maxy) maxy = pos[o + 1];
    if (pos[o + 2] < minz) minz = pos[o + 2]; if (pos[o + 2] > maxz) maxz = pos[o + 2];
  }
  let minEdge = Infinity;
  for (const k of seen.keys()) {
    const a = Math.floor(k / 1e6), b = k - a * 1e6;
    const d = Math.hypot(pos[a * 3] - pos[b * 3], pos[a * 3 + 1] - pos[b * 3 + 1], pos[a * 3 + 2] - pos[b * 3 + 2]);
    if (d > 1e-9 && d < minEdge) minEdge = d;
  }
  const vol = Math.abs(volume6) / 6;              // в единицах модели³
  const mm = 1 / U;
  return {
    faces: nf, vertices: topo.vk,
    openEdges: open, flipEdges: flip, nonManifold: over,
    degenerate: degen,
    watertight: open === 0 && flip === 0 && over === 0,
    volumeMM3: vol * mm * mm * mm,
    sizeMM: [(maxx - minx) * mm, (maxy - miny) * mm, (maxz - minz) * mm],
    minEdgeMM: minEdge * mm
  };
}

/** Перед экспортом: выкинуть вершины, на которые не ссылается ни одна грань. */
export function compactMesh(positions, index) {
  const nv = positions.length / 3;
  const used = new Int32Array(nv).fill(-1);
  let n = 0;
  for (let i = 0; i < index.length; i++) if (used[index[i]] < 0) used[index[i]] = n++;
  const p = new Float32Array(n * 3);
  for (let v = 0; v < nv; v++) {
    const u = used[v];
    if (u < 0) continue;
    p[u * 3] = positions[v * 3];
    p[u * 3 + 1] = positions[v * 3 + 1];
    p[u * 3 + 2] = positions[v * 3 + 2];
  }
  const idx = new Int32Array(index.length);
  for (let i = 0; i < index.length; i++) idx[i] = used[index[i]];
  return { positions: p, index: idx, dropped: nv - n };
}
