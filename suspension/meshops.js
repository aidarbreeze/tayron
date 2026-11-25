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

  return { count, vk, pos0, canon, tris, off, nbr, welded };
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
export function paintBall(topo, reg, p, r, pos) {
  const { vk } = topo;
  const P = pos || topo.pos0;
  const r2 = r * r;
  let added = 0;
  for (let v = 0; v < vk; v++) {
    if (reg[v]) continue;
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
