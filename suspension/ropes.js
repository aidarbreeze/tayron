/* ============================================================
   Канат: путь, привязанный к граням кузова, и его отдельный объект-трубка.

   Что здесь считается (и почему именно так):
   • Путь каната — это набор точек, каждая из которых получена попаданием луча
     в грань кузова, поэтому канат повторяет все изгибы поверхности и «привязан»
     к её треугольникам (при правке кузова он остаётся там, где лёг).
   • Перед постройкой трубки путь пересэмплируется с ровным шагом: иначе на
     медленных участках штриха кольца лепятся друг на друга, а на быстрых —
     трубка гранёная.
   • «Верх» сечения берётся из нормали поверхности, а не из случайного вектора:
     так канат не закручивается вокруг своей оси и его можно класть на борт.
   • Трубка замкнута (торцы — веера): её можно проверить той же проверкой, что и
     кузов, и выгрузить отдельным объектом.

   Всё в единицах модели (U = 5.8824 ед./мм).
   ============================================================ */

/** длина ломаной, ед. модели */
export function pathLen(pts) {
  let L = 0;
  for (let i = 1; i < pts.length; i++) {
    L += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]);
  }
  return L;
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const mul = (a, k) => [a[0] * k, a[1] * k, a[2] * k];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };

/** сгладить ломаную усреднением по трём (концы остаются на месте) */
export function smoothPath(pts, nrms, iters = 2) {
  let p = pts.map(q => q.slice());
  let n = nrms ? nrms.map(q => q.slice()) : null;
  for (let k = 0; k < iters; k++) {
    const q = p.map((v, i) => (i === 0 || i === p.length - 1) ? v.slice()
      : [(p[i - 1][0] + 2 * v[0] + p[i + 1][0]) / 4,
         (p[i - 1][1] + 2 * v[1] + p[i + 1][1]) / 4,
         (p[i - 1][2] + 2 * v[2] + p[i + 1][2]) / 4]);
    p = q;
    if (n) n = n.map((v, i) => (i === 0 || i === n.length - 1) ? v.slice()
      : norm([(n[i - 1][0] + 2 * v[0] + n[i + 1][0]) / 4,
              (n[i - 1][1] + 2 * v[1] + n[i + 1][1]) / 4,
              (n[i - 1][2] + 2 * v[2] + n[i + 1][2]) / 4]));
  }
  return { pts: p, nrms: n };
}

/** пересадить ломаную на ровный шаг step; концы сохраняются */
export function resamplePath(pts, nrms, step) {
  if (pts.length < 2 || !(step > 0)) return { pts: pts.map(p => p.slice()), nrms: nrms ? nrms.map(n => n.slice()) : null };
  const out = [pts[0].slice()];
  const outN = nrms ? [nrms[0].slice()] : null;
  let carry = 0;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    const seg = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    if (seg < 1e-9) continue;
    let t = step - carry;
    while (t <= seg) {
      const f = t / seg;
      out.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]);
      if (nrms && outN) {
        const A = nrms[i - 1], B = nrms[i] || A;
        outN.push(norm([A[0] + (B[0] - A[0]) * f, A[1] + (B[1] - A[1]) * f, A[2] + (B[2] - A[2]) * f]));
      }
      t += step;
    }
    carry = seg - (t - step);
  }
  const last = pts[pts.length - 1];
  const tail = out[out.length - 1];
  if (Math.hypot(last[0] - tail[0], last[1] - tail[1], last[2] - tail[2]) > step * 0.35) {
    out.push(last.slice());
    if (nrms && outN) outN.push((nrms[nrms.length - 1] || [0, 1, 0]).slice());
  }
  return { pts: out, nrms: outN };
}

/** касательная в точке i (центральная разность, на концах — односторонняя) */
export function tangentAt(pts, i) {
  const n = pts.length;
  const a = pts[Math.max(0, i - 1)], b = pts[Math.min(n - 1, i + 1)];
  const t = sub(b, a);
  if (Math.hypot(t[0], t[1], t[2]) < 1e-9) return [1, 0, 0];
  return norm(t);
}

/**
 * Трубка по пути — отдельный объект каната.
 * @param {Array<[number,number,number]>} path центры колец (на поверхности)
 * @param {Array<[number,number,number]>|null} nrms нормали поверхности в тех же точках
 * @param {number} radius радиус трубки, ед. модели
 * @param {number} sides граней в кольце
 * @param {number} off подъём оси над поверхностью (0 — вровень, radius — лежит сверху)
 */
export function tubeArrays(path, nrms, radius, sides = 10, off = 0) {
  const n = path.length;
  const S = Math.max(3, sides | 0);
  const verts = n * S + 2;
  const positions = new Float32Array(verts * 3);
  const index = new Uint32Array(((n - 1) * S * 2 + S * 2) * 3);
  let ii = 0;
  const put = (v, p) => { positions[v * 3] = p[0]; positions[v * 3 + 1] = p[1]; positions[v * 3 + 2] = p[2]; };

  for (let i = 0; i < n; i++) {
    const t = tangentAt(path, i);
    let up = nrms && nrms[i] ? nrms[i] : [0, 1, 0];
    up = sub(up, mul(t, dot(up, t)));            // нормаль поверхности без касательной части
    if (Math.hypot(up[0], up[1], up[2]) < 1e-6) {
      up = Math.abs(t[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      up = sub(up, mul(t, dot(up, t)));
    }
    up = norm(up);
    const e1 = cross(up, t);                     // e1 × up = t — обход колец правый
    const c = off ? [path[i][0] + (nrms && nrms[i] ? nrms[i][0] : up[0]) * off,
                     path[i][1] + (nrms && nrms[i] ? nrms[i][1] : up[1]) * off,
                     path[i][2] + (nrms && nrms[i] ? nrms[i][2] : up[2]) * off] : path[i];
    for (let j = 0; j < S; j++) {
      const a = (j / S) * Math.PI * 2;
      put(i * S + j, [c[0] + radius * (Math.cos(a) * e1[0] + Math.sin(a) * up[0]),
                      c[1] + radius * (Math.cos(a) * e1[1] + Math.sin(a) * up[1]),
                      c[2] + radius * (Math.cos(a) * e1[2] + Math.sin(a) * up[2])]);
    }
  }
  /* боковина */
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < S; j++) {
      const j2 = (j + 1) % S;
      const a = i * S + j, b = i * S + j2, c = (i + 1) * S + j2, d = (i + 1) * S + j;
      index[ii++] = a; index[ii++] = b; index[ii++] = c;
      index[ii++] = a; index[ii++] = c; index[ii++] = d;
    }
  }
  /* торцы: веера к центрам крайних колец */
  const c0 = n * S, c1 = n * S + 1;
  const centre = (v) => {
    let x = 0, y = 0, z = 0;
    for (let j = 0; j < S; j++) { const k = v * S + j; x += positions[k * 3]; y += positions[k * 3 + 1]; z += positions[k * 3 + 2]; }
    return [x / S, y / S, z / S];
  };
  put(c0, centre(0)); put(c1, centre(n - 1));
  for (let j = 0; j < S; j++) {
    const j2 = (j + 1) % S;
    index[ii++] = c0; index[ii++] = j2; index[ii++] = j;             // начало: нормаль −t
    index[ii++] = c1; index[ii++] = (n - 1) * S + j; index[ii++] = (n - 1) * S + j2; // конец: +t
  }
  return { positions, index, rings: n, sides: S, verts, radius, off };
}

/** Нормали пути без выбросов: у кромки панели луч попадает в крутую грань, и
    одна-две нормали смотрят вбок. Если по ним поднимать поверхность, полоса
    «съезжает» за край панели (проверено стендом rope_half: у уступа 10 мм
    поверхность поднималась на 11 мм). Среднее по окну + порог 35°: одиночный
    выброс заменяется на среднее, плавный изгиб остаётся как есть. */
export function cleanNormals(pts, nrms, win = 3, deg = 35, rounds = 2) {
  if (!nrms || nrms.length !== pts.length) return nrms;
  const lim = (deg * Math.PI) / 180;
  const unit = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; };
  const angle = (a, b) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));
  let cur = nrms.map(unit);
  for (let r = 0; r < rounds; r++) {
    const next = cur.map((n) => n.slice());
    for (let i = 0; i < cur.length; i++) {
      let sx = 0, sy = 0, sz = 0, k = 0;
      for (let j = Math.max(0, i - win); j <= Math.min(cur.length - 1, i + win); j++) {
        sx += cur[j][0]; sy += cur[j][1]; sz += cur[j][2]; k++;
      }
      const avg = unit([sx / k, sy / k, sz / k]);
      if (angle(cur[i], avg) > lim) next[i] = avg;
    }
    cur = next;
  }
  return cur;
}

/** Трубка, обрезанная поверхностью: остаётся только часть, которая торчит
    НАД плоскостью поверхности, всё, что ушло внутрь модели, не рисуется и не
    существует. Сечение — круговой сегмент, закрытый хордой (плоским срезом).

    off — насколько ось трубки приподнята над поверхностью вдоль нормали:
      off = 0     — половина каната (сегмент 180°, ровно то, что нужно);
      off = R     — канат лежит сверху целиком (сегмент 360°, обычная трубка);
      off = −R/2  — канат утоплен на полтора радиуса, видна четверть.
    Плоскость среза локальная: своя в каждом кольце, по нормали поверхности,
    поэтому на изогнутом кузове срез идёт по коже, а не по одной плоскости. */
export function arcTubeArrays(path, nrms, radius, sides = 10, off = 0) {
  const n = path.length;
  const R = radius;
  const S = Math.max(4, (sides | 0) + (sides % 2));      // точки на дуге, включая концы
  const s0 = Math.max(-1, Math.min(1, -off / R));        // sin(a0): что ниже — отрезано
  const a0 = Math.asin(s0), a1 = Math.PI - a0;           // дуга: a0 … a1
  const verts = n * S;                                   // только кольца: торцы — веера
  const positions = new Float32Array(verts * 3);
  const tris = (n - 1) * ((S - 1) * 2 + 2) + (S - 2) * 2; // бока + дно + два торца
  const index = new Uint32Array(tris * 3);
  let ii = 0;
  const put = (v, p) => { positions[v * 3] = p[0]; positions[v * 3 + 1] = p[1]; positions[v * 3 + 2] = p[2]; };
  for (let i = 0; i < n; i++) {
    const t = tangentAt(path, i);
    let up = nrms && nrms[i] ? nrms[i] : [0, 1, 0];
    up = sub(up, mul(t, dot(up, t)));
    if (Math.hypot(up[0], up[1], up[2]) < 1e-6) {
      up = Math.abs(t[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      up = sub(up, mul(t, dot(up, t)));
    }
    up = norm(up);
    const e1 = cross(up, t);
    const nrm = nrms && nrms[i] ? nrms[i] : up;
    const c = [path[i][0] + nrm[0] * off, path[i][1] + nrm[1] * off, path[i][2] + nrm[2] * off];
    for (let j = 0; j < S; j++) {
      const a = a0 + (a1 - a0) * (j / (S - 1));
      put(i * S + j, [c[0] + R * (Math.cos(a) * e1[0] + Math.sin(a) * up[0]),
                      c[1] + R * (Math.cos(a) * e1[1] + Math.sin(a) * up[1]),
                      c[2] + R * (Math.cos(a) * e1[2] + Math.sin(a) * up[2])]);
    }
  }
  /* бока */
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < S - 1; j++) {
      const a = i * S + j, b = i * S + j + 1, c = (i + 1) * S + j + 1, d = (i + 1) * S + j;
      index[ii++] = a; index[ii++] = b; index[ii++] = c;
      index[ii++] = a; index[ii++] = c; index[ii++] = d;
    }
  }
  /* плоский срез (хорда) — дно каната, которое и лежит на поверхности */
  for (let i = 0; i < n - 1; i++) {
    const a = i * S, b = (i + 1) * S, c = (i + 1) * S + (S - 1), d = i * S + (S - 1);
    index[ii++] = a; index[ii++] = b; index[ii++] = c;
    index[ii++] = a; index[ii++] = c; index[ii++] = d;
  }
  /* торцы: веер из крайней точки хорды — сечение выпуклое, центр не нужен,
     а хорда остаётся одним ребром, общим с дном (иначе висят швы) */
  for (let j = 1; j < S - 1; j++) {
    index[ii++] = 0; index[ii++] = j + 1; index[ii++] = j;                    // начало: нормаль −t
  }
  const base = (n - 1) * S;
  for (let j = 1; j < S - 1; j++) {
    index[ii++] = base; index[ii++] = base + j; index[ii++] = base + j + 1;   // конец: нормаль +t
  }
  return { positions, index, rings: n, sides: S, verts, radius: R, off,
    cut: 180 - 2 * (a0 * 180) / Math.PI,        // сколько градусов сечения осталось
    half: off === 0 };
}

/** габарит трубки — для отчётов и тестов */
export function tubeStats(t) {
  let x0 = Infinity, y0 = Infinity, z0 = Infinity, x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
  const P = t.positions;
  for (let i = 0; i < P.length; i += 3) {
    if (P[i] < x0) x0 = P[i]; if (P[i] > x1) x1 = P[i];
    if (P[i + 1] < y0) y0 = P[i + 1]; if (P[i + 1] > y1) y1 = P[i + 1];
    if (P[i + 2] < z0) z0 = P[i + 2]; if (P[i + 2] > z1) z1 = P[i + 2];
  }
  return { verts: t.verts, rings: t.rings, sides: t.sides, tris: t.index.length / 3,
    size: [x1 - x0, y1 - y0, z1 - z0], min: [x0, y0, z0], max: [x1, y1, z1] };
}

/** излом пути: сумма углов между соседними отрезками, градусы (0 — прямая) */
export function pathTurn(pts) {
  let sum = 0;
  for (let i = 1; i + 1 < pts.length; i++) {
    const a = sub(pts[i], pts[i - 1]), b = sub(pts[i + 1], pts[i]);
    const la = Math.hypot(a[0], a[1], a[2]), lb = Math.hypot(b[0], b[1], b[2]);
    if (la < 1e-9 || lb < 1e-9) continue;
    const c = Math.max(-1, Math.min(1, dot(a, b) / (la * lb)));
    sum += Math.acos(c) * 180 / Math.PI;
  }
  return sum;
}

/** максимальный излом на одном узле, градусы */
export function pathKink(pts) {
  let mx = 0;
  for (let i = 1; i + 1 < pts.length; i++) {
    const a = sub(pts[i], pts[i - 1]), b = sub(pts[i + 1], pts[i]);
    const la = Math.hypot(a[0], a[1], a[2]), lb = Math.hypot(b[0], b[1], b[2]);
    if (la < 1e-9 || lb < 1e-9) continue;
    const c = Math.max(-1, Math.min(1, dot(a, b) / (la * lb)));
    mx = Math.max(mx, Math.acos(c) * 180 / Math.PI);
  }
  return mx;
}

/** выровнять путь: лапласово сглаживание, концы на месте (k — доля шага) */
export function relaxPath(pts, nrms, iters = 10, k = 0.5) {
  let p = pts.map(q => q.slice());
  let n = nrms ? nrms.map(q => q.slice()) : null;
  const m = p.length;
  if (m < 3) return { pts: p, nrms: n };
  for (let it = 0; it < iters; it++) {
    const q = p.map((v, i) => (i === 0 || i === m - 1) ? v
      : [v[0] + k * ((p[i - 1][0] + p[i + 1][0]) / 2 - v[0]),
         v[1] + k * ((p[i - 1][1] + p[i + 1][1]) / 2 - v[1]),
         v[2] + k * ((p[i - 1][2] + p[i + 1][2]) / 2 - v[2])]);
    p = q;
    if (n) n = n.map((v, i) => (i === 0 || i === n.length - 1) ? v
      : norm([n[i - 1][0] + n[i + 1][0], n[i - 1][1] + n[i + 1][1], n[i - 1][2] + n[i + 1][2]]));
  }
  return { pts: p, nrms: n };
}

/** поставить путь по прямой между концами (точки равномерно, нормали — по концам) */
/** Скользящее среднее по длине дуги: убирает всё короче окна ±halfW.
    Концы остаются на месте — канат не «съедается» по краям. */
export function smoothWindow(pts, nrms, halfW, passes = 2) {
  const n = pts.length;
  if (n < 3 || !(halfW > 0)) return { pts: pts.map(p => p.slice()), nrms: (nrms || []).map(v => v.slice()) };
  const s = [0];
  for (let i = 1; i < n; i++) {
    const a = pts[i - 1], b = pts[i];
    s.push(s[i - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
  }
  let cur = pts.map(p => p.slice());
  for (let pass = 0; pass < passes; pass++) {
    const next = cur.map(p => p.slice());
    let lo = 0, hi = 0;
    for (let i = 0; i < n; i++) {
      while (lo < i && s[i] - s[lo] > halfW) lo++;
      if (hi < i) hi = i;
      while (hi + 1 < n && s[hi + 1] - s[i] <= halfW) hi++;
      let ax = 0, ay = 0, az = 0, c = 0;
      for (let j = lo; j <= hi; j++) { ax += cur[j][0]; ay += cur[j][1]; az += cur[j][2]; c++; }
      next[i] = [ax / c, ay / c, az / c];
    }
    next[0] = cur[0].slice(); next[n - 1] = cur[n - 1].slice();
    cur = next;
  }
  return { pts: cur, nrms: (nrms || []).map(v => v.slice()) };
}

export function straightPath(pts, nrms, count = 0) {
  const m = pts.length;
  const A = pts[0], B = pts[m - 1];
  const n = count > 1 ? count : Math.max(2, m);
  const out = [], outN = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    out.push([A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t]);
    if (nrms) {
      const NA = nrms[0], NB = nrms[m - 1];
      outN.push(norm([NA[0] + (NB[0] - NA[0]) * t, NA[1] + (NB[1] - NA[1]) * t, NA[2] + (NB[2] - NA[2]) * t]));
    }
  }
  return { pts: out, nrms: nrms ? outN : null };
}
