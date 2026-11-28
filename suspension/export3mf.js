/* ============================================================
   Запись 3MF прямо в браузере (без зависимостей).

   3MF — это ZIP с тремя файлами: [Content_Types].xml, _rels/.rels и
   3D/3dmodel.model (сама сетка в XML). Сжимать не обязательно: ZIP
   допускает метод STORE, поэтому нужен только CRC32.

   Формат чисел и структура — как в m3mf.py/Tayron5_flex.3mf: единицы
   модели (миллиметры файла), печать уменьшением до 150 мм по длине.
   ============================================================ */

const CT = '<?xml version="1.0" encoding="UTF-8"?>\n' +
  '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
  '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
  '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>' +
  '</Types>';

const RELS = '<?xml version="1.0" encoding="UTF-8"?>\n' +
  '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' +
  '<Relationship Target="/3D/3dmodel.model" Id="rel0" ' +
  'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>';

/* ---------- CRC32 ---------- */
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(u8) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < u8.length; i++) c = CRC_TABLE[(c ^ u8[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

const enc = new TextEncoder();

/* ---------- ZIP (метод STORE, без сжатия) ---------- */
export function zipStore(files) {
  const chunks = [], central = [];
  let offset = 0;
  const put = (u8) => { chunks.push(u8); offset += u8.length; };
  const u16 = (v) => [v & 0xFF, (v >>> 8) & 0xFF];
  const u32 = (v) => [v & 0xFF, (v >>> 8) & 0xFF, (v >>> 16) & 0xFF, (v >>> 24) & 0xFF];

  for (const f of files) {
    const name = enc.encode(f.name);
    const data = typeof f.data === 'string' ? enc.encode(f.data) : f.data;
    const crc = crc32(data);
    const head = new Uint8Array([
      0x50, 0x4B, 0x03, 0x04,        // local file header
      20, 0,                          // version needed
      0, 0,                           // flags
      0, 0,                           // method: 0 = store
      0, 0, 0, 0,                     // time/date (фиксировано)
      ...u32(crc), ...u32(data.length), ...u32(data.length),
      ...u16(name.length), 0, 0
    ]);
    put(head); put(name); put(data);
    central.push({ name, crc, size: data.length, offset: offset - data.length - name.length - head.length });
  }

  const cdStart = offset;
  for (const e of central) {
    const rec = new Uint8Array([
      0x50, 0x4B, 0x01, 0x02,          // central directory header (46 байт)
      ...u16(20), ...u16(20),          // version made by / needed
      ...u16(0),  ...u16(0),           // flags, method (store)
      ...u16(0),  ...u16(0),           // время, дата
      ...u32(e.crc),                   // CRC32
      ...u32(e.size), ...u32(e.size),  // сжатый / исходный размер
      ...u16(e.name.length),           // длина имени
      ...u16(0), ...u16(0),            // extra, comment
      ...u16(0), ...u16(0),            // диск, внутренние атрибуты
      ...u32(0),                       // внешние атрибуты
      ...u32(e.offset)                 // смещение локального заголовка
    ]);
    put(rec); put(e.name);
  }
  const cdSize = offset - cdStart;
  put(new Uint8Array([
    0x50, 0x4B, 0x05, 0x06, 0, 0, 0, 0,
    ...u16(central.length), ...u16(central.length),
    ...u32(cdSize), ...u32(cdStart), 0, 0
  ]));

  const total = chunks.reduce((s, c) => s + c.length, 0);
  const out = new Uint8Array(total);
  let p = 0;
  for (const c of chunks) { out.set(c, p); p += c.length; }
  return out;
}

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');

/* ---------- сетка → 3MF ----------
   positions — Float32Array [x,y,z,…] (в единицах модели),
   indices   — Int32Array  [i,j,k,…]  (индексы в positions).            */
export function meshTo3mf(positions, indices, opts = {}) {
  const name = opts.name || 'Кузов';
  const meta = opts.meta || 'VW Tayron';
  const nv = positions.length / 3;
  const xml = [];
  xml.push('<?xml version="1.0" encoding="UTF-8"?>');
  xml.push('<model unit="millimeter" xml:lang="ru-RU" ' +
    'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">');
  xml.push('<metadata name="Title">' + esc(meta) + '</metadata>');
  xml.push('<resources><basematerials id="1">' +
    '<base name="PLA" displaycolor="#FFB06040"/></basematerials>');
  xml.push('<object id="2" type="model" name="' + esc(name) + '" pid="1" pindex="0">' +
    '<mesh><vertices>');
  for (let i = 0; i < nv; i++) {
    xml.push('<vertex x="' + positions[i * 3].toFixed(4) +
      '" y="' + positions[i * 3 + 1].toFixed(4) +
      '" z="' + positions[i * 3 + 2].toFixed(4) + '"/>');
  }
  xml.push('</vertices><triangles>');
  for (let i = 0; i < indices.length; i += 3) {
    xml.push('<triangle v1="' + indices[i] + '" v2="' + indices[i + 1] +
      '" v3="' + indices[i + 2] + '"/>');
  }
  xml.push('</triangles></mesh></object></resources>');
  xml.push('<build><item objectid="2"/></build></model>');
  return zipStore([
    { name: '[Content_Types].xml', data: CT },
    { name: '_rels/.rels', data: RELS },
    { name: '3D/3dmodel.model', data: xml.join('') }
  ]);
}

/* ---------- сетка → STL (двоичный) ----------
   STL безразмерный: пишем те же числа, что и в 3MF (печать уменьшением до 17 %). */
export function meshToSTL(positions, indices, name = 'Tayron') {
  const nf = indices.length / 3;
  const buf = new ArrayBuffer(84 + nf * 50);
  const dv = new DataView(buf);
  const head = ('Tayron body: ' + name).slice(0, 79);
  for (let i = 0; i < head.length; i++) dv.setUint8(i, head.charCodeAt(i) & 0x7f);
  dv.setUint32(80, nf, true);
  let off = 84;
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i] * 3, b = indices[i + 1] * 3, c = indices[i + 2] * 3;
    const ax = positions[a], ay = positions[a + 1], az = positions[a + 2];
    const bx = positions[b], by = positions[b + 1], bz = positions[b + 2];
    const cx = positions[c], cy = positions[c + 1], cz = positions[c + 2];
    const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
    const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
    let nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    dv.setFloat32(off, nx, true); dv.setFloat32(off + 4, ny, true); dv.setFloat32(off + 8, nz, true);
    dv.setFloat32(off + 12, ax, true); dv.setFloat32(off + 16, ay, true); dv.setFloat32(off + 20, az, true);
    dv.setFloat32(off + 24, bx, true); dv.setFloat32(off + 28, by, true); dv.setFloat32(off + 32, bz, true);
    dv.setFloat32(off + 36, cx, true); dv.setFloat32(off + 40, cy, true); dv.setFloat32(off + 44, cz, true);
    dv.setUint16(off + 48, 0, true);
    off += 50;
  }
  return new Uint8Array(buf);
}

/* ---------- отдать файл пользователю ---------- */
export function downloadBytes(bytes, filename, type = 'model/3mf') {
  const blob = new Blob([bytes], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { a.remove(); setTimeout(() => URL.revokeObjectURL(url), 4000); }, 1500);
  return { size: bytes.length, url, filename };
}
