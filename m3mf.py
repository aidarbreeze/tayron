"""Минимальное чтение/запись 3MF (только сетки, без трансформов).

Запись проверена на Cura/PrusaSlicer/Bambu Studio: один файл — много объектов,
у каждого своё имя, все координаты — в миллиметрах печати.
"""
import zipfile
import numpy as np

try:                      # lxml есть не везде — хватает и стандартного xml
    from lxml import etree
except Exception:         # pragma: no cover
    import xml.etree.ElementTree as etree

NS = {'m': 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'}

CT = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>'''

RELS = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>'''


def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
                  .replace('>', '&gt;').replace('"', '&quot;'))


def write(path, parts, meta=None):
    """parts: [(имя, trimesh.Trimesh), ...] — координаты как есть, без сдвигов."""
    head = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<model unit="millimeter" xml:lang="ru-RU" '
            'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">']
    if meta:
        head.append('<metadata name="Title">%s</metadata>' % _esc(meta))
    head += ['<resources>',
             '<basematerials id="1"><base name="PLA" displaycolor="#FFB06040"/></basematerials>']
    body = []
    for i, (name, m) in enumerate(parts, start=2):
        V = np.asarray(m.vertices, float)
        F = np.asarray(m.faces)
        s = ['<object id="%d" type="model" name="%s" pid="1" pindex="0">' % (i, _esc(name)),
             '<mesh><vertices>']
        s += ['<vertex x="%.4f" y="%.4f" z="%.4f"/>' % tuple(v) for v in V]
        s.append('</vertices><triangles>')
        s += ['<triangle v1="%d" v2="%d" v3="%d"/>' % tuple(f) for f in F]
        s.append('</triangles></mesh></object>')
        body.append(''.join(s))
    head += body
    head.append('</resources><build>')
    head += ['<item objectid="%d"/>' % (k + 2) for k in range(len(parts))]
    head.append('</build></model>')
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', CT)
        z.writestr('_rels/.rels', RELS)
        z.writestr('3D/3dmodel.model', ''.join(head))
    return path


def read(path):
    """→ [(имя, Trimesh), ...] в порядке файла."""
    import trimesh
    z = zipfile.ZipFile(path)
    root = etree.fromstring(z.read('3D/3dmodel.model'))
    out = []
    for obj in root.iterfind('.//m:object', NS):
        V = np.array([[float(e.get('x')), float(e.get('y')), float(e.get('z'))]
                      for e in obj.iterfind('.//m:vertex', NS)])
        F = np.array([[int(e.get('v1')), int(e.get('v2')), int(e.get('v3'))]
                      for e in obj.iterfind('.//m:triangle', NS)])
        if len(F):
            out.append((obj.get('name') or 'object', 
                        trimesh.Trimesh(vertices=V, faces=F, process=False)))
    return out
