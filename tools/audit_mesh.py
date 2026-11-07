#!/usr/bin/env python3
"""Reproducible, non-destructive STL audit. Coordinates retain the source units."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def audit(path):
    path = Path(path)
    raw = trimesh.load(path, process=False)
    mesh = raw.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    components = sorted(mesh.split(only_watertight=False), key=lambda m: len(m.faces), reverse=True)
    edge_counts = np.bincount(mesh.edges_unique_inverse)
    lengths = np.linalg.norm(mesh.triangles[:, [1, 2, 0]] - mesh.triangles[:, [0, 1, 2]], axis=2)
    result = {
        'file': str(path),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'units': 'unspecified in STL; no automatic millimetre conversion',
        'raw_vertices': len(raw.vertices),
        'vertices_after_index_welding': len(mesh.vertices),
        'faces': len(mesh.faces),
        'bounds': mesh.bounds.tolist(),
        'extents': mesh.extents.tolist(),
        'surface_area': float(mesh.area),
        'signed_volume': float(mesh.volume),
        'watertight': bool(mesh.is_watertight),
        'winding_consistent': bool(mesh.is_winding_consistent),
        'positive_component_volumes': all(c.volume > 0 for c in components),
        'boundary_edges': int((edge_counts == 1).sum()),
        'nonmanifold_edges': int((edge_counts > 2).sum()),
        'degenerate_faces_after_welding': int((mesh.area_faces < 1e-10).sum()),
        'duplicate_faces_after_welding': int(len(mesh.faces) - len(np.unique(np.sort(mesh.faces, axis=1), axis=0))),
        'components': [
            {'faces': len(c.faces), 'vertices': len(c.vertices), 'bounds': c.bounds.tolist(),
             'watertight': bool(c.is_watertight), 'winding_consistent': bool(c.is_winding_consistent),
             'signed_volume': float(c.volume), 'euler_number': int(c.euler_number)}
            for c in components
        ],
        'edge_length_quantiles_0_50_99_100': np.quantile(lengths, [0, .5, .99, 1]).tolist(),
        'notes': [
            'Index welding is an audit operation; STL itself stores separate triangle vertices.',
            'Separate closed wheel shells are not automatically treated as stray objects.',
            'Closed edges and consistent winding do not establish absence of inter-shell intersections.',
            'Visual fidelity and fitness for a particular manufacturing process require separate checks.',
        ],
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stl')
    ap.add_argument('--output')
    args = ap.parse_args()
    result = audit(args.stl)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
