"""Display geometry only: original STEP faces + oriented feature-cell boundaries.

Planar clipping is analytic. Curved intersections are tessellated display
approximations. The finite viewing envelope never contributes a boundary face.
Extraction, training geometry and recognition results are not modified.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

_import_path = list(sys.path)
from prepare import ROOT, core, gp, write_json
# prepare imports the existing extractor/viewer by adding their directories;
# restore path precedence so a later "import app" resolves to THIS demo.
sys.path[:] = _import_path

CACHE_VERSION = 2
OCC_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def sample_index():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    return {s['sample']: s for s in manifest['samples']}


@lru_cache(maxsize=8)
def load_sample(sample):
    if sample not in sample_index():
        raise ValueError('Unknown dataset sample')
    return json.loads((ROOT / 'data/extracted' / f'{sample}.json').read_text())


@lru_cache(maxsize=1)
def space_specs():
    data = json.loads((ROOT / 'config/feature_space_seeds.json').read_text())
    return {f['category_id']: f for f in data['features']}


@lru_cache(maxsize=8)
def topology_meshes(sample):
    data = load_sample(sample)
    source = ROOT / 'data/steps' / (sample + '.step')
    if not source.exists():
        source = ROOT / 'data/steps' / Path(sample_index()[sample]['copy']).name
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != data['source_sha256']:
        raise ValueError('STEP has changed since extraction; refusing incorrect face IDs')
    cache = ROOT / 'data/viewer_cache' / f'{sample}.json'
    with OCC_LOCK:
        if cache.exists():
            saved = json.loads(cache.read_text())
            if saved.get('version') == CACHE_VERSION and saved.get('sha256') == digest:
                return saved
        occ = gp._occ_imports()
        shape = gp._read_step(source, occ)
        # Exactly the extractor's ordering, not a separately assumed face-map order.
        faces = core.unique_shapes(core.explore_shapes(shape, occ['TopAbs_FACE']))
        assert len(faces) == data['fag_statistics']['face_count']
        extent = np.linalg.norm(np.array(data['bbox'][3:])-data['bbox'][:3])
        deflection = max(float(extent)*.001,1e-7)
        occ['BRepMesh_IncrementalMesh'](shape, deflection, False, 0.25, True)
        meshes = []
        for fid, face in enumerate(faces):
            mesh = gp._triangulate_face(face, occ)
            if not mesh['triangles']:
                occ['breptools'].Clean(face)
                occ['BRepMesh_IncrementalMesh'](face, deflection*.1, False, 0.15, True)
                mesh = gp._triangulate_face(face, occ)
            if not mesh['triangles']:
                raise ValueError(f'{sample}: face {fid} has no display triangulation')
            meshes.append(dict(face_id=fid, **mesh))
        edges, _ = gp._map_shapes(shape, occ['TopAbs_EDGE'], occ)
        polylines = []
        for edge in data['edges']:
            if edge['degenerate']:
                continue
            face_ids = sorted({data['coedges'][c]['face_id'] for c in edge['coedge_ids']})
            polylines.append(dict(edge_id=edge['id'], face_ids=face_ids,
                                  points=gp._sample_edge(edges[edge['id']], occ)))
        saved = dict(version=CACHE_VERSION, sha256=digest, faces=meshes, edges=polylines)
        write_json(cache, saved)
        return saved


def corners(bounds):
    return np.array(list(itertools.product(*zip(bounds[:3], bounds[3:]))))


def signed_side(role, points):
    """Positive on the void side, including angular guarding from the demo."""
    p = role['parameters']; points = np.asarray(points)
    if p['type'] == 'plane':
        return points @ np.array(role['normal']) - role['offset']
    if p['type'] != 'cylinder':
        raise ValueError(f"Unsupported feature-space role {p['type']}")
    delta = points - p['origin']; z = np.array(p['axis_z'])
    radial = delta - np.outer(delta @ z, z)
    distance = np.linalg.norm(radial, axis=1)
    value = role['sigma'] * (distance - p['radius'])
    if role.get('domain_guarded'):
        theta = np.arctan2(radial @ p['axis_y'], radial @ p['axis_x']) % (2 * math.pi)
        active = np.zeros(len(points), dtype=bool)
        for lo, hi in role['angular_domains']:
            active |= (theta >= lo - 1e-9) & (theta <= hi + 1e-9)
        value = np.where(active, value, np.maximum(abs(value), 1e-10))
    return value


def clip_polygon(poly, field, tol):
    """Sutherland-Hodgman; exact for planes, piecewise linear for curved fields."""
    if len(poly) < 3:
        return np.empty((0, 3))
    values = field(poly)
    if np.all(values >= -tol):
        return poly
    if np.all(values < -tol):
        return np.empty((0, 3))
    result = []
    for i, b in enumerate(poly):
        a = poly[i - 1]; va, vb = values[i - 1], values[i]
        ain, bin = va >= -tol, vb >= -tol
        if ain != bin:
            t = np.clip(va / (va - vb), 0, 1)
            result.append(a + t * (b - a))
        if bin:
            result.append(b)
    return np.asarray(result).reshape(-1, 3)


def linear_constraints(space, role_id, bounds):
    fields = []
    for r in space['roles']:
        if r['role_id'] != role_id and r['parameters']['type'] == 'plane':
            fields.append(lambda pts, r=r: signed_side(r, pts))
    for axis in range(3):
        fields.append(lambda pts, a=axis: pts[:, a] - bounds[a])
        fields.append(lambda pts, a=axis: bounds[a + 3] - pts[:, a])
    return fields


def subdivide(triangle, depth):
    if depth == 0:
        return [triangle]
    a, b, c = triangle; ab = (a+b)/2; bc = (b+c)/2; ca = (c+a)/2
    return sum((subdivide(np.array(t), depth-1) for t in
                [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]), [])


def role_boundary(space, role, bounds):
    """Only the active seed support patch; never render viewport closing caps."""
    bounds = np.asarray(bounds, dtype=float); box = corners(bounds)
    p = role['parameters']; origin = np.array(p['origin'])
    ax, ay, az = (np.array(p[k]) for k in ('axis_x', 'axis_y', 'axis_z'))
    linear = linear_constraints(space, role['role_id'], bounds)
    curved = [r for r in space['roles'] if r['role_id'] != role['role_id'] and r['parameters']['type'] == 'cylinder']
    tol = max(float(np.linalg.norm(bounds[3:] - bounds[:3])) * 1e-9, 1e-10)
    polygons = []
    if p['type'] == 'plane':
        u, v = (box-origin) @ ax, (box-origin) @ ay
        poly = np.array([origin + x*ax + y*ay for x,y in
                         [(u.min(),v.min()),(u.max(),v.min()),(u.max(),v.max()),(u.min(),v.max())]])
        for field in linear:
            poly = clip_polygon(poly, field, tol)
        for i in range(1, len(poly)-1):
            triangle = poly[[0,i,i+1]]
            polygons.extend(subdivide(triangle, 5) if curved else [triangle])
    elif p['type'] == 'cylinder':
        t = (box-origin) @ az
        intervals = role.get('angular_domains', [[0,2*math.pi]]) if role.get('domain_guarded') else [[0,2*math.pi]]
        for lo, hi in intervals:
            nu = max(3, int(math.ceil((hi-lo)/(2*math.pi)*128))+1)
            nv = 17 if curved else 3
            us = np.linspace(lo,hi,nu); vs = np.linspace(t.min(),t.max(),nv)
            vertices = np.array([origin+p['radius']*(math.cos(u)*ax+math.sin(u)*ay)+h*az for h in vs for u in us])
            for j in range(nv-1):
                for i in range(nu-1):
                    a = j*nu+i
                    for ids in ([a,a+1,a+nu+1],[a,a+nu+1,a+nu]):
                        poly = vertices[ids]
                        for field in linear:
                            poly = clip_polygon(poly,field,tol)
                            if len(poly) < 3: break
                        if len(poly) >= 3: polygons.append(poly)
    else:
        raise ValueError(f"Unsupported role surface {p['type']}")
    verts, tris = [], []
    for poly in polygons:
        for other in curved:
            poly = clip_polygon(poly, lambda pts, r=other: signed_side(r, pts), tol)
            if len(poly) < 3: break
        if len(poly) < 3: continue
        base = len(verts); verts.extend(poly.tolist())
        for j in range(1,len(poly)-1):
            if np.linalg.norm(np.cross(poly[j]-poly[0],poly[j+1]-poly[0])) > tol*tol:
                tris.append([base,base+j,base+j+1])
    return dict(vertices=verts, triangles=tris)


def normal_arrow(role, mesh):
    if not mesh['triangles']:
        return None
    pts = np.asarray(mesh['vertices']); triangles = np.asarray(mesh['triangles'])
    centers = pts[triangles].mean(axis=1)
    # Use a visible triangle near the patch center, not the center of a hole.
    anchor = centers[np.argmin(np.linalg.norm(centers-centers.mean(axis=0),axis=1))]
    p = role['parameters']
    if p['type'] == 'plane':
        normal = np.array(role['normal'])
    else:
        delta = anchor-p['origin']; z = np.array(p['axis_z'])
        radial = delta - np.dot(delta,z)*z
        radial /= max(np.linalg.norm(radial),1e-12)
        anchor = np.array(p['origin'])+np.dot(delta,z)*z+p['radius']*radial
        normal = role['sigma']*radial
    return anchor, normal/np.linalg.norm(normal)


def planar_feasible(space):
    """Necessary global feasibility check; never flip outward normals to fit."""
    planes = [r for r in space['roles'] if r['parameters']['type'] == 'plane']
    if not planes:
        return True
    result = linprog(np.zeros(3), A_ub=-np.array([r['normal'] for r in planes]),
                     b_ub=-np.array([r['offset'] for r in planes]),
                     bounds=[(None,None)]*3, method='highs')
    if result.status == 2:
        return False
    if not result.success:
        raise ValueError('Unable to verify oriented plane feasibility')
    return True


@lru_cache(maxsize=96)
def feature_boundaries(sample, feature_id, extension):
    data = load_sample(sample)
    space = next(f for f in data['feature_spaces'] if f['id'] == feature_id)
    spec = space_specs()[space['category_id']]
    if {r['role_id'] for r in space['roles']} != {n['id'] for n in spec['nodes']}:
        raise ValueError('Extracted roles do not match the feature-space seed definition')
    face_ids = {i for r in space['roles'] for i in r['source_face_ids']}
    meshes = topology_meshes(sample)
    points = np.array([v for f in meshes['faces'] if f['face_id'] in face_ids for v in f['vertices']])
    span = max(float(np.linalg.norm(points.max(axis=0)-points.min(axis=0))),1e-6)
    margin = span * extension
    bounds = np.r_[points.min(axis=0)-margin, points.max(axis=0)+margin]
    feasible = planar_feasible(space)
    patches = [dict(role=r, mesh=role_boundary(space,r,bounds) if feasible else dict(vertices=[],triangles=[])) for r in space['roles']]
    return dict(patches=patches, bounds=bounds.tolist(), span=span,
                planar_feasible=feasible,
                approximate=any(r['parameters']['type']=='cylinder' for r in space['roles']))
