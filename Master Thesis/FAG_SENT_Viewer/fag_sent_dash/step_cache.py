from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import numpy as np

from .models import PartRecord


def _jsonable_array(a: np.ndarray) -> list:
    return np.asarray(a, dtype=float).round(7).tolist()


def _occ_available() -> bool:
    try:
        import OCC.Core.STEPControl  # noqa
        return True
    except Exception:
        return False


def tessellate_step_with_occ(step_path: Path, linear_deflection: float = 0.08, angular_deflection: float = 0.4) -> dict[str, Any]:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRep import BRep_Tool
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCC.Core.GCPnts import GCPnts_UniformAbscissa
    from OCC.Core.BRepTools import breptools

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read STEP: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, linear_deflection, False, angular_deflection, True)

    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 1
    all_pts = []
    while exp.More():
        face = exp.Current()
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation(face, loc)
        if tri is not None:
            trsf = loc.Transformation()
            verts = []
            for i in range(1, tri.NbNodes() + 1):
                p = tri.Node(i).Transformed(trsf)
                verts.append([p.X(), p.Y(), p.Z()])
            tris = []
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                a, b, c = t.Get()
                tris.append([a - 1, b - 1, c - 1])
            arr = np.array(verts, dtype=float) if verts else np.zeros((0, 3))
            if len(arr):
                all_pts.append(arr)
            try:
                surf = BRepAdaptor_Surface(face)
                surf_type = str(surf.GetType())
                u1, u2, v1, v2 = breptools.UVBounds(face)
            except Exception:
                surf_type, u1, u2, v1, v2 = "unknown", 0, 1, 0, 1
            faces.append({
                "id": face_id,
                "vertices": _jsonable_array(arr),
                "triangles": tris,
                "surface_type": surf_type,
                "uv_bounds": [float(u1), float(u2), float(v1), float(v2)],
            })
        else:
            faces.append({"id": face_id, "vertices": [], "triangles": [], "surface_type": "not_tessellated", "uv_bounds": [0,1,0,1]})
        face_id += 1
        exp.Next()

    edges = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    edge_id = 1
    while exp.More():
        edge = exp.Current()
        curve = BRepAdaptor_Curve(edge)
        first = curve.FirstParameter()
        last = curve.LastParameter()
        if not math.isfinite(first) or not math.isfinite(last) or abs(last - first) > 1e8:
            first, last = 0.0, 1.0
        pts = []
        for t in np.linspace(first, last, 40):
            p = curve.Value(float(t))
            pts.append([p.X(), p.Y(), p.Z()])
        arr = np.array(pts, dtype=float)
        all_pts.append(arr)
        edges.append({"id": edge_id, "points": _jsonable_array(arr), "curve_type": str(curve.GetType())})
        edge_id += 1
        exp.Next()

    if all_pts:
        pcat = np.vstack(all_pts)
        bbox = [*pcat.min(axis=0).tolist(), *pcat.max(axis=0).tolist()]
    else:
        bbox = [0, 0, 0, 1, 1, 1]
    return {"backend": "pythonocc", "faces": faces, "step_edges": edges, "bbox": bbox}


def build_part_cache(part: PartRecord, cache_dir: str | Path, force: bool = False) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    out = cache_root / f"{part.part_id}.webcache.json"
    if out.exists() and not force:
        return out
    if not _occ_available():
        raise RuntimeError("pythonocc-core is required to build STEP webcache. Install via conda-forge.")
    geom = tessellate_step_with_occ(part.step_path)
    payload = {
        "part_id": part.part_id,
        "json_path": str(part.json_path),
        "step_path": str(part.step_path),
        "nodes": part.nodes,
        "fag_edges": [e.__dict__ for e in part.edges],
        "geometry": geom,
        "note": "FAG edge id is the index of JSON edges. STEP edge id is OCC traversal index; these are not necessarily identical.",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return out


def load_part_cache(part: PartRecord, cache_dir: str | Path) -> dict[str, Any]:
    p = Path(cache_dir) / f"{part.part_id}.webcache.json"
    if not p.exists():
        p = build_part_cache(part, cache_dir)
    return json.loads(p.read_text(encoding="utf-8"))
