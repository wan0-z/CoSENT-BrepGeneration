"""Copy STEP data and extract feature-seed instances + exact training geometry.

Only feature_seeds.json is used for recognition. The space seed file defines
the oriented condition representation after recognition, never the matcher.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent
sys.path.insert(0, str(THESIS / "Seed_Extractor"))
sys.path.insert(0, str(THESIS / "geometry_reconstruct_viewer"))
import extract_one_step as core
import extract_geometry_sequence as viewer

gp = viewer.gp
RELATIONS = gp.RELATION_ORDER


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def xyz(p):
    return [float(p.X()), float(p.Y()), float(p.Z())]


def frame(pos):
    return dict(origin=xyz(pos.Location()), axis_x=xyz(pos.XDirection()),
                axis_y=xyz(pos.YDirection()), axis_z=xyz(pos.Direction()))


def surface_parameters(face, occ):
    a = occ["BRepAdaptor_Surface"](face, True)
    name = gp._face_type(face, occ)
    out = {"type": name}
    getters = {"plane": "Plane", "cylinder": "Cylinder", "cone": "Cone",
               "sphere": "Sphere", "torus": "Torus"}
    if name in getters:
        item = getattr(a, getters[name])()
        out.update(frame(item.Position()))
        if name in ("cylinder", "sphere"):
            out["radius"] = float(item.Radius())
        elif name == "cone":
            out.update(reference_radius=float(item.RefRadius()), semi_angle=float(item.SemiAngle()))
        elif name == "torus":
            out.update(major_radius=float(item.MajorRadius()), minor_radius=float(item.MinorRadius()))
    elif name in ("bspline_surface", "bezier_surface"):
        sp = a.BSpline() if name == "bspline_surface" else a.Bezier()
        nu, nv = sp.NbUPoles(), sp.NbVPoles()
        out.update(u_degree=sp.UDegree(), v_degree=sp.VDegree(), n_u_poles=nu, n_v_poles=nv,
                   poles=[xyz(sp.Pole(i,j)) for i in range(1,nu+1) for j in range(1,nv+1)],
                   weights=[float(sp.Weight(i,j)) for i in range(1,nu+1) for j in range(1,nv+1)])
        if name == "bspline_surface":
            for axis in ("U", "V"):
                out[axis.lower()+"_knots"] = [float(getattr(sp,axis+"Knot")(i)) for i in range(1,getattr(sp,"Nb"+axis+"Knots")()+1)]
                out[axis.lower()+"_multiplicities"] = [int(getattr(sp,axis+"Multiplicity")(i)) for i in range(1,getattr(sp,"Nb"+axis+"Knots")()+1)]
                out[axis.lower()+"_periodic"] = bool(getattr(sp,"Is"+axis+"Periodic")())
            out["quasi_uniform"] = all(
                len(out[k+"_knots"]) <= 2 or np.allclose(np.diff(out[k+"_knots"]),np.diff(out[k+"_knots"])[0])
                for k in ("u","v")) and all(
                out[k+"_multiplicities"][0] == out[k+"_degree"]+1 and
                out[k+"_multiplicities"][-1] == out[k+"_degree"]+1 and
                all(m == 1 for m in out[k+"_multiplicities"][1:-1]) for k in ("u","v"))
    else:
        raise ValueError(f"Unsupported exact surface family {name}; no partial/fake geometry record")
    return out


def bbox(shape):
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.BRepBndLib import brepbndlib
    box = Bnd_Box()
    # Exact geometry bounding algorithm, not sampled display-polyline extents.
    brepbndlib.AddOptimal(shape, box, False, False)
    if box.IsVoid():
        raise ValueError("Void geometry bounding box")
    return list(map(float,box.Get()))


def angular_domains(face_ids, faces, parameters, occ):
    """Merge cylinder patch angular domains in the representative support frame.

    Keep only angular clipping in the condition. Axial endpoints and arbitrary
    trim loops are topology to generate, not conditioning leakage.
    """
    o=np.array(parameters["origin"]); x=np.array(parameters["axis_x"]); y=np.array(parameters["axis_y"])
    intervals=[]
    for fid in face_ids:
        a=occ["BRepAdaptor_Surface"](faces[fid],True)
        u0,u1,v0,v1=map(float,occ["breptools"].UVBounds(faces[fid]))
        if u1-u0 >= 2*math.pi-1e-6:
            return [[0.,2*math.pi]]
        samples=[np.array(xyz(a.Value(float(u),(v0+v1)/2)))-o for u in np.linspace(u0,u1,33)]
        angles=np.unwrap([math.atan2(float(p@y),float(p@x)) for p in samples])
        lo,hi=float(min(angles)),float(max(angles))
        lo_mod=lo%(2*math.pi); hi_mod=lo_mod+hi-lo
        if hi_mod <= 2*math.pi:
            intervals.append([lo_mod,hi_mod])
        else:
            intervals.extend([[lo_mod,2*math.pi],[0.,hi_mod-2*math.pi]])
    merged=[]
    for lo,hi in sorted(intervals):
        if merged and lo <= merged[-1][1]+1e-6:
            merged[-1][1]=max(hi,merged[-1][1])
        else:
            merged.append([lo,hi])
    return merged


def make_spaces(instances, specs, faces, face_to_surface, surfaces, occ):
    result=[]
    for inst in instances:
        spec=specs[int(inst["category_id"])]
        roles=[]
        for node in spec["nodes"]:
            rid=str(node["id"])
            groups=inst.get("role_groups",{})
            members=groups.get(rid) or groups.get(node["role"])
            if not members:
                mapped=inst.get("seed_to_model",{}).get(rid)
                members=[mapped] if mapped is not None else inst["face_ids"] if len(spec["nodes"])==1 else []
            if not members:
                raise ValueError(f"Missing role {rid} in recognized {inst['category_name']}")
            params=surface_parameters(faces[members[0]],occ)
            sigma=-1 if faces[members[0]].Orientation()==occ["TopAbs_REVERSED"] else 1
            role=dict(role_id=int(rid),role=node["role"],source_face_ids=members,
                      surface_ids=sorted({face_to_surface[f] for f in members}),parameters=params,
                      sigma=sigma,trim_by_roles=[int(n["id"]) for n in spec["nodes"] if n["id"]!=node["id"]])
            if params["type"]=="plane":
                role["normal"]=(sigma*np.array(params["axis_z"])).tolist()
                role["offset"]=float(np.dot(role["normal"],params["origin"]))
            elif params["type"]=="cylinder":
                role["angular_domains"]=angular_domains(members,faces,params,occ)
                role["domain_guarded"]=spec["space_model"]!="global_oriented_support_cell"
            roles.append(role)
        result.append(dict(id=inst["instance_id"],category_id=inst["category_id"],name=inst["category_name"],
            space_model=spec["space_model"],roles=roles,relations=spec.get("relations",[]),
            group_constraints=spec.get("group_constraints",[]),
            predicate="intersection of oriented void-side support constraints; trimmed cylinders are angular-domain guarded",
            clipping="mutual role constraints; open directions remain unbounded; finite bbox is only an evaluation envelope",
            recognition_collision=inst.get("feature_space_collision"),
            validation="feature_seeds recognition; feature-space draft rules stored, not a certification of all space predicates"))
    return result


def contains(space, points, epsilon=0.):
    """Executable oriented implicit-cell membership in original model units."""
    pts=np.atleast_2d(points).astype(float); mask=np.ones(len(pts),dtype=bool)
    for r in space["roles"]:
        p=r["parameters"]; delta=pts-np.array(p["origin"])
        if p["type"]=="plane":
            mask &= pts@np.array(r["normal"]) >= r["offset"]+epsilon
        elif p["type"]=="cylinder":
            radial=delta-np.outer(delta@np.array(p["axis_z"]),p["axis_z"])
            phi=np.sum(radial**2,axis=1)-p["radius"]**2
            active=np.ones(len(pts),bool)
            if r["domain_guarded"]:
                theta=np.arctan2(radial@np.array(p["axis_y"]),radial@np.array(p["axis_x"]))%(2*math.pi)
                active=np.zeros(len(pts),bool)
                for lo,hi in r["angular_domains"]:
                    active |= (theta>=lo-1e-9)&(theta<=hi+1e-9)
            mask &= (~active)|(r["sigma"]*phi>=epsilon)
    return mask


def extract(step):
    occ=gp._occ_imports(); shape=gp._read_step(step,occ)
    graph,faces,stats=core.build_attributed_fag(shape)
    if stats["failed_two_face_edge_count"]:
        raise ValueError(f"Failed FAG edge attributes: {stats}")
    core.FEATURE_SEED_JSON_PATH=ROOT/"config/feature_seeds.json"
    instances,_,_=core.extract_features_in_priority_order(graph,core.load_feature_seeds(),shape)
    brep_edges,edge_map=gp._map_shapes(shape,occ["TopAbs_EDGE"],occ)
    vertices,vmap=gp._map_shapes(shape,occ["TopAbs_VERTEX"],occ)
    # Reconcile extractor edge ordering through OCC identity, never raw indices.
    core_edges=[]; seen=set()
    for f in faces:
        for e in core.unique_shapes(core.explore_shapes(f,occ["TopAbs_EDGE"])):
            eid=int(edge_map.FindIndex(e))-1
            if eid not in seen:
                seen.add(eid); core_edges.append(eid)
    attrs={}
    for _,_,_,a in graph.edges(keys=True,data=True):
        attrs[core_edges[a["topological_edge_id"]]]=a
    groups={}
    for fid,f in enumerate(faces):
        key=viewer._surface_key(f,fid,occ)
        groups.setdefault(key,[]).append(fid)
    face_conditions=defaultdict(set)
    for inst in instances:
        for fid in inst["face_ids"]:
            face_conditions[fid].add(inst["instance_id"])
    ordered=sorted(groups.values(),key=lambda ids:(min((i for f in ids for i in face_conditions[f]),default=10**9),ids[0]))
    surfaces=[]; f2s={}
    for sid,fids in enumerate(ordered):
        surfaces.append(dict(id=sid,face_ids=fids,condition_ids=sorted({i for f in fids for i in face_conditions[f]}),
                             parameters=surface_parameters(faces[fids[0]],occ)))
        f2s.update({f:sid for f in fids})
    nodes=[]; loops=[]; edge_nodes=defaultdict(list)
    for fid,f in enumerate(faces):
        outer=occ["breptools"].OuterWire(f)
        wx=occ["TopExp_Explorer"](f,occ["TopAbs_WIRE"])
        while wx.More():
            w=occ["topods"].Wire(wx.Current()); members=[]
            it=occ["BRepTools_WireExplorer"](w,f)
            while it.More():
                edge=occ["topods"].Edge(it.Current()); eid=int(edge_map.FindIndex(edge))-1
                sv=occ["topexp"].FirstVertex(edge,True); ev=occ["topexp"].LastVertex(edge,True)
                if sv.IsNull() or ev.IsNull():
                    raise ValueError("Vertex-free coedge requires a separate periodic-curve grammar")
                a=attrs.get(eid,{})
                node=dict(id=len(nodes),face_id=fid,surface_id=f2s[fid],loop_id=len(loops),
                    local_edge_id_in_loop=len(members),is_outer_loop=w.IsSame(outer),edge_id=eid,
                    start_vertex_id=int(vmap.FindIndex(sv))-1,end_vertex_id=int(vmap.FindIndex(ev))-1,
                    orientation=1 if edge.Orientation()!=occ["TopAbs_REVERSED"] else -1,
                    face_orientation=1 if f.Orientation()!=occ["TopAbs_REVERSED"] else -1,
                    surface_type=gp._face_type(f,occ),edge_type=gp._edge_type(edge,occ),
                    convexity=a.get("convexity","unknown"),dihedral_type=a.get("dihedral_type","unknown"),
                    dihedral_angle=a.get("dihedral_angle_degrees"),
                    curve_key=repr(gp._curve_signature(edge,eid,occ)))
                members.append(node["id"]); edge_nodes[eid].append(node["id"]); nodes.append(node); it.Next()
            if not members:
                raise ValueError("Wire explorer returned no coedges; refusing unordered topology fallback")
            for i,nid in enumerate(members):
                nodes[nid]["next"]=members[(i+1)%len(members)]
                nodes[nid]["previous"]=members[(i-1)%len(members)]
            loops.append(dict(id=len(loops),face_id=fid,coedge_ids=members,outer=w.IsSame(outer)))
            wx.Next()
    edges=[]
    for eid,e in enumerate(brep_edges):
        deg=bool(occ["BRep_Tool"].Degenerated(e))
        try:
            eb=bbox(e)
        except Exception:
            if not deg or not edge_nodes[eid]: raise
            v=vertices[nodes[edge_nodes[eid][0]]["start_vertex_id"]]; pt=xyz(occ["BRep_Tool"].Pnt(v)); eb=pt+pt
        if len(edge_nodes[eid])>2:
            raise ValueError("Nonmanifold edge has more than two coedges")
        edges.append(dict(id=eid,bbox=eb,coedge_ids=edge_nodes[eid],degenerate=deg))
    specs={s["category_id"]:s for s in json.loads((ROOT/"config/feature_space_seeds.json").read_text())["features"]}
    spaces=make_spaces(instances,specs,faces,f2s,surfaces,occ)
    return dict(schema_version=1,sample=step.stem,source_sha256=hashlib.sha256(step.read_bytes()).hexdigest(),
                bbox=bbox(shape),fag_statistics=stats,instances=instances,feature_spaces=spaces,surfaces=surfaces,
                vertices=[dict(id=i,xyz=xyz(occ["BRep_Tool"].Pnt(v))) for i,v in enumerate(vertices)],
                coedges=nodes,edges=edges,loops=loops,relation_order=RELATIONS)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=THESIS/"datasets/eccv2026-cad-challenge-data/train/target_step")
    ap.add_argument("--count",type=int,default=100)
    ap.add_argument("--only",help="Extract only this copied sample stem")
    ap.add_argument("--force",action="store_true")
    args=ap.parse_args()
    for d in ("data/steps","data/extracted","logs/extraction","config"): (ROOT/d).mkdir(parents=True,exist_ok=True)
    for name in ("feature_seeds.json","feature_space_seeds.json"):
        dest=ROOT/"config"/name
        if not dest.exists(): shutil.copy2(THESIS/"Seed_Extractor/data"/name,dest)
    manifest_path=ROOT/"data/manifest.json"
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if len(manifest["samples"])!=args.count: raise ValueError("Existing manifest has a different count; choose a new demo directory")
    else:
        paths=sorted((p for p in args.source.iterdir() if p.suffix.lower() in (".step",".stp")),key=lambda p:p.name)
        if len(paths)<args.count: raise ValueError("Not enough STEP files")
        samples=[]
        for src in paths[:args.count]:
            dst=ROOT/"data/steps"/src.name
            if dst.exists() and dst.read_bytes()!=src.read_bytes(): raise FileExistsError(dst)
            if not dst.exists(): shutil.copy2(src,dst)
            samples.append(dict(sample=src.stem,source=str(src.resolve()),copy=str(dst),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
        manifest=dict(selection="first naturally/numerically sorted training target STEP files; no feature balancing or replacement",samples=samples)
        write_json(manifest_path,manifest)
    results=[]
    for entry in manifest["samples"]:
        sample=entry["sample"]
        if args.only and sample!=args.only: continue
        out=ROOT/"data/extracted"/f"{sample}.json"; start=time.monotonic()
        try:
            if out.exists() and not args.force:
                data=json.loads(out.read_text())
            else:
                with (ROOT/"logs/extraction"/f"{sample}.log").open("w") as log,contextlib.redirect_stdout(log):
                    data=extract(Path(entry["copy"]))
                write_json(out,data)
            row=dict(sample=sample,status="ok",faces=data["fag_statistics"]["face_count"],coedges=len(data["coedges"]),
                     features=len(data["instances"]),seconds=round(time.monotonic()-start,3))
        except Exception as e:
            row=dict(sample=sample,status="error",error=f"{type(e).__name__}: {e}",traceback=traceback.format_exc())
        results.append(row); print(json.dumps(row,ensure_ascii=False),flush=True)
        write_json(ROOT/"data/extraction_report.json",results)


if __name__=="__main__": main()
