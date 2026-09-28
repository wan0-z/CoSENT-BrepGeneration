"""Surface-first causal token grammar with exact-source audit and quantization."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent
GROUPS={"topology":0,"vertex":1,"edge_bbox":2,"surface":3,"condition":4}


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


class Quantizer:
    """Per-model isotropic frame, shared scalar bins, no learned geometry model.

    The radius also encloses support origins and control points, so normalization
    never silently clips geometry outside the model's trimmed-face bbox.
    """
    def __init__(self,data,bins=1024):
        self.bins=bins
        bb=np.array(data["bbox"]); self.center=(bb[:3]+bb[3:])/2
        points=[v["xyz"] for v in data["vertices"]]
        lengths=[]
        for s in data["surfaces"]:
            p=s["parameters"]
            if "origin" in p: points.append(p["origin"])
            points.extend(p.get("poles",[]))
            lengths.extend(abs(p[k]) for k in ("radius","reference_radius","major_radius","minor_radius") if k in p)
        for e in data["edges"]: points.extend([e["bbox"][:3],e["bbox"][3:]])
        # Conditions can use a different equivalent origin than their surface group.
        for f in data["feature_spaces"]:
            points.extend(r["parameters"]["origin"] for r in f["roles"])
        self.scale=max(float(np.max(np.abs(np.array(points)-self.center))),max(lengths,default=0),1e-9)*1.000001
        self.errors=[]

    def scalar(self,x,lo,hi,kind):
        if not math.isfinite(x) or x<lo-1e-8 or x>hi+1e-8:
            raise ValueError(f"Out-of-range quantization {kind}: {x} not in [{lo},{hi}]")
        q=int(round((x-lo)/(hi-lo)*(self.bins-1)))
        q=max(0,min(self.bins-1,q))
        restored=lo+q/(self.bins-1)*(hi-lo)
        self.errors.append((kind,abs(restored-x)))
        return f"<Q{q}>"

    def point(self,p):
        return [self.scalar(float((x-c)/self.scale),-1,1,"coordinate") for x,c in zip(p,self.center)]

    def direction(self,p):
        return [self.scalar(float(x),-1,1,"direction") for x in p]

    def meta(self):
        return dict(center=self.center.tolist(),scale=self.scale,bins=self.bins,
                    point_max_error_per_axis=self.scale/(self.bins-1),
                    observed_max_normalized_error=max((e for _,e in self.errors),default=0))


def ordered_coedges(data):
    """Stable viewer order, but classify same-surface second occurrences dynamically."""
    order=[]; seen=set()
    for surface in data["surfaces"]:
        nodes=sorted([n for n in data["coedges"] if n["surface_id"]==surface["id"]],
                     key=lambda n:(not n["is_outer_loop"],n["face_id"],n["loop_id"],n["local_edge_id_in_loop"]))
        new=[]; mates=[]
        for n in nodes:
            if n["edge_id"] in seen: mates.append(n)
            else: new.append(n); seen.add(n["edge_id"])
        order.extend(new+mates)
    return order


class Stream:
    def __init__(self,q): self.tokens=[]; self.groups=[]; self.q=q
    def add(self,*tokens,group="topology"):
        self.tokens.extend(tokens); self.groups.extend([GROUPS[group]]*len(tokens))
    def integer(self,n,group="topology"):
        if int(n)<0: raise ValueError(f"Negative ID/integer {n}")
        self.add(f"<I{int(n)}>",group=group)
    def vector(self,tag,value,kind,group):
        self.add(f"<{tag}>",group=group)
        self.add(*(self.q.point(value) if kind=="point" else self.q.direction(value)),group=group)
        self.add(f"</{tag}>",group=group)
    def geometry(self,p,group):
        self.add(f"<TYPE_{p['type']}>",group=group)
        for key in ("origin","axis_z","axis_x","axis_y"):
            if key in p: self.vector(key.upper(),p[key],"point" if key=="origin" else "direction",group)
        for key in ("radius","reference_radius","major_radius","minor_radius"):
            if key in p:
                self.add(f"<{key.upper()}>",self.q.scalar(float(p[key]/self.q.scale),0,1,"length"),group=group)
        if "semi_angle" in p:
            self.add("<SEMI_ANGLE>",self.q.scalar(float(p["semi_angle"]),-math.pi,math.pi,"angle"),group=group)
        if "poles" in p:
            for k in ("u_degree","v_degree","n_u_poles","n_v_poles"):
                self.add(f"<{k.upper()}>",group=group); self.integer(p[k],group)
            self.add("<QUASI_UNIFORM>" if p.get("quasi_uniform") else "<GENERAL_SPLINE>",group=group)
            for axis in ("u","v"):
                if axis+"_knots" not in p: continue
                knots=p[axis+"_knots"]; lo=knots[0]; span=knots[-1]-lo
                self.add(f"<{axis.upper()}_KNOTS>",group=group)
                self.add(*[self.q.scalar((k-lo)/span,0,1,"knot") for k in knots],group=group)
                self.add(f"</{axis.upper()}_KNOTS>",f"<{axis.upper()}_MULTIPLICITIES>",group=group)
                for m in p[axis+"_multiplicities"]: self.integer(m,group)
                self.add(f"</{axis.upper()}_MULTIPLICITIES>",f"<{axis.upper()}_PERIODIC_{int(p[axis+'_periodic'])}>",group=group)
            self.add("<CONTROL_POINTS>",group=group)
            for point in p["poles"]: self.vector("CP",point,"point",group)
            self.add("</CONTROL_POINTS>","<WEIGHTS>",group=group)
            largest=max(p["weights"])
            self.add(*[self.q.scalar(w/largest,0,1,"weight") for w in p["weights"]],group=group)
            self.add("</WEIGHTS>",group=group)


def serialize(data,conditional=True,bins=1024):
    q=Quantizer(data,bins); s=Stream(q); s.add("<BOS>")
    conditions=data["feature_spaces"] if conditional else []
    condmap={f["id"]:i for i,f in enumerate(conditions)}
    if not conditions: s.add("<NO_CONDITION>",group="condition")
    else:
        s.add("<CONDITION>",group="condition")
        for f in conditions:
            s.add("<FEATURE>",group="condition"); s.integer(condmap[f["id"]],"condition")
            s.add(f"<FEATURE_{f['name']}>",f"<SPACE_{f['space_model']}>",group="condition")
            for role in f["roles"]:
                s.add("<CONDITION_SURFACE>",group="condition"); s.integer(role["role_id"],"condition")
                s.add(f"<ROLE_{role['role']}>",group="condition")
                s.geometry(role["parameters"],"condition")
                s.add(f"<VOID_SIDE_{role['sigma']}>","<CLIP_BY_ROLES>",group="condition")
                for rid in role["trim_by_roles"]: s.integer(rid,"condition")
                s.add("</CLIP_BY_ROLES>",group="condition")
                if "angular_domains" in role and role["domain_guarded"]:
                    s.add("<ANGULAR_DOMAINS>",group="condition")
                    for lo,hi in role["angular_domains"]:
                        s.add("<INTERVAL>",q.scalar(lo,0,2*math.pi,"angle"),q.scalar(hi,0,2*math.pi,"angle"),"</INTERVAL>",group="condition")
                    s.add("</ANGULAR_DOMAINS>",group="condition")
                s.add("</CONDITION_SURFACE>",group="condition")
            s.add("</FEATURE>",group="condition")
        s.add("</CONDITION>",group="condition")
    prefix_length=len(s.tokens)
    s.add("<MODEL>")
    order=ordered_coedges(data); cmap={}; vmap={}; emap={}; fmap={}; lmap={}; previous=[]; seen_edges={}
    audit=[]; bbox_emitted=set(); in_context=False
    for surf in data["surfaces"]:
        s_nodes=[n for n in order if n["surface_id"]==surf["id"]]
        cond_ids=[i for i in surf["condition_ids"] if i in condmap]
        if not cond_ids and not in_context: s.add("<CONTEXT>"); in_context=True
        if cond_ids and in_context: raise AssertionError("Condition surface follows context surface")
        s.add("<SURFACE>"); s.integer(surf["id"])
        for f in conditions:
            for role in f["roles"]:
                if surf["id"] in role["surface_ids"]:
                    s.add("<CONDITION_REF>"); s.integer(condmap[f["id"]]); s.integer(role["role_id"])
        s.add("<COEDGES>"); mates_started=False
        for node in s_nodes:
            start=len(s.tokens); old=node["id"]; cid=len(cmap); cmap[old]=cid; eid=node["edge_id"]
            is_mate=eid in seen_edges
            if is_mate: mates_started=True
            elif mates_started: raise AssertionError("New coedge follows mate within a surface")
            s.add("<COEDGE>"); s.integer(cid)
            for field,table,tag in (("face_id",fmap,"FACE"),("loop_id",lmap,"LOOP")):
                oid=node[field]; new=oid not in table
                if new: table[oid]=len(table)
                s.add(f"<{'NEW' if new else 'REF'}_{tag}>"); s.integer(table[oid])
                if tag=="FACE" and new: s.add(f"<FACE_ORIENTATION_{node['face_orientation']}>")
                if tag=="LOOP" and new: s.add("<OUTER>" if node["is_outer_loop"] else "<INNER>")
            s.add(f"<TYPE_{node['surface_type']}>",f"<EDGE_TYPE_{node['edge_type']}>",
                  f"<CONVEXITY_{node['convexity']}>",f"<DIHEDRAL_{node['dihedral_type']}>")
            if node["dihedral_angle"] is None: s.add("<ANGLE_UNKNOWN>")
            else: s.add("<DIHEDRAL_ANGLE>",q.scalar(float(node["dihedral_angle"]),0,360,"dihedral"))
            s.add(f"<ORIENTATION_{node['orientation']}>")
            for field,tag in (("start_vertex_id","START_VERTEX"),("end_vertex_id","END_VERTEX")):
                vid=node[field]; s.add(f"<{tag}>")
                if vid not in vmap:
                    vmap[vid]=len(vmap); s.add("<NEW_POINT>",group="vertex"); s.integer(vmap[vid],"vertex")
                    s.vector("XYZ",data["vertices"][vid]["xyz"],"point","vertex"); s.add("</NEW_POINT>",group="vertex")
                else: s.add("<POINT_REF>"); s.integer(vmap[vid])
                s.add(f"</{tag}>")
            s.add("<PREVIOUS_CONNECTIONS>")
            for prior in previous:
                bits=[eid==prior["edge_id"],node["next"]==prior["id"],node["previous"]==prior["id"],
                      node["curve_key"]==prior["curve_key"] or eid==prior["edge_id"],
                      node["surface_id"]==prior["surface_id"],node["face_id"]==prior["face_id"],
                      node["start_vertex_id"]==prior["start_vertex_id"],node["end_vertex_id"]==prior["end_vertex_id"]]
                mask=sum(int(b)<<i for i,b in enumerate(bits))
                if mask:
                    s.add("<LINK>"); s.integer(cmap[prior["id"]]); s.add(f"<REL_{mask}>")
            s.add("</PREVIOUS_CONNECTIONS>")
            if is_mate:
                s.add("<MATE>"); s.integer(cmap[seen_edges[eid]]); s.add("<EDGE_REF>"); s.integer(emap[eid])
                if eid in bbox_emitted: raise AssertionError("bbox repeated")
                s.add("<EDGE_BBOX>",group="edge_bbox")
                eb=data["edges"][eid]["bbox"]
                s.vector("P_MIN",eb[:3],"point","edge_bbox"); s.vector("P_MAX",eb[3:],"point","edge_bbox")
                s.add("</EDGE_BBOX>",group="edge_bbox"); bbox_emitted.add(eid)
            else:
                seen_edges[eid]=old; emap[eid]=len(emap); s.add("<NEW_EDGE>"); s.integer(emap[eid])
            s.add("</COEDGE>"); previous.append(node)
            audit.append(dict(coedge_id=cid,source_coedge_id=old,surface_id=surf["id"],edge_id=emap[eid],
                              kind="mate" if is_mate else "new",token_start=start,token_end=len(s.tokens)))
        s.add("</COEDGES>","<LOOP_ORDER>")
        for lp in data["loops"]:
            if lp["face_id"] not in surf["face_ids"]: continue
            s.add("<LOOP_REF>"); s.integer(lmap[lp["id"]])
            s.add("<ORDERED_COEDGE_REFS>")
            for n in lp["coedge_ids"]: s.integer(cmap[n])
            s.add("</ORDERED_COEDGE_REFS>")
        s.add("</LOOP_ORDER>","<SURFACE_GEOMETRY>",group="surface")
        s.geometry(surf["parameters"],"surface")
        s.add("</SURFACE_GEOMETRY>",group="surface"); s.add("</SURFACE>")
    if in_context: s.add("</CONTEXT>")
    s.add("</MODEL>","<END_MODEL>","<EOS>")
    expected={e["id"] for e in data["edges"] if len(e["coedge_ids"])==2}
    assert expected==bbox_emitted
    assert len(cmap)==len(data["coedges"]) and len(vmap)==len(data["vertices"])
    return dict(sample=data["sample"],tokens=s.tokens,groups=s.groups,prefix_length=prefix_length,
                quantization=q.meta(),audit=audit,counts=dict(coedges=len(cmap),vertices=len(vmap),edges=len(emap),
                bbox_records=len(bbox_emitted),unpaired_edges=len(emap)-len(bbox_emitted),surfaces=len(data["surfaces"])),
                maps=dict(vertices=vmap,coedges=cmap,edges=emap,faces=fmap,loops=lmap))


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--bins",type=int,default=1024); args=ap.parse_args()
    paths=sorted((ROOT/"data/extracted").glob("*.json"))
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    if len(paths)!=len(manifest["samples"]): raise ValueError("Extraction incomplete; do not silently train on fewer than the copied samples")
    all_tokens={"<PAD>","<UNK>"}|{f"<Q{i}>" for i in range(args.bins)}|{f"<REL_{i}>" for i in range(256)}
    seed_specs=json.loads((ROOT/"config/feature_space_seeds.json").read_text())["features"]
    all_tokens.update(f"<FEATURE_{f['name']}>" for f in seed_specs)
    all_tokens.update(f"<ROLE_{r['role']}>" for f in seed_specs for r in f["nodes"])
    all_tokens.update(f"<SPACE_{f['space_model']}>" for f in seed_specs)
    all_tokens.update(f"<TYPE_{t}>" for t in ("plane","cylinder","cone","sphere","torus","bspline_surface","bezier_surface"))
    all_tokens.update(f"<EDGE_TYPE_{t}>" for t in ("line","circle","ellipse","hyperbola","parabola","bspline_curve","bezier_curve","other_curve"))
    all_tokens.update({"<QUASI_UNIFORM>","<GENERAL_SPLINE>"})
    summary=[]; categories=Counter(); types=Counter()
    for path in paths:
        data=json.loads(path.read_text()); categories.update(f["category_name"] for f in data["instances"])
        types.update(s["parameters"]["type"] for s in data["surfaces"])
        variants={}
        for label,cond in (("conditional",True),("unconditional",False)):
            seq=serialize(data,cond,args.bins); variants[label]=seq
            all_tokens.update(seq["tokens"])
        save(ROOT/"data/sequences"/path.name,variants)
        c=variants["conditional"]
        summary.append(dict(sample=data["sample"],tokens=len(c["tokens"]),prefix_tokens=c["prefix_length"],
                            features=len(data["instances"]),**c["counts"]))
        print(f"{path.stem}: {len(c['tokens'])} tokens, {c['prefix_length']} condition tokens",flush=True)
    # IDs are deterministic ordinal symbols, not learned numerical codebooks.
    max_id=max(int(t[2:-1]) for t in all_tokens if t.startswith("<I") and t[2:-1].isdigit())
    all_tokens.update(f"<I{i}>" for i in range(max_id+1))
    vocab=["<PAD>","<UNK>"]+sorted(all_tokens-{"<PAD>","<UNK>"})
    save(ROOT/"data/vocabulary.json",dict(tokens=vocab,token_to_id={t:i for i,t in enumerate(vocab)},groups=GROUPS,bins=args.bins))
    names=[p.stem for p in paths]; random.Random(42).shuffle(names); nval=max(1,len(names)//10)
    save(ROOT/"data/split.json",dict(seed=42,train=sorted(names[nval:]),validation=sorted(names[:nval])))
    seeds=json.loads((ROOT/"config/feature_seeds.json").read_text())["features"]
    save(ROOT/"data/corpus_report.json",dict(samples=summary,vocabulary_size=len(vocab),
        class_counts={f["name"]:categories[f["name"]] for f in seeds},surface_types=dict(types),
        observed_classes=len(categories),total_tokens=sum(s["tokens"] for s in summary),
        split="90% sample-level train / 10% held-out validation; no cross-sample windows"))


if __name__=="__main__": main()
