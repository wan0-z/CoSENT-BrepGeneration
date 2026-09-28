"""Invariant tests for topology causality, geometric timing and real training logs."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
import torch

from model import PrefixTransformer
from prepare import contains
from serialize import Quantizer,ordered_coedges,serialize

ROOT=Path(__file__).resolve().parent


class MethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads((ROOT/"data/extracted/000000.json").read_text())

    def test_exact_sources_not_modified(self):
        import hashlib
        manifest=json.loads((ROOT/"data/manifest.json").read_text())
        self.assertEqual(len(manifest["samples"]),100)
        for item in manifest["samples"]:
            self.assertEqual(hashlib.sha256(Path(item["copy"]).read_bytes()).hexdigest(),item["sha256"])
            self.assertEqual(hashlib.sha256(Path(item["source"]).read_bytes()).hexdigest(),item["sha256"])

    def test_quantization_error_bound(self):
        q=Quantizer(self.data)
        for v in self.data["vertices"]:
            encoded=q.point(v["xyz"])
            restored=[q.center[i]+(int(t[2:-1])/(q.bins-1)*2-1)*q.scale for i,t in enumerate(encoded)]
            self.assertLessEqual(float(np.max(np.abs(np.array(restored)-v["xyz"]))),q.scale/(q.bins-1)+1e-10)

    def test_causal_attention_has_no_future_target_leakage(self):
        torch.manual_seed(7); torch.set_num_threads(2)
        model=PrefixTransformer(50,dim=32,layers=2,heads=4,dropout=0,prefix_window=8).eval()
        prefix=torch.randint(0,50,(19,)); body=torch.randint(0,50,(15,)); pos=torch.arange(70,85)
        modified=body.clone(); modified[9:]=(modified[9:]+1)%50
        with torch.no_grad():
            a=model(prefix,body,pos); b=model(prefix,modified,pos)
        torch.testing.assert_close(a[:9],b[:9],rtol=0,atol=1e-6)
        changed=prefix.clone(); changed[0]=(changed[0]+1)%50
        with torch.no_grad(): c=model(changed,body,pos)
        self.assertGreater(float((a-c).abs().max()),1e-7)

    def test_same_surface_seam_is_mate_after_all_new_edges(self):
        fixture={"surfaces":[{"id":0}],"coedges":[
            dict(id=i,surface_id=0,edge_id=e,is_outer_loop=True,face_id=0,loop_id=0,local_edge_id_in_loop=i)
            for i,e in enumerate([4,4,5,6,6])]}
        self.assertEqual([n["id"] for n in ordered_coedges(fixture)],[0,2,3,1,4])

    def test_unconditional_has_no_dangling_condition_references(self):
        seq=serialize(self.data,False)
        self.assertIn("<NO_CONDITION>",seq["tokens"])
        self.assertNotIn("<CONDITION_REF>",seq["tokens"])

    def test_blind_slot_oriented_cell_not_finite_box(self):
        def plane(n,o):
            return dict(parameters=dict(type="plane",origin=o),normal=n,offset=float(np.dot(n,o)))
        space={"roles":[plane([1,0,0],[0,0,0]),plane([-1,0,0],[2,0,0]),
                        plane([0,1,0],[0,0,0]),plane([0,0,1],[0,0,0])]}
        np.testing.assert_array_equal(contains(space,[[1,1,1],[1,10000,10000],[3,1,1],[1,-1,1]]),[True,True,False,False])

    def test_trimmed_cylinder_keeps_angular_guard(self):
        role=dict(parameters=dict(type="cylinder",origin=[0,0,0],axis_x=[1,0,0],axis_y=[0,1,0],axis_z=[0,0,1],radius=1),
                  sigma=-1,domain_guarded=True,angular_domains=[[0,math.pi/2],[3*math.pi/2,2*math.pi]])
        np.testing.assert_array_equal(contains({"roles":[role]},[[0.5,0,10],[2,0,0],[-100,0,0]]),[True,False,True])

    def test_all_corpus_geometry_and_reference_invariants(self):
        paths=sorted((ROOT/"data/sequences").glob("*.json")); self.assertEqual(len(paths),100)
        vocab=json.loads((ROOT/"data/vocabulary.json").read_text())["token_to_id"]
        specs=json.loads((ROOT/"config/feature_seeds.json").read_text())["features"]
        self.assertEqual(len(specs),20)
        for f in specs: self.assertIn(f"<FEATURE_{f['name']}>",vocab)
        for path in paths:
            variants=json.loads(path.read_text())
            for seq in variants.values():
                tokens=seq["tokens"]; seen_points=set(); cid=-1
                self.assertEqual(len(tokens),len(seq["groups"]))
                self.assertTrue(all(t in vocab for t in tokens))
                self.assertEqual(tokens.count("<NEW_POINT>"),seq["counts"]["vertices"])
                self.assertEqual(tokens.count("<EDGE_BBOX>"),seq["counts"]["bbox_records"])
                self.assertEqual(tokens[-2:],["<END_MODEL>","<EOS>"])
                for i,t in enumerate(tokens[:-1]):
                    if t=="<COEDGE>": cid=int(tokens[i+1][2:-1])
                    elif t=="<NEW_POINT>":
                        vid=int(tokens[i+1][2:-1]); self.assertNotIn(vid,seen_points); seen_points.add(vid)
                    elif t=="<POINT_REF>": self.assertIn(int(tokens[i+1][2:-1]),seen_points)
                    elif t in ("<MATE>","<LINK>"): self.assertLess(int(tokens[i+1][2:-1]),cid)
                for item in seq["audit"]:
                    span=tokens[item["token_start"]:item["token_end"]]
                    self.assertEqual("<EDGE_BBOX>" in span,item["kind"]=="mate")
                    if item["kind"]=="mate": self.assertLess(span.index("<MATE>"),span.index("<EDGE_BBOX>"))
                bysurface={}
                for a in seq["audit"]:
                    self.assertFalse(a["kind"]=="new" and bysurface.get(a["surface_id"],False))
                    bysurface[a["surface_id"]]=a["kind"]=="mate"

    def test_split_is_sample_disjoint(self):
        split=json.loads((ROOT/"data/split.json").read_text())
        self.assertEqual(len(split["train"]),90); self.assertEqual(len(split["validation"]),10)
        self.assertFalse(set(split["train"])&set(split["validation"]))


if __name__=="__main__": unittest.main(verbosity=2)
