"""Display invariants, including unbounded openings and partial cylinder domains."""
import math
import unittest

import numpy as np

from app import feature_options, make_app, make_figure
from feature_viewer_geometry import feature_boundaries, load_sample, planar_feasible, role_boundary, sample_index, signed_side, topology_meshes


def plane(rid, origin, normal):
    normal=np.array(normal,dtype=float)
    helper=np.array([1.,0,0]) if abs(normal[0])<.8 else np.array([0.,1,0])
    x=np.cross(helper,normal); x/=np.linalg.norm(x); y=np.cross(normal,x)
    return dict(role_id=rid,role=str(rid),normal=normal.tolist(),offset=float(normal@origin),sigma=1,
                parameters=dict(type='plane',origin=origin,axis_x=x.tolist(),axis_y=y.tolist(),axis_z=normal.tolist()))


class ViewerTests(unittest.TestCase):
    def test_blind_slot_has_only_four_seed_boundaries(self):
        roles=[plane(0,[0,0,0],[1,0,0]),plane(1,[2,0,0],[-1,0,0]),
               plane(2,[0,0,0],[0,1,0]),plane(3,[0,0,0],[0,0,1])]
        bounds=[-3,-3,-3,5,8,9]; space=dict(roles=roles)
        for role in roles:
            mesh=role_boundary(space,role,bounds); self.assertTrue(mesh['triangles'])
            pts=np.array(mesh['vertices'])
            self.assertLess(np.abs(signed_side(role,pts)).max(),1e-8)
            for r in roles: self.assertGreaterEqual(signed_side(r,pts).min(),-1e-8)
        bottom=role_boundary(space,roles[3],bounds)
        self.assertAlmostEqual(np.array(bottom['vertices'])[:,1].max(),8)
        side=role_boundary(space,roles[0],bounds)
        self.assertAlmostEqual(np.array(side['vertices'])[:,2].max(),9)
        # No extra y=8 or z=9 plane was created to close these open directions.
        self.assertEqual(len(roles),4)

    def test_cylinder_not_extended_to_full_period_or_capped(self):
        role=dict(role_id=0,role='wall',sigma=-1,domain_guarded=True,angular_domains=[[0,math.pi/2]],
                  parameters=dict(type='cylinder',origin=[0,0,0],axis_x=[1,0,0],axis_y=[0,1,0],axis_z=[0,0,1],radius=1))
        mesh=role_boundary(dict(roles=[role]),role,[-2,-2,-4,2,2,7]); pts=np.array(mesh['vertices'])
        self.assertTrue(mesh['triangles'])
        np.testing.assert_allclose(np.linalg.norm(pts[:,:2],axis=1),1,atol=1e-8)
        self.assertGreaterEqual(pts[:,:2].min(),-1e-8)
        self.assertAlmostEqual(pts[:,2].min(),-4); self.assertAlmostEqual(pts[:,2].max(),7)

    def test_model_faces_and_colored_selection(self):
        data=load_sample('000000'); geometry=topology_meshes('000000')
        self.assertEqual(len(geometry['faces']),data['fag_statistics']['face_count'])
        expected={i for r in data['feature_spaces'][0]['roles'] for i in r['source_face_ids']}
        fig,_,_=make_figure('000000','0',['faces'])
        actual={t.meta['face_id'] for t in fig.data if t.meta and t.meta['kind']=='topology'}
        self.assertEqual(expected,actual)
        self.assertFalse(any(t.meta and t.meta['kind']=='feature_space' for t in fig.data))

    def test_all_samples_available_and_options_valid(self):
        self.assertEqual(len(sample_index()),100)
        for name in sample_index():
            options,first=feature_options(name)
            self.assertIn(first,[o['value'] for o in options])
            self.assertEqual(len(options),1+len(load_sample(name)['feature_spaces']))

    def test_each_observed_class_can_draw_boundaries(self):
        representatives={}
        for name in sample_index():
            for f in load_sample(name)['feature_spaces']:
                if planar_feasible(f): representatives.setdefault(f['name'],(name,f['id']))
        self.assertEqual(len(representatives),13)
        for name,(sample,fid) in representatives.items():
            with self.subTest(feature=name):
                b=feature_boundaries(sample,fid,.3)
                self.assertTrue(any(p['mesh']['triangles'] for p in b['patches']))
                for p in b['patches']:
                    if p['mesh']['vertices']:
                        self.assertTrue(np.isfinite(p['mesh']['vertices']).all())

    def test_inconsistent_extraction_is_reported_not_fabricated(self):
        boundary=feature_boundaries('000004',0,.3)
        self.assertFalse(boundary['planar_feasible'])
        self.assertFalse(any(p['mesh']['triangles'] for p in boundary['patches']))
        fig,status,note=make_figure('000004','0',['faces','spaces'])
        self.assertIn('法向约束互斥',note)
        self.assertTrue(any(t.meta and t.meta['kind']=='topology' for t in fig.data))

    def test_dash_layout_and_real_callback(self):
        app=make_app(); client=app.server.test_client()
        self.assertEqual(client.get('/').status_code,200)
        self.assertEqual(client.get('/_dash-layout').status_code,200)
        key=next(k for k in app.callback_map if 'scene.figure' in k)
        response=client.post('/_dash-update-component',json=dict(output=key,
            outputs=[dict(id='scene',property='figure'),dict(id='status',property='children'),dict(id='note',property='children')],
            inputs=[dict(id='sample',property='value',value='000000'),dict(id='feature',property='value',value='0'),
                    dict(id='layers',property='value',value=['spaces','normals']),dict(id='extension',property='value',value=.3),dict(id='opacity',property='value',value=.25)],
            state=[],changedPropIds=['feature.value']))
        self.assertEqual(response.status_code,200)
        result=response.get_json()['response']
        self.assertIn('000000.step',result['status']['children'])
        self.assertTrue(result['scene']['figure']['data'])


if __name__=='__main__': unittest.main(verbosity=2)
