"""Viewer-only regression tests; no extraction data is modified."""
import unittest
import numpy as np
import app


def vertices(traces):
    return np.concatenate([np.column_stack([t.x, t.y, t.z]) for t in traces])


class GeometryTests(unittest.TestCase):
    def test_all_seed_layers(self):
        for name, feature in app.FEATURES.items():
            with self.subTest(name=name):
                layers = app.build_layers(feature)
                self.assertEqual(len(layers['surfaces']), len(feature['nodes']))
                self.assertEqual(len(layers['trimmed']), len(feature['nodes']))
                self.assertTrue(all(t.opacity == 1 for t in layers['recession']))
                self.assertTrue(np.isfinite(vertices(sum(layers.values(), []))).all())
                fig, sag, _ = app.update_view(name, list(layers))
                codes = [a.text for a in sag.layout.annotations][:len(feature['edges'])]
                self.assertEqual(codes, [e['code'] for e in feature['edges']])
                self.assertEqual(fig.layout.scene.uirevision, 'seed-camera')

    def test_planar_relations_match_sag(self):
        for name, feature in app.FEATURES.items():
            spec = app.planar_spec(name)
            if spec is None:
                continue
            for edge in feature['edges']:
                a, b = edge['nodes']
                dot = float(spec[0][a][0] @ spec[0][b][0])
                parallel = abs(abs(dot)-1) < 1e-8
                opposite = parallel and dot < 0
                perpendicular = abs(dot) < 1e-8
                self.assertEqual('00'+''.join(str(int(v)) for v in (parallel, opposite, perpendicular)), edge['code'], name)

    def test_blind_bottoms_do_not_move(self):
        names = ['2sides_through_step', 'triangular_pocket', 'rectangular_pocket',
                 '6sides_pocket', 'blind_hole', 'circular_blind_step',
                 'rectangular_blind_slot', 'rectangular_blind_step',
                 'circular_end_pocket', 'v_circular_end_blind_slot',
                 'h_circular_end_blind_slot', 'Oring']
        for name in names:
            layers = app.build_layers(app.FEATURES[name])
            trimmed, red = vertices(layers['trimmed']), vertices(layers['recession'])
            self.assertAlmostEqual(red[:,2].min(), trimmed[:,2].min(), msg=name)
            self.assertAlmostEqual(red[:,2].max(), trimmed[:,2].max()+.1*np.ptp(trimmed[:,2]), msg=name)

    def test_through_axis_and_v_legs(self):
        layers = app.build_layers(app.FEATURES['triangular_through_slot'])
        trimmed, red = vertices(layers['trimmed']), vertices(layers['recession'])
        self.assertAlmostEqual(red[:,2].min(), -.78)
        self.assertAlmostEqual(red[:,2].max(), .78)
        self.assertAlmostEqual(red[:,1].min(), -.5)
        self.assertAlmostEqual(red[:,1].max()+.5, 1.1*(trimmed[:,1].max()+.5))
        self.assertAlmostEqual(red[:,0].max(), 1.1*trimmed[:,0].max())

    def test_back_wall_stays_fixed(self):
        layers = app.build_layers(app.FEATURES['rectangular_blind_slot'])
        red = vertices(layers['recession'])
        np.testing.assert_allclose(red.min(axis=0), [-.65, -.65, -.65])
        np.testing.assert_allclose(red.max(axis=0), [.65, .78, .78])


if __name__ == '__main__':
    unittest.main()
