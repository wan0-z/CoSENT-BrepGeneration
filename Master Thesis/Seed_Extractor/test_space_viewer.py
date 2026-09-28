"""UI/cache regression checks; no STEP work or external writes."""
import json
import unittest
from pathlib import Path
import app


class SpaceViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(__file__).parent / 'output_space/single/20221123_142528_10/cache.json'
        cls.cache = json.loads(cls.path.read_text())

    def test_overlay_is_cached_closed_mesh(self):
        normal = app.build_3d_figure(self.cache, 'orbit')
        overlay = app.build_3d_figure(self.cache, 'orbit', True)
        self.assertEqual(len(overlay.data) - len(normal.data), len(self.cache['instances']))
        self.assertEqual(normal.layout.scene.uirevision, overlay.layout.scene.uirevision)
        for instance in self.cache['instances']:
            mesh = instance['collision']['mesh']
            self.assertTrue(mesh['triangles'])
            self.assertTrue(all(0 <= i < len(mesh['vertices']) for t in mesh['triangles'] for i in t))
        for trace in overlay.data[len(normal.data):]:
            self.assertEqual(trace.color, '#ff0000')
            self.assertAlmostEqual(trace.opacity, .22)

    def test_metrics_follow_class(self):
        text = app.accuracy_summary(self.cache, 'space', '21')
        self.assertIn('51.85%', text)
        self.assertIn('circular_blind_step 0.00%', text)
        self.assertIn('rectangular_blind_slot 100.00%', app.accuracy_summary(self.cache, 'space', '17'))

    def test_dash_layout_and_callback(self):
        client = app.app.server.test_client()
        self.assertEqual(client.get('/_dash-layout').status_code, 200)
        result = app.update_all(str(self.path), 'pan', ['show'], '21', 'space')
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].layout.scene.dragmode, 'pan')
        key, callback = next((k, v) for k, v in app.app.callback_map.items() if 'part-viewer.figure' in k)
        self.assertIn('recession-layer', [i['id'] for i in callback['inputs']])


if __name__ == '__main__':
    unittest.main()
