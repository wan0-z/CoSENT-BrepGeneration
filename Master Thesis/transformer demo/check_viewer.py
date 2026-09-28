"""Build all 100 display caches and audit oriented-plane consistency."""
import collections
import io
import json
import time
import unittest

from feature_viewer_geometry import ROOT, load_sample, planar_feasible, sample_index, topology_meshes


def main():
    start=time.monotonic(); rows=[]; inconsistent=[]
    for sample in sample_index():
        data=load_sample(sample); geometry=topology_meshes(sample)
        for feature in data['feature_spaces']:
            if not planar_feasible(feature):
                inconsistent.append(dict(sample=sample,feature_id=feature['id'],name=feature['name']))
        rows.append(dict(sample=sample,faces=len(geometry['faces']),triangles=sum(len(f['triangles']) for f in geometry['faces'])))
    capture=io.StringIO()
    tests=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_feature_viewer'))
    report=dict(samples=len(rows),tests_passed=tests.testsRun if tests.wasSuccessful() else None,
                topology=rows,inconsistent_oriented_plane_instances=inconsistent,
                inconsistent_by_class=dict(collections.Counter(f['name'] for f in inconsistent)),
                seconds=time.monotonic()-start,
                scope='Original STEP face meshes checked for all samples. Curved clipping is a display approximation, not full Boolean certification.')
    (ROOT/'logs/viewer_tests.txt').write_text(capture.getvalue())
    (ROOT/'logs/viewer_audit.json').write_text(json.dumps(report,indent=2))
    print(capture.getvalue())
    print(json.dumps({k:v for k,v in report.items() if k not in ('topology','inconsistent_oriented_plane_instances')},indent=2))
    if not tests.wasSuccessful(): raise RuntimeError('Viewer tests failed')


if __name__=='__main__': main()
