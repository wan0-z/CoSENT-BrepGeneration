"""Verify saved corpus, causal model tests, checkpoints and TensorBoard events."""
import argparse
import hashlib
import io
import json
import math
import unittest
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='demo_v2')
    args=parser.parse_args()
    output=ROOT/'logs'; output.mkdir(exist_ok=True)
    capture=io.StringIO()
    result=unittest.TextTestRunner(stream=capture,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_demo'))
    (output/'tests.txt').write_text(capture.getvalue())
    print(capture.getvalue())
    if not result.wasSuccessful(): raise RuntimeError('Method tests failed')
    run=ROOT/'checkpoints'/args.run
    metrics=json.loads((run/'metrics.json').read_text())
    final=metrics['history'][-1]
    event=EventAccumulator(str(ROOT/'runs'/args.run),size_guidance={'scalars':0}).Reload()
    scalars={}
    for tag in event.Tags()['scalars']:
        rows=event.Scalars(tag)
        assert all(math.isfinite(r.value) for r in rows),tag
        scalars[tag]=dict(count=len(rows),first_step=rows[0].step,last_step=rows[-1].step,
                          first_value=rows[0].value,last_value=rows[-1].value)
    required=['train/step_loss','train/step_accuracy','train/topology_loss','train/vertex_loss',
              'train/edge_bbox_loss','train/surface_loss','validation/loss','validation/accuracy']
    assert all(k in scalars for k in required)
    assert scalars['train/step_loss']['last_step']==final['step']
    assert abs(scalars['validation/loss']['last_value']-final['validation']['loss'])<1e-5
    assert len(metrics['sample_visits'])==90 and min(metrics['sample_visits'].values())>0
    checkpoint=torch.load(run/'last.pt',map_location='cpu')
    assert checkpoint['step']==final['step']
    vocab=json.loads((ROOT/'data/vocabulary.json').read_text())
    assert json.loads((run/'vocabulary.json').read_text())==vocab
    assert checkpoint['vocab_sha256']==hashlib.sha256((ROOT/'data/vocabulary.json').read_bytes()).hexdigest()
    report=dict(run=args.run,tests_passed=result.testsRun,training_samples_visited=90,
                steps=final['step'],vocabulary_size=len(vocab['tokens']),
                initial_validation=metrics['initial_validation'],final_validation=final['validation'],
                scalars=scalars,checkpoint_and_vocabulary_consistent=True)
    (output/f'{args.run}_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='scalars'},indent=2))


if __name__=='__main__': main()
