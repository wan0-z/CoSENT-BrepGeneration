# Unlabelled STEP feature extraction

This folder applies `Seed_Extractor/extractor_one_step_space.py` to the unlabelled
ECCV 2026 target STEP files. Before extraction, adjacent faces on the same
cylindrical support are selectively unified. Non-cylindrical face boundaries are
protected, so the preprocessing does not globally merge planar feature faces.

Use the existing `aagnet-mac` Conda environment:

```bash
conda run -n aagnet-mac python "extractor ulabeled/extract_batch.py"
conda run -n aagnet-mac python "extractor ulabeled/dashboard.py"
```

The default extraction target is 50 samples. To grow the existing set to 2,000
without recalculating the first 50:

```bash
conda run -n aagnet-mac python "extractor ulabeled/extract_batch.py" \
  --sample-number 2000 --no-redo-exist
```

To recompute all 2,000 selected samples:

```bash
conda run -n aagnet-mac python "extractor ulabeled/extract_batch.py" \
  --sample-number 2000 --redo-exist
```

The Dashboard is available at `http://127.0.0.1:8051` while it is running. Its
first dropdown contains the 20 extracted feature types. The second dropdown is
restricted to STEP files containing the selected feature.

