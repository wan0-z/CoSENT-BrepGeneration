# Feature-space seed viewer

Run `python app.py` in this directory and open `http://127.0.0.1:8051`.

The viewer reads `../Seed_Extractor/data/feature_space_seeds.json` without modifying it. The left panel shows geometry; the right panel shows the SAG with the JSON's five-bit edge labels. Supporting surfaces, trimmed feature faces, and collision volumes are independent overlay layers. Plotly `uirevision` preserves the camera while layers are toggled.

Green layers contain only seed surfaces, not artificial bounding-box caps. Red, opaque collision volumes include closure caps. The finite illustration extends 10% of the trimmed extent in open directions only; fixed floors, back walls, and other closed seed boundaries do not move. This finite collision envelope is not the mathematical infinite recession cone. Cylinder tessellation does not introduce additional surface nodes. Circular-end pocket has four side surfaces plus one bottom, hence five nodes.

Run regression checks with `python -m unittest discover -s . -p 'test_*.py'`. After changing the code, restart the server (debug auto-reload is disabled).
