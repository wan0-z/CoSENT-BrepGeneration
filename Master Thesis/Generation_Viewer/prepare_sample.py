"""CLI: convert one named sample into a CoAG/CoSENT Dash web cache."""

from __future__ import annotations

import argparse
import json

from coag_pipeline import build_sample_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", help="Sample basename, e.g. 20221123_142528_10")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic CoSENT/layout seed")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing cache")
    args = parser.parse_args()
    output = build_sample_cache(args.sample, seed=args.seed, force=args.force)
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(f"Web cache: {output}")
    print(
        f"faces={payload['n_faces']} brep_edges={payload['n_brep_edges']} "
        f"coedges={payload['n_coedges']} graph_edges={payload['n_graph_edges']} "
        f"subgraphs={len(payload['subgraphs'])} generation_steps={len(payload['generation_order'])}"
    )


if __name__ == "__main__":
    main()

