from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from fag_sent_dash.models import list_json_files, load_part_json
    from fag_sent_dash.step_cache import build_part_cache
else:
    from .models import list_json_files, load_part_json
    from .step_cache import build_part_cache


APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_JSON_DIR = APP_DIR.parent / "Generation_Viewer" / "data" / "graphs"
DEFAULT_CACHE_DIR = APP_DIR / "web_cache"


def select_jsons(json_dir: str | Path, seed: int, n: int) -> list[Path]:
    files = list_json_files(json_dir)
    rng = random.Random(seed)
    if len(files) <= n:
        return files
    return sorted(rng.sample(files, n))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", default=str(DEFAULT_JSON_DIR))
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    selected = select_jsons(args.json_dir, args.seed, args.n)
    print(f"Selected {len(selected)} JSON files")
    for idx, p in enumerate(selected, 1):
        try:
            part = load_part_json(p)
            out = build_part_cache(part, args.cache_dir, force=args.force)
            print(f"[{idx}/{len(selected)}] OK {p.name} -> {out}")
        except Exception as exc:
            print(f"[{idx}/{len(selected)}] FAIL {p}: {exc}")


if __name__ == "__main__":
    main()
