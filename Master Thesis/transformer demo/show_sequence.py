"""Export a readable XML-like token stream without touching training data."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample",default="000000",nargs="?"); ap.add_argument("--unconditional",action="store_true")
    args=ap.parse_args()
    variants=json.loads((ROOT/"data/sequences"/f"{args.sample}.json").read_text())
    seq=variants["unconditional" if args.unconditional else "conditional"]
    boundaries={"<FEATURE>","<CONDITION_SURFACE>","<SURFACE>","<COEDGE>","<NEW_POINT>","<MATE>",
                "<EDGE_BBOX>","<SURFACE_GEOMETRY>","<LOOP_ORDER>","<CONTEXT>","<END_MODEL>"}
    lines=[]; current=[]
    for t in seq["tokens"]:
        if t in boundaries or len(" ".join(current))>140:
            if current: lines.append(" ".join(current))
            current=[]
        current.append(t)
    if current: lines.append(" ".join(current))
    out=ROOT/"examples"/f"{args.sample}.tokens.txt"; out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines)+"\n"); print(out)


if __name__=="__main__": main()
