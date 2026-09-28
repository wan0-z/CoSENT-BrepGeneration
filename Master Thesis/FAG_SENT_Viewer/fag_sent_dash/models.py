from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class FagEdge:
    edge_id: int
    u: int
    v: int
    shared_edges: int = 1

    @property
    def key(self) -> tuple[int, int]:
        return tuple(sorted((self.u, self.v)))


@dataclass
class PartRecord:
    json_path: Path
    step_path: Path
    nodes: list[int]
    edges: list[FagEdge]
    raw: dict[str, Any]

    @property
    def part_id(self) -> str:
        return self.json_path.stem

    def edge_id_between(self, a: int, b: int) -> int | None:
        key = tuple(sorted((a, b)))
        for e in self.edges:
            if e.key == key:
                return e.edge_id
        return None


def load_part_json(path: str | Path) -> PartRecord:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))

    # Native MFInstSeg/AAG format used by Generation_Viewer/data/graphs:
    # [part_name, {"graph": {"edges": [[src...], [dst...]], "num_nodes": N}, ...}]
    if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], dict):
        part_name = str(data[0])
        graph = data[1].get("graph", {})
        node_count = int(graph.get("num_nodes", 0))
        nodes = list(range(1, node_count + 1))
        edge_arrays = graph.get("edges", [[], []])
        sources = edge_arrays[0] if len(edge_arrays) > 0 else []
        targets = edge_arrays[1] if len(edge_arrays) > 1 else []
        # The dataset stores both directions. Collapse them into one undirected
        # FAG adjacency and convert its 0-based face IDs to the 1-based IDs used
        # by the STEP tessellation cache.
        pairs = sorted({tuple(sorted((int(u), int(v)))) for u, v in zip(sources, targets) if int(u) != int(v)})
        edges = [FagEdge(index + 1, u + 1, v + 1, 1) for index, (u, v) in enumerate(pairs)]
        step_path = _find_matching_step(part_name)
        normalized_raw = {"source_schema": "mfinstseg_aag", "source": data}
        return PartRecord(p.resolve(), step_path, nodes, edges, normalized_raw)

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported FAG JSON format: {p}")

    raw_nodes = data.get("nodes", list(range(1, int(data.get("n_faces", data.get("face_count", 0))) + 1)))
    if raw_nodes and isinstance(raw_nodes[0], dict):
        face_ids = [int(node.get("face_id", index)) for index, node in enumerate(raw_nodes)]
        offset = 1 if face_ids and min(face_ids) == 0 else 0
        nodes = [face_id + offset for face_id in face_ids]
    else:
        nodes = [int(x) for x in raw_nodes]

    raw_edges = data.get("edges", [])
    edges = []
    node_offset = 1 if nodes and min(nodes) == 1 and raw_edges and "source" in raw_edges[0] else 0
    for index, edge in enumerate(raw_edges):
        u = int(edge.get("u", edge.get("source"))) + node_offset
        v = int(edge.get("v", edge.get("target"))) + node_offset
        shared = int(edge.get("shared_edges", len(edge.get("relations", [])) or 1))
        edges.append(FagEdge(index + 1, u, v, shared))

    raw_step_path = data.get("step_path")
    step_path = Path(raw_step_path) if raw_step_path else _find_matching_step(p.stem)
    if raw_step_path and (not step_path.is_absolute() or not step_path.exists()):
        local_match = _find_matching_step(Path(str(raw_step_path).replace("\\", "/")).stem)
        step_path = local_match if local_match.exists() else (p.parent / step_path).resolve()
    return PartRecord(json_path=p.resolve(), step_path=step_path, nodes=nodes, edges=edges, raw=data)


def _find_matching_step(part_name: str) -> Path:
    workspace = Path(__file__).resolve().parents[2]
    steps_dir = workspace / "Generation_Viewer" / "data" / "steps"
    for suffix in (".step", ".stp", ".STEP", ".STP"):
        candidate = steps_dir / f"{part_name}{suffix}"
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Cannot find matching STEP for {part_name} in {steps_dir}")


def list_json_files(json_dir: str | Path) -> list[Path]:
    d = Path(json_dir)
    return sorted([p for p in d.glob("*.json") if p.is_file()])
