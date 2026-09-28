from __future__ import annotations

from dataclasses import dataclass
import random
import networkx as nx
from .models import PartRecord


@dataclass
class SentStep:
    segment_id: int
    face_id: int
    frontier_faces: list[int]
    prev_face_id: int | None = None
    fag_edge_id: int | None = None


@dataclass
class SentResult:
    steps: list[SentStep]
    trails: list[list[SentStep]]

    @property
    def face_order(self) -> list[int]:
        return [s.face_id for s in self.steps]

    @property
    def alternating_sequence(self) -> list[dict]:
        """face-edge-face-edge... sequence used by the generation frame."""
        seq: list[dict] = []
        for i, s in enumerate(self.steps):
            if i == 0:
                seq.append({"type": "face", "id": s.face_id, "segment": s.segment_id})
            else:
                if s.fag_edge_id is not None:
                    seq.append({"type": "edge", "id": s.fag_edge_id, "from": s.prev_face_id, "to": s.face_id, "segment": s.segment_id})
                seq.append({"type": "face", "id": s.face_id, "segment": s.segment_id})
        return seq


def build_graph(part: PartRecord) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(part.nodes)
    for e in part.edges:
        g.add_edge(e.u, e.v, edge_id=e.edge_id, shared_edges=e.shared_edges)
    return g


def sample_sent(part: PartRecord, seed: int = 0) -> SentResult:
    """Implements Algorithm 1 Causal and Hamiltonian SENT Sampling.

    Nodes are FAG faces. A graph edge is the BREP edge shared by two faces.
    The returned steps record each trail segment and the associated FAG edge id.
    """
    rng = random.Random(seed)
    g = build_graph(part)
    unvisited = set(part.nodes)
    if not unvisited:
        return SentResult([], [])

    v = rng.choice(sorted(unvisited))
    unvisited.remove(v)
    trails: list[list[SentStep]] = []
    current: list[SentStep] = []
    segment = 1

    # first neighborhood trail, equivalent to t <- [(v, empty)] initially
    current.append(SentStep(segment, v, [], None, None))

    while unvisited:
        nbr_unvisited = sorted(set(g.neighbors(v)) & unvisited)
        if not nbr_unvisited:
            trails.append(current)
            current = []
            v = rng.choice(sorted(unvisited))
            unvisited.remove(v)
            frontier = sorted(set(g.neighbors(v)) & (set(part.nodes) - unvisited))
            segment += 1
            current.append(SentStep(segment, v, frontier, None, None))
        else:
            u = rng.choice(nbr_unvisited)
            unvisited.remove(u)
            frontier = sorted((set(g.neighbors(u)) - {v}) & (set(part.nodes) - unvisited))
            eid = g.edges[v, u].get("edge_id")
            segment += 1
            current.append(SentStep(segment, u, frontier, v, int(eid) if eid is not None else None))
            v = u

    if current:
        trails.append(current)
    steps = [s for tr in trails for s in tr]
    return SentResult(steps=steps, trails=trails)
