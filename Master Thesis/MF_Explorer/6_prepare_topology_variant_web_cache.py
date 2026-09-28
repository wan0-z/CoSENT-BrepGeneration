# -*- coding: utf-8 -*-
"""
Generate topology-variant history web cache and augment web_cache/samples_index.json.

Run order:
  1) python view_machining_feature_graph.py
  2) python check_topology_variants_surface_aware.py
  3) python prepare_mftinstseg_ui_cache.py
  4) python prepare_topology_variant_web_cache_surface_layout.py

This version is designed for the surface-aware variant definition.

What this script adds into each subgraph entry in samples_index.json:
  topology_variant: {
    relation: "main" | "variant" | "non_variant" | "unknown",
    variant_type: "edge_split" | "surface_aware_node_split" | ...,
    reason: "...",
    reason_label: "...",
    green_check: true/false,
    history_steps: [
      {
        title,
        kind,
        image,
        image_simple,
        image_detail,
        note,
        main_faces,
        surface_groups,
        generalized_edges
      }, ...
    ]
  }

Compatibility with app.py:
  The old fields history_steps[*].image, image_simple, image_detail, title, kind,
  and note are preserved. If your app.py only renders those image fields, no app.py
  change is normally needed.

Core visualization rule for surface-aware variants:
  Every history PNG uses the same main-type layout. In each view, one face from
  each underlying surface group becomes a large "main node" fixed at the main-type
  position. Other faces on the same surface are drawn as smaller satellites around
  that main node. Views are generated greedily until every feature face has appeared
  at least once as a large main node.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------------------------------------------------------
# Import helpers from your topology checker.
#
# The original cache script imported from expired_tools.check_topology_variants.
# This file keeps that compatible, but also tries the surface-aware checker first.
# -----------------------------------------------------------------------------
try:  # preferred if you placed the new checker in expired_tools/
    from expired_tools.check_topology_variants_surface_aware import (  # type: ignore
        FACE_CATEGORIES,
        build_full_graph,
        build_generalized_feature_graph,
        load_json,
        parse_cls,
        parse_feature_instances,
        safe_name,
        save_json,
        topology_type_json_to_graph,
        unwrap_graph_json,
        unwrap_label_json,
    )
except Exception:
    try:  # original project layout used by your pasted script
        from expired_tools.check_topology_variants import (  # type: ignore
            FACE_CATEGORIES,
            build_full_graph,
            build_generalized_feature_graph,
            load_json,
            parse_cls,
            parse_feature_instances,
            safe_name,
            save_json,
            topology_type_json_to_graph,
            unwrap_graph_json,
            unwrap_label_json,
        )
    except Exception:
        try:  # if scripts are in the same folder
            from check_topology_variants_surface_aware import (  # type: ignore
                FACE_CATEGORIES,
                build_full_graph,
                build_generalized_feature_graph,
                load_json,
                parse_cls,
                parse_feature_instances,
                safe_name,
                save_json,
                topology_type_json_to_graph,
                unwrap_graph_json,
                unwrap_label_json,
            )
        except Exception:
            from utils.check_topology_variants import (  # type: ignore
                FACE_CATEGORIES,
                build_full_graph,
                build_generalized_feature_graph,
                load_json,
                parse_cls,
                parse_feature_instances,
                safe_name,
                save_json,
                topology_type_json_to_graph,
                unwrap_graph_json,
                unwrap_label_json,
            )


LAYOUT_SEED = 42
MAX_SURFACE_MAIN_VIEW_CANDIDATES = 50000

RELATION_LABELS = {
    "main": "belong to main type",
    "variant": "belong to main type variant",
    "non_variant": "not variant",
    "unknown": "topology variant information not found",
}

REASON_LABELS = {
    "direct_graph_isomorphic_to_main": "Direct feature graph is isomorphic to the main type.",
    "same_node_count_generalized_graph_contains_main": "Same node count as main type; generalized graph contains the main type.",
    "all_nodes_covered": "Every split / extended node can be covered by a generalized main type subset.",
    "all_faces_covered_by_surface_aware_main": "Every split face can be covered by a surface-aware main-type subset.",
    "candidate_has_fewer_nodes_than_main": "Candidate has fewer nodes than the main type.",
    "same_node_count_but_generalized_graph_does_not_contain_main": "Same node count, but generalized graph does not contain the main type.",
}

# Deterministic color palette for surface groups. These colors are used only for
# the generated PNG cache, not for analytic charts.
SURFACE_COLORS = [
    "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
    "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
    "#ccebc5", "#ffed6f", "#a6cee3", "#b2df8a", "#fb9a99",
    "#fdbf6f", "#cab2d6", "#ffff99", "#1f78b4", "#33a02c",
]


# =============================================================================
# Basic helpers
# =============================================================================

def normalize_reason(reason: str) -> str:
    if not reason:
        return ""
    if reason in REASON_LABELS:
        return REASON_LABELS[reason]
    if reason.startswith("face_") and "_not_covered:" in reason:
        return f"At least one face is not covered by any surface-aware main type subset ({reason})."
    if reason.startswith("node_") and "_not_covered:" in reason:
        return f"At least one node is not covered by any generalized main type subset ({reason})."
    if reason.startswith("surface_group_count_"):
        return f"Surface group count does not match the main type node count ({reason})."
    if reason.startswith("surface_aware_not_covered_after_"):
        return f"No surface-aware main type cover was found ({reason})."
    if reason.startswith("surface_aware_combination_limit_exceeded_"):
        return f"Search stopped because the surface-aware combination limit was exceeded ({reason})."
    if reason.startswith("not_covered_after_"):
        return f"No generalized main type cover was found ({reason})."
    if reason.startswith("combination_limit_exceeded_"):
        return f"Search stopped because the combination limit was exceeded ({reason})."
    return reason


def rel_cache_path(path: Path, cache_dir: Path) -> str:
    return str(path.relative_to(cache_dir)).replace("\\", "/")


def load_index(cache_dir: Path) -> Dict[str, Any]:
    index_path = cache_dir / "samples_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Cannot find {index_path}. Run prepare_mftinstseg_ui_cache.py first.")
    return load_json(index_path)


def load_classification_records(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find classification json: {path}. Run check_topology_variants_surface_aware.py first.")
    records = load_json(path)
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in records:
        result[(str(r["sample_name"]), int(r["instance_id"]))] = r
    return result


def graph_node_label(G: nx.Graph, n: int) -> str:
    cat_id = int(G.nodes[n].get("category_id", -1))
    cat_name = FACE_CATEGORIES[cat_id] if 0 <= cat_id < len(FACE_CATEGORIES) else "unknown"
    return f"{n}\n{cat_id}:{cat_name}"


def node_label_local(G: nx.Graph, n: int) -> Tuple[str, int]:
    a = G.nodes[n]
    return str(a.get("role", "feature")), int(a.get("category_id", -1))


def edge_label_local(G: nx.Graph, u: int, v: int) -> str:
    return str(G.edges[u, v].get("edge_role", "internal"))


def edge_key(u: int, v: int) -> Tuple[int, int]:
    return tuple(sorted((int(u), int(v))))


def sorted_int_keys(d: Dict[Any, Any]) -> List[Any]:
    return sorted(d.keys(), key=lambda x: int(x))


# =============================================================================
# Main-type embedding helpers
# =============================================================================

def main_embedding_mapping(main_G: nx.Graph, candidate_G: nx.Graph) -> Optional[Dict[int, int]]:
    """
    Return one embedding main_node -> candidate_node.

    Relaxed rule:
      - every main node maps to a distinct candidate node with same role/category;
      - every main edge must exist in candidate with same edge_role;
      - candidate may have extra edges.
    """
    if main_G.number_of_nodes() > candidate_G.number_of_nodes():
        return None
    if main_G.number_of_edges() > candidate_G.number_of_edges():
        return None

    main_nodes = sorted(list(main_G.nodes()), key=lambda n: main_G.degree(n), reverse=True)
    cand_nodes = list(candidate_G.nodes())

    candidates_by_main_node: Dict[int, List[int]] = {}
    for m in main_nodes:
        possible = [c for c in cand_nodes if node_label_local(main_G, m) == node_label_local(candidate_G, c)]
        if not possible:
            return None
        candidates_by_main_node[int(m)] = [int(x) for x in possible]

    mapping: Dict[int, int] = {}
    used: Set[int] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(main_nodes):
            return True
        m = int(main_nodes[idx])
        for c in candidates_by_main_node[m]:
            if c in used:
                continue
            ok = True
            for m_prev, c_prev in mapping.items():
                if main_G.has_edge(m, m_prev):
                    if not candidate_G.has_edge(c, c_prev):
                        ok = False
                        break
                    if edge_label_local(main_G, m, m_prev) != edge_label_local(candidate_G, c, c_prev):
                        ok = False
                        break
            if not ok:
                continue
            mapping[m] = c
            used.add(c)
            if backtrack(idx + 1):
                return True
            used.remove(c)
            del mapping[m]
        return False

    if backtrack(0):
        return dict(mapping)
    return None


def induced_relabel_keep_attrs(G: nx.Graph, nodes: Iterable[int]) -> nx.Graph:
    H = G.subgraph(sorted(int(x) for x in nodes)).copy()
    return nx.convert_node_labels_to_integers(H, ordering="sorted")


def embedding_for_original_face_subset(
    main_G: nx.Graph,
    generalized_G: nx.Graph,
    feature_nodes: List[int],
    subset_original_faces: Sequence[int],
) -> Optional[Dict[int, int]]:
    """
    Return main_node -> original_face_id for one selected original-face subset.
    """
    face_to_rel = {int(face): i for i, face in enumerate(sorted(int(x) for x in feature_nodes))}
    try:
        subset_rel = sorted(face_to_rel[int(face)] for face in subset_original_faces)
    except KeyError:
        return None

    subset_G = induced_relabel_keep_attrs(generalized_G, subset_rel)
    emb_local = main_embedding_mapping(main_G, subset_G)
    if emb_local is None:
        return None

    subset_rel_sorted = sorted(subset_rel)
    result: Dict[int, int] = {}
    for main_node, local_candidate_node in emb_local.items():
        rel_node = subset_rel_sorted[int(local_candidate_node)]
        result[int(main_node)] = int(feature_nodes[rel_node])
    return result


def selected_faces_from_main_to_face(main_to_face: Dict[int, int]) -> List[int]:
    return [int(main_to_face[m]) for m in sorted(main_to_face.keys())]


# =============================================================================
# Generalized edge / skip-path helpers
# =============================================================================

def edge_is_direct_original(full_G: nx.Graph, original_u: int, original_v: int) -> bool:
    return full_G.has_edge(int(original_u), int(original_v))


def shortest_external_path(
    full_G: nx.Graph,
    instance_original_nodes: Set[int],
    original_u: int,
    original_v: int,
) -> Optional[List[int]]:
    """
    Find a shortest path from original_u to original_v whose intermediate nodes
    are outside the current feature instance. Direct edges return None.
    """
    original_u = int(original_u)
    original_v = int(original_v)

    if original_u == original_v:
        return None
    if full_G.has_edge(original_u, original_v):
        return None

    blocked = set(int(x) for x in instance_original_nodes) - {original_u, original_v}
    H = full_G.copy()
    H.remove_nodes_from([n for n in blocked if n in H])

    try:
        return [int(x) for x in nx.shortest_path(H, original_u, original_v)]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def generalized_edge_infos_for_main_view(
    main_G: nx.Graph,
    main_to_face: Dict[int, int],
    full_G: nx.Graph,
    instance_original_nodes: Set[int],
) -> List[Dict[str, Any]]:
    infos: List[Dict[str, Any]] = []
    for mu, mv in sorted(main_G.edges()):
        if int(mu) not in main_to_face or int(mv) not in main_to_face:
            continue
        fu, fv = int(main_to_face[int(mu)]), int(main_to_face[int(mv)])
        if edge_is_direct_original(full_G, fu, fv):
            infos.append({
                "main_edge": [int(mu), int(mv)],
                "faces": [fu, fv],
                "kind": "direct",
                "skipped_nodes": [],
                "path": [fu, fv],
            })
        else:
            path = shortest_external_path(full_G, instance_original_nodes, fu, fv)
            skipped = path[1:-1] if path and len(path) > 2 else []
            infos.append({
                "main_edge": [int(mu), int(mv)],
                "faces": [fu, fv],
                "kind": "skip" if skipped else "generalized",
                "skipped_nodes": [int(x) for x in skipped],
                "path": [int(x) for x in path] if path else [],
            })
    return infos


def format_edge_infos(edge_infos: List[Dict[str, Any]]) -> str:
    if not edge_infos:
        return "No main-type edge information was generated."
    parts = []
    for info in edge_infos:
        faces = info.get("faces", [])
        if info.get("kind") == "direct":
            parts.append(f"{faces[0]}-{faces[1]}: direct")
        else:
            skipped = info.get("skipped_nodes", []) or []
            if skipped:
                parts.append(f"{faces[0]}-{faces[1]}: skip {skipped}")
            else:
                parts.append(f"{faces[0]}-{faces[1]}: generalized/no explicit skip path")
    return "; ".join(parts)


# =============================================================================
# Surface group helpers
# =============================================================================

def surface_groups_from_record_or_fallback(
    cls_record: Dict[str, Any],
    feature_nodes: List[int],
) -> Dict[int, List[int]]:
    """
    Prefer the surface-aware checker field:
      surface_groups_as_original_faces: {gid: [original_face_ids]}

    Fallback for old classification files:
      each face is treated as one surface group.
    """
    raw = cls_record.get("surface_groups_as_original_faces", {}) or {}
    result: Dict[int, List[int]] = {}

    if isinstance(raw, dict) and raw:
        for k, v in raw.items():
            if not isinstance(v, list):
                continue
            faces = sorted(int(x) for x in v if int(x) in set(feature_nodes))
            if faces:
                result[int(k)] = faces

    if result:
        return dict(sorted(result.items(), key=lambda kv: int(kv[0])))

    # Old-file fallback: no surface grouping information.
    return {i: [int(face)] for i, face in enumerate(sorted(int(x) for x in feature_nodes))}


def face_to_surface_group(surface_groups: Dict[int, List[int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for gid, faces in surface_groups.items():
        for f in faces:
            out[int(f)] = int(gid)
    return out


def surface_group_color(gid: int) -> str:
    return SURFACE_COLORS[int(gid) % len(SURFACE_COLORS)]


# =============================================================================
# Drawing helpers
# =============================================================================

def draw_topology_graph(
    G: nx.Graph,
    out_path: Path,
    title: str,
    highlight_nodes: Optional[Set[int]] = None,
    highlight_edges: Optional[Set[Tuple[int, int]]] = None,
    node_note: Optional[Dict[int, str]] = None,
    pos: Optional[Dict[int, Tuple[float, float]]] = None,
) -> Dict[int, Tuple[float, float]]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    highlight_nodes = set(highlight_nodes or set())
    highlight_edges = {edge_key(*e) for e in (highlight_edges or set())}
    node_note = node_note or {}

    if pos is None:
        pos = nx.spring_layout(G, seed=LAYOUT_SEED) if G.number_of_nodes() else {}

    figsize = (max(5.0, math.sqrt(max(G.number_of_nodes(), 1)) * 2.4), 4.8)
    plt.figure(figsize=figsize)

    normal_edges = []
    marked_edges = []
    for u, v in G.edges():
        e = edge_key(u, v)
        if e in highlight_edges:
            marked_edges.append((u, v))
        else:
            normal_edges.append((u, v))

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color="#888888", width=1.6)
    nx.draw_networkx_edges(G, pos, edgelist=marked_edges, edge_color="red", width=2.8)

    node_colors = ["#FFD966" if int(n) in highlight_nodes else "#D9EAF7" for n in G.nodes()]
    node_sizes = [1350 if int(n) in highlight_nodes else 1000 for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=1.2)

    labels = {
        int(n): graph_node_label(G, int(n)) + (f"\n{node_note[int(n)]}" if int(n) in node_note else "")
        for n in G.nodes()
    }
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}


def compute_satellite_position(
    center: Tuple[float, float],
    gid: int,
    idx: int,
    total: int,
    radius: float = 0.23,
) -> Tuple[float, float]:
    """
    Deterministically place same-surface satellites around the selected main node.
    gid changes the angle offset so adjacent surface groups do not all place
    satellites in the same direction.
    """
    if total <= 0:
        return center
    base = (gid * 1.113 + 0.35) % (2.0 * math.pi)
    angle = base + (2.0 * math.pi * idx / max(total, 1))
    r = radius * (1.0 + 0.10 * (idx // 6))
    return (float(center[0] + math.cos(angle) * r), float(center[1] + math.sin(angle) * r))


def draw_surface_aware_main_view(
    main_G: nx.Graph,
    main_pos: Dict[int, Tuple[float, float]],
    main_to_face: Dict[int, int],
    surface_groups: Dict[int, List[int]],
    full_G: nx.Graph,
    instance_original_nodes: List[int],
    out_path: Path,
    title: str,
    show_detail_paths: bool = True,
) -> List[Dict[str, Any]]:
    """
    Draw one history view:
      - selected faces are large main nodes at fixed main-type positions;
      - other faces from the same underlying surface are small satellites;
      - main-type edges are drawn between large nodes;
      - indirect/generalized edges are dashed red and can show skipped nodes.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    instance_original_set = set(int(x) for x in instance_original_nodes)
    f_to_gid = face_to_surface_group(surface_groups)
    selected_faces = set(int(f) for f in main_to_face.values())
    face_to_main = {int(face): int(main_node) for main_node, face in main_to_face.items()}

    edge_infos = generalized_edge_infos_for_main_view(
        main_G=main_G,
        main_to_face=main_to_face,
        full_G=full_G,
        instance_original_nodes=instance_original_set,
    )

    # Build positions.
    pos: Dict[str, Tuple[float, float]] = {}
    labels: Dict[str, str] = {}
    node_colors: List[str] = []
    node_sizes: List[int] = []
    node_edge_colors: List[str] = []
    node_linewidths: List[float] = []
    node_ids: List[str] = []

    # Main selected nodes at fixed main-type positions.
    for main_node in sorted(main_G.nodes()):
        if int(main_node) not in main_to_face:
            continue
        face = int(main_to_face[int(main_node)])
        gid = f_to_gid.get(face, -1)
        node_id = f"F{face}"
        center = main_pos.get(int(main_node), (0.0, 0.0))
        pos[node_id] = (float(center[0]), float(center[1]))
        labels[node_id] = f"M{main_node}\nface {face}\nS{gid}"
        node_ids.append(node_id)
        node_colors.append(surface_group_color(gid))
        node_sizes.append(1700)
        node_edge_colors.append("black")
        node_linewidths.append(2.0)

    # Satellites: same-surface faces that are not selected in this view.
    for gid, faces in sorted(surface_groups.items(), key=lambda kv: int(kv[0])):
        selected_in_group = [f for f in faces if int(f) in selected_faces]
        if selected_in_group:
            anchor_face = int(selected_in_group[0])
            anchor_main = face_to_main[anchor_face]
            anchor_pos = main_pos.get(anchor_main, (0.0, 0.0))
        else:
            # Should not happen for a valid surface-aware main view, but keep a
            # deterministic fallback in case old records are used.
            anchor_pos = (0.0, 0.0)

        satellites = [int(f) for f in sorted(faces) if int(f) not in selected_faces]
        for idx, face in enumerate(satellites):
            node_id = f"F{face}"
            pos[node_id] = compute_satellite_position(anchor_pos, int(gid), idx, len(satellites))
            labels[node_id] = f"face {face}\nS{gid}"
            node_ids.append(node_id)
            node_colors.append(surface_group_color(int(gid)))
            node_sizes.append(760)
            node_edge_colors.append("#555555")
            node_linewidths.append(1.0)

    plt.figure(figsize=(8.6, 6.6))

    # Draw main-type edges using selected face nodes.
    direct_segments = []
    skip_segments = []
    for info in edge_infos:
        fu, fv = [int(x) for x in info.get("faces", [])]
        a, b = f"F{fu}", f"F{fv}"
        if a not in pos or b not in pos:
            continue
        if info.get("kind") == "direct":
            direct_segments.append((a, b, info))
        else:
            skip_segments.append((a, b, info))

    for a, b, _info in direct_segments:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        plt.plot([x1, x2], [y1, y2], color="#555555", linewidth=2.2, solid_capstyle="round", zorder=1)

    for a, b, info in skip_segments:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        plt.plot([x1, x2], [y1, y2], color="red", linewidth=2.3, linestyle="dashed", zorder=1)

        skipped = [int(x) for x in info.get("skipped_nodes", []) or []]
        if show_detail_paths and skipped:
            dx = x2 - x1
            dy = y2 - y1
            norm = math.sqrt(dx * dx + dy * dy) or 1.0
            ox = -dy / norm * 0.14
            oy = dx / norm * 0.14
            chain_points = [(x1, y1)]
            ext_pos: Dict[int, Tuple[float, float]] = {}
            denom = len(skipped) + 1
            for idx, ext_node in enumerate(skipped, start=1):
                t = idx / denom
                x = x1 * (1 - t) + x2 * t + ox
                y = y1 * (1 - t) + y2 * t + oy
                ext_pos[int(ext_node)] = (x, y)
                chain_points.append((x, y))
            chain_points.append((x2, y2))

            for p1, p2 in zip(chain_points, chain_points[1:]):
                plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="red", linewidth=0.9, linestyle="dashed", alpha=0.55, zorder=0)

            nx.draw_networkx_nodes(
                full_G.subgraph(skipped),
                ext_pos,
                nodelist=skipped,
                node_color="#DDDDDD",
                node_size=420,
                edgecolors="#777777",
                linewidths=0.8,
            )
            nx.draw_networkx_labels(full_G.subgraph(skipped), ext_pos, labels={n: str(n) for n in skipped}, font_size=6)

        label = ""
        skipped = [int(x) for x in info.get("skipped_nodes", []) or []]
        if skipped:
            label = "skip: " + ",".join(str(x) for x in skipped)
        elif info.get("kind") != "direct":
            label = "skip path not found"
        if label:
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            plt.text(mx, my, label, color="red", fontsize=7, ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75), zorder=4)

    # Draw faint surface-group links between main node and its satellites.
    for gid, faces in sorted(surface_groups.items(), key=lambda kv: int(kv[0])):
        selected_in_group = [int(f) for f in faces if int(f) in selected_faces]
        if not selected_in_group:
            continue
        anchor = f"F{selected_in_group[0]}"
        if anchor not in pos:
            continue
        ax, ay = pos[anchor]
        for face in faces:
            face = int(face)
            if face in selected_faces:
                continue
            sid = f"F{face}"
            if sid not in pos:
                continue
            sx, sy = pos[sid]
            plt.plot([ax, sx], [ay, sy], color=surface_group_color(int(gid)), linewidth=1.0, linestyle=":", alpha=0.75, zorder=0)

    # Draw nodes and labels. Use matplotlib scatter to allow per-node sizes/colors.
    xs = [pos[n][0] for n in node_ids]
    ys = [pos[n][1] for n in node_ids]
    plt.scatter(xs, ys, s=node_sizes, c=node_colors, edgecolors=node_edge_colors, linewidths=node_linewidths, zorder=3)
    for nid in node_ids:
        x, y = pos[nid]
        plt.text(x, y, labels[nid], fontsize=7, ha="center", va="center", zorder=5)

    # Legend-like surface summary.
    surface_lines = []
    for gid, faces in sorted(surface_groups.items(), key=lambda kv: int(kv[0])):
        surface_lines.append(f"S{gid}: {','.join(str(int(f)) for f in faces)}")
    if surface_lines:
        plt.text(
            0.01,
            0.01,
            "Surface groups\n" + "\n".join(surface_lines[:14]),
            transform=plt.gca().transAxes,
            fontsize=7,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#999999", alpha=0.80),
        )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    return edge_infos


# =============================================================================
# Surface-aware view generation
# =============================================================================

def enumerate_valid_surface_main_views(
    main_G: nx.Graph,
    generalized_G: nx.Graph,
    feature_nodes: List[int],
    surface_groups: Dict[int, List[int]],
    max_candidates: int = MAX_SURFACE_MAIN_VIEW_CANDIDATES,
) -> List[Dict[str, Any]]:
    """
    Enumerate candidate views: choose one face from each surface group, and keep
    only combinations that can embed the main type.

    Each returned item contains:
      main_to_face: {main_node: original_face_id}
      selected_faces: [original_face_id, ...]
    """
    main_n = main_G.number_of_nodes()
    if len(surface_groups) != main_n:
        return []

    groups_sorted = [gid for gid, _faces in sorted(surface_groups.items(), key=lambda kv: int(kv[0]))]
    choices = [sorted(int(f) for f in surface_groups[gid]) for gid in groups_sorted]

    views: List[Dict[str, Any]] = []
    checked = 0
    seen_keys: Set[Tuple[int, ...]] = set()

    for combo in itertools.product(*choices):
        checked += 1
        if checked > max_candidates:
            break
        combo_faces = tuple(sorted(int(f) for f in combo))
        if combo_faces in seen_keys:
            continue
        seen_keys.add(combo_faces)

        main_to_face = embedding_for_original_face_subset(
            main_G=main_G,
            generalized_G=generalized_G,
            feature_nodes=feature_nodes,
            subset_original_faces=list(combo_faces),
        )
        if main_to_face is None:
            continue
        views.append({
            "main_to_face": main_to_face,
            "selected_faces": selected_faces_from_main_to_face(main_to_face),
            "selected_faces_key": tuple(sorted(int(x) for x in main_to_face.values())),
        })

    return views


def choose_minimal_history_views(valid_views: List[Dict[str, Any]], all_faces: Set[int]) -> List[Dict[str, Any]]:
    """
    Greedy set-cover style selection.

    Goal: every original feature face should appear as a large main node at least
    once. At each step, select the valid main view that covers the most faces that
    have not yet appeared as main nodes.
    """
    uncovered = set(int(x) for x in all_faces)
    selected: List[Dict[str, Any]] = []
    used_keys: Set[Tuple[int, ...]] = set()

    while uncovered:
        best: Optional[Dict[str, Any]] = None
        best_gain: Set[int] = set()

        for view in valid_views:
            key = tuple(sorted(int(x) for x in view.get("selected_faces_key", [])))
            if key in used_keys:
                continue
            faces = set(int(x) for x in view.get("selected_faces", []))
            gain = faces & uncovered
            if len(gain) > len(best_gain):
                best = view
                best_gain = gain
            elif best is not None and len(gain) == len(best_gain) and len(gain) > 0:
                # deterministic tie-breaker: lexicographically smaller face tuple
                if key < tuple(sorted(int(x) for x in best.get("selected_faces_key", []))):
                    best = view
                    best_gain = gain

        if best is None or not best_gain:
            break

        key = tuple(sorted(int(x) for x in best.get("selected_faces_key", [])))
        selected.append(best)
        used_keys.add(key)
        uncovered -= set(int(x) for x in best.get("selected_faces", []))

    return selected


def fallback_views_from_coverage(
    main_G: nx.Graph,
    generalized_G: nx.Graph,
    feature_nodes: List[int],
    cls_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Fallback for old classification JSONs or cases where surface-group enumeration
    is unavailable. It uses coverage[*].covering_subset_original_face_ids when present.
    """
    coverage = cls_record.get("coverage", {}) or {}
    views: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, ...]] = set()

    for raw_key in sorted_int_keys(coverage):
        cov = coverage[raw_key] or {}
        subset = cov.get("covering_subset_original_face_ids")
        if subset is None:
            # Old checker used local relabeled nodes as covering_subset.
            local_subset = cov.get("covering_subset") or cov.get("covering_subset_local") or []
            subset = []
            for local_id in local_subset:
                local_id = int(local_id)
                if 0 <= local_id < len(feature_nodes):
                    subset.append(int(feature_nodes[local_id]))
        subset = [int(x) for x in subset or []]
        if not subset:
            continue
        key = tuple(sorted(subset))
        if key in seen:
            continue
        main_to_face = embedding_for_original_face_subset(main_G, generalized_G, feature_nodes, subset)
        if main_to_face is None:
            continue
        views.append({
            "main_to_face": main_to_face,
            "selected_faces": selected_faces_from_main_to_face(main_to_face),
            "selected_faces_key": tuple(sorted(int(x) for x in main_to_face.values())),
        })
        seen.add(key)

    return choose_minimal_history_views(views, set(int(x) for x in feature_nodes))


# =============================================================================
# Old generalized graph drawing retained for edge_split and fallback
# =============================================================================

def spread_non_overlapping_positions(
    pos: Dict[int, Tuple[float, float]],
    fixed_nodes: Optional[Set[int]] = None,
    min_dist: float = 0.22,
    max_iter: int = 250,
) -> Dict[int, Tuple[float, float]]:
    if not pos:
        return {}

    fixed_nodes = set(int(x) for x in (fixed_nodes or set()))
    work = {int(k): [float(v[0]), float(v[1])] for k, v in pos.items()}
    node_ids = list(work.keys())

    def small_direction(a: int, b: int) -> Tuple[float, float]:
        sx = ((a * 37 + b * 17) % 100) / 100.0 - 0.5
        sy = ((a * 53 + b * 29) % 100) / 100.0 - 0.5
        if abs(sx) + abs(sy) < 1e-6:
            sx, sy = 0.31, -0.27
        return sx, sy

    for _ in range(max_iter):
        moved = False
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                ax, ay = work[a]
                bx, by = work[b]
                dx = bx - ax
                dy = by - ay
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 1e-9:
                    dx, dy = small_direction(a, b)
                    dist = math.sqrt(dx * dx + dy * dy)

                if dist >= min_dist:
                    continue

                overlap = min_dist - dist
                ux, uy = dx / dist, dy / dist
                moved = True

                a_fixed = a in fixed_nodes
                b_fixed = b in fixed_nodes

                if not a_fixed and not b_fixed:
                    push = overlap / 2.0 + 1e-3
                    work[a][0] -= ux * push
                    work[a][1] -= uy * push
                    work[b][0] += ux * push
                    work[b][1] += uy * push
                elif a_fixed and not b_fixed:
                    push = overlap + 1e-3
                    work[b][0] += ux * push
                    work[b][1] += uy * push
                elif not a_fixed and b_fixed:
                    push = overlap + 1e-3
                    work[a][0] -= ux * push
                    work[a][1] -= uy * push

        if not moved:
            break

    return {int(k): (float(v[0]), float(v[1])) for k, v in work.items()}


def draw_generalized_history_graph(
    generalized_G: nx.Graph,
    full_G: nx.Graph,
    instance_original_nodes: List[int],
    out_path: Path,
    title: str,
    highlight_nodes: Optional[Set[int]] = None,
    fixed_core_pos: Optional[Dict[int, Tuple[float, float]]] = None,
    show_detail_paths: bool = True,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    highlight_nodes = set(highlight_nodes or set(generalized_G.nodes()))
    instance_original_nodes = [int(x) for x in sorted(instance_original_nodes)]
    instance_original_set = set(instance_original_nodes)
    rel_to_original = {i: original for i, original in enumerate(instance_original_nodes)}

    fixed_core_pos = fixed_core_pos or {}
    fixed_nodes = [n for n in fixed_core_pos if n in generalized_G]
    if generalized_G.number_of_nodes():
        if fixed_nodes:
            pos_core = nx.spring_layout(
                generalized_G,
                seed=LAYOUT_SEED,
                k=max(0.85, 1.8 / max(math.sqrt(generalized_G.number_of_nodes()), 1.0)),
                iterations=200,
                pos={int(k): v for k, v in fixed_core_pos.items() if int(k) in generalized_G},
                fixed=fixed_nodes,
            )
        else:
            pos_core = nx.spring_layout(
                generalized_G,
                seed=LAYOUT_SEED,
                k=max(0.85, 1.8 / max(math.sqrt(generalized_G.number_of_nodes()), 1.0)),
                iterations=200,
            )
        pos_core = spread_non_overlapping_positions(
            {int(k): (float(v[0]), float(v[1])) for k, v in pos_core.items()},
            fixed_nodes=set(int(x) for x in fixed_nodes),
            min_dist=0.24,
            max_iter=300,
        )
    else:
        pos_core = {}

    plt.figure(figsize=(max(7.0, math.sqrt(max(generalized_G.number_of_nodes(), 1)) * 2.8), 5.5))

    direct_edges = []
    indirect_edges = []
    indirect_paths: Dict[Tuple[int, int], List[int]] = {}

    for u, v in generalized_G.edges():
        ou, ov = rel_to_original[int(u)], rel_to_original[int(v)]
        if edge_is_direct_original(full_G, ou, ov):
            direct_edges.append((u, v))
        else:
            indirect_edges.append((u, v))
            p = shortest_external_path(full_G, instance_original_set, ou, ov)
            if p:
                indirect_paths[edge_key(u, v)] = p

    nx.draw_networkx_edges(generalized_G, pos_core, edgelist=direct_edges, edge_color="#555555", width=2.0)
    nx.draw_networkx_edges(generalized_G, pos_core, edgelist=indirect_edges, edge_color="red", width=2.2, style="dashed")

    node_colors = ["#FFD966" if int(n) in highlight_nodes else "#D9EAF7" for n in generalized_G.nodes()]
    node_sizes = [1350 if int(n) in highlight_nodes else 950 for n in generalized_G.nodes()]
    nx.draw_networkx_nodes(generalized_G, pos_core, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=1.2)

    labels = {}
    for n in generalized_G.nodes():
        original = rel_to_original[int(n)]
        cat_id = int(generalized_G.nodes[n].get("category_id", -1))
        cat_name = FACE_CATEGORIES[cat_id] if 0 <= cat_id < len(FACE_CATEGORIES) else "unknown"
        labels[int(n)] = f"r{n}\nface {original}\n{cat_id}:{cat_name}"
    nx.draw_networkx_labels(generalized_G, pos_core, labels=labels, font_size=7)

    if show_detail_paths:
        for (u, v), path in indirect_paths.items():
            if len(path) <= 2:
                continue
            external = path[1:-1]
            p1 = pos_core[u]
            p2 = pos_core[v]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = math.sqrt(dx * dx + dy * dy) or 1.0
            ox = -dy / norm * 0.12
            oy = dx / norm * 0.12

            chain_points = [p1]
            ext_pos = {}
            denom = len(external) + 1
            for idx, ext_node in enumerate(external, start=1):
                t = idx / denom
                x = p1[0] * (1 - t) + p2[0] * t + ox
                y = p1[1] * (1 - t) + p2[1] * t + oy
                ext_pos[ext_node] = (x, y)
                chain_points.append((x, y))
            chain_points.append(p2)

            for a, b in zip(chain_points, chain_points[1:]):
                plt.plot([a[0], b[0]], [a[1], b[1]], color="red", linewidth=0.8, linestyle="dashed", alpha=0.55)

            nx.draw_networkx_nodes(
                full_G.subgraph(external),
                ext_pos,
                nodelist=external,
                node_color="#DDDDDD",
                node_size=420,
                edgecolors="#777777",
                linewidths=0.8,
            )
            nx.draw_networkx_labels(full_G.subgraph(external), ext_pos, labels={n: str(n) for n in external}, font_size=6)

            label_x = (p1[0] + p2[0]) / 2.0 - ox * 0.5
            label_y = (p1[1] + p2[1]) / 2.0 - oy * 0.5
            plt.text(label_x, label_y, f"skip {external}", color="red", fontsize=7, ha="center", va="center")

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# =============================================================================
# Build topology_variant cache for each instance
# =============================================================================

def build_history_for_instance(
    sample_name: str,
    inst: Dict[str, Any],
    cls_record: Dict[str, Any],
    full_G: nx.Graph,
    cls_map: Dict[int, int],
    cache_dir: Path,
) -> Dict[str, Any]:
    relation = str(cls_record.get("relation", "unknown"))
    variant_type = str(cls_record.get("variant_type", "") or "")
    reason = str(cls_record.get("reason", "") or "")

    history_dir = cache_dir / "topology_variant_history" / f"{safe_name(sample_name)}__I{int(inst['instance_id']):03d}"
    steps: List[Dict[str, Any]] = []

    main_json_path = Path(str(cls_record.get("main_type_json", "")))
    main_G: Optional[nx.Graph] = None
    main_pos: Dict[int, Tuple[float, float]] = {}

    if main_json_path.exists():
        main_data = load_json(main_json_path)
        main_G = topology_type_json_to_graph(main_data)
        main_png = history_dir / "00_main_type.png"
        main_pos = draw_topology_graph(
            main_G,
            main_png,
            title=f"Main type | {inst['category_id']}:{inst['category_name']}",
            highlight_nodes=set(main_G.nodes()),
            highlight_edges={edge_key(*e) for e in main_G.edges()},
        )
        main_rel = rel_cache_path(main_png, cache_dir)
        steps.append({
            "title": "Main type",
            "kind": "main_type",
            "image": main_rel,
            "image_simple": main_rel,
            "image_detail": main_rel,
            "note": "Most common topology type used as the fixed anchor layout. All variant history views reuse this layout for their large main nodes.",
        })

    # For main instances, do not show history images, same behavior as your old script.
    if relation == "main":
        return {
            "relation": relation,
            "relation_label": RELATION_LABELS[relation],
            "variant_type": variant_type,
            "reason": reason,
            "reason_label": normalize_reason(reason),
            "green_check": False,
            "history_steps": [],
        }

    if relation == "variant" and main_G is not None:
        feature_nodes = sorted(int(x) for x in inst["nodes"])
        generalized_G = build_generalized_feature_graph(full_G, feature_nodes, cls_map)
        surface_groups = surface_groups_from_record_or_fallback(cls_record, feature_nodes)

        if variant_type == "surface_aware_node_split" or cls_record.get("surface_groups_as_original_faces"):
            valid_views = enumerate_valid_surface_main_views(
                main_G=main_G,
                generalized_G=generalized_G,
                feature_nodes=feature_nodes,
                surface_groups=surface_groups,
            )
            selected_views = choose_minimal_history_views(valid_views, set(feature_nodes))

            # Fallback to checker coverage when enumeration found nothing.
            if not selected_views:
                selected_views = fallback_views_from_coverage(
                    main_G=main_G,
                    generalized_G=generalized_G,
                    feature_nodes=feature_nodes,
                    cls_record=cls_record,
                )

            main_seen: Set[int] = set()
            for view_idx, view in enumerate(selected_views, start=1):
                main_to_face = {int(k): int(v) for k, v in view["main_to_face"].items()}
                selected_faces = set(int(x) for x in main_to_face.values())
                main_seen.update(selected_faces)

                png_simple = history_dir / f"surface_main_view_{view_idx:03d}_simple.png"
                png_detail = history_dir / f"surface_main_view_{view_idx:03d}_detail.png"

                edge_infos_simple = draw_surface_aware_main_view(
                    main_G=main_G,
                    main_pos=main_pos,
                    main_to_face=main_to_face,
                    surface_groups=surface_groups,
                    full_G=full_G,
                    instance_original_nodes=feature_nodes,
                    out_path=png_simple,
                    title=f"Surface-aware main view {view_idx} | selected faces {sorted(selected_faces)}",
                    show_detail_paths=False,
                )
                edge_infos_detail = draw_surface_aware_main_view(
                    main_G=main_G,
                    main_pos=main_pos,
                    main_to_face=main_to_face,
                    surface_groups=surface_groups,
                    full_G=full_G,
                    instance_original_nodes=feature_nodes,
                    out_path=png_detail,
                    title=f"Surface-aware main view {view_idx} | selected faces {sorted(selected_faces)}",
                    show_detail_paths=True,
                )
                edge_infos = edge_infos_detail or edge_infos_simple

                steps.append({
                    "title": f"Surface-aware main view {view_idx}",
                    "kind": "surface_aware_main_view",
                    "image": rel_cache_path(png_detail, cache_dir),
                    "image_simple": rel_cache_path(png_simple, cache_dir),
                    "image_detail": rel_cache_path(png_detail, cache_dir),
                    "note": (
                        "Large nodes form one main-type topology and keep the fixed main-type positions. "
                        "Small surrounding nodes are other faces from the same underlying surface. "
                        "A red dashed edge means the generalized connection skips external nodes. "
                        f"Main faces in this view: {sorted(selected_faces)}. "
                        f"Edges: {format_edge_infos(edge_infos)}"
                    ),
                    "main_faces": sorted(int(x) for x in selected_faces),
                    "surface_groups": {str(g): [int(x) for x in faces] for g, faces in surface_groups.items()},
                    "generalized_edges": edge_infos,
                })

            # If some face never became a main node, make this explicit in metadata.
            missing = sorted(set(feature_nodes) - main_seen)
            if missing and steps:
                steps[-1]["note"] += f" Uncovered-as-main faces after greedy view selection: {missing}."

        else:
            # Same-node-count edge split or old variant records: keep old generalized graph view.
            current_png_simple = history_dir / "01_generalized_equivalent_simple.png"
            current_png_detail = history_dir / "01_generalized_equivalent_detail.png"
            fixed_pos: Dict[int, Tuple[float, float]] = {}
            emb = main_embedding_mapping(main_G, generalized_G)
            if emb:
                fixed_pos = {int(cand): main_pos[int(main)] for main, cand in emb.items() if int(main) in main_pos}

            draw_generalized_history_graph(
                generalized_G=generalized_G,
                full_G=full_G,
                instance_original_nodes=feature_nodes,
                out_path=current_png_simple,
                title="Generalized main-type equivalent | direct edges suppress indirect paths",
                highlight_nodes=set(generalized_G.nodes()),
                fixed_core_pos=fixed_pos,
                show_detail_paths=False,
            )
            draw_generalized_history_graph(
                generalized_G=generalized_G,
                full_G=full_G,
                instance_original_nodes=feature_nodes,
                out_path=current_png_detail,
                title="Generalized main-type equivalent | direct edges suppress indirect paths",
                highlight_nodes=set(generalized_G.nodes()),
                fixed_core_pos=fixed_pos,
                show_detail_paths=True,
            )
            steps.append({
                "title": "Generalized equivalent current instance",
                "kind": variant_type or "edge_split",
                "image": rel_cache_path(current_png_detail, cache_dir),
                "image_simple": rel_cache_path(current_png_simple, cache_dir),
                "image_detail": rel_cache_path(current_png_detail, cache_dir),
                "note": "Simple mode shows coarse red skip edges. Detail mode expands skipped paths with gray external nodes and thin dashed red segments.",
            })

    green_check = relation == "non_variant" and reason == "candidate_has_fewer_nodes_than_main"
    return {
        "relation": relation,
        "relation_label": RELATION_LABELS.get(relation, relation),
        "variant_type": variant_type,
        "reason": reason,
        "reason_label": normalize_reason(reason),
        "green_check": green_check,
        "history_steps": steps if relation == "variant" else [],
        "surface_groups_as_original_faces": cls_record.get("surface_groups_as_original_faces", {}),
        "surface_group_count": cls_record.get("surface_group_count", ""),
    }


def augment_sample(
    sample_meta: Dict[str, Any],
    records_by_key: Dict[Tuple[str, int], Dict[str, Any]],
    cache_dir: Path,
) -> Dict[str, Any]:
    sample_name = str(sample_meta["sample_name"])
    graph_path = Path(str(sample_meta["graph_json"]))
    label_path = Path(str(sample_meta["label_json"]))

    raw_graph_obj = load_json(graph_path)
    _, graph_data = unwrap_graph_json(raw_graph_obj)
    raw_label_obj = load_json(label_path)
    _, label_data = unwrap_label_json(raw_label_obj)

    full_G = build_full_graph(graph_data)
    cls_map = parse_cls(label_data)
    instances = parse_feature_instances(label_data, cls_map, full_G.number_of_nodes())
    inst_by_id = {int(inst["instance_id"]): inst for inst in instances}

    for sg in sample_meta.get("subgraphs", []):
        iid = int(sg["instance_id"])
        rec = records_by_key.get((sample_name, iid))
        inst = inst_by_id.get(iid)
        if rec is None or inst is None:
            sg["topology_variant"] = {
                "relation": "unknown",
                "relation_label": RELATION_LABELS["unknown"],
                "variant_type": "",
                "reason": "classification_record_not_found",
                "reason_label": "Classification record was not found for this instance.",
                "green_check": False,
                "history_steps": [],
            }
            continue

        sg["topology_variant"] = build_history_for_instance(
            sample_name=sample_name,
            inst=inst,
            cls_record=rec,
            full_G=full_G,
            cls_map=cls_map,
            cache_dir=cache_dir,
        )

    return sample_meta


# =============================================================================
# CLI
# =============================================================================

def main():
    # 手动填写参数
    dataset_name = "mfinstseg"
    cache_dir_input = fr"web_cache\{dataset_name}"

    classification_json_input = (
        f"generalized_main_topology_surface_aware_output\{dataset_name}/"
        "instance_classification_details.json"
    )

    # 留空表示覆盖 cache_dir 下的 samples_index.json
    # 也可以填写例如：
    # output_index_input = "output/updated_samples_index.json"
    output_index_input = ""

    cache_dir = Path(cache_dir_input).resolve()
    classification_json = Path(classification_json_input).resolve()

    index = load_index(cache_dir)
    records_by_key = load_classification_records(classification_json)

    augmented_samples = []

    for sample_meta in index.get("samples", []):
        sample_name = sample_meta.get("sample_name")

        try:
            augmented_sample = augment_sample(
                sample_meta,
                records_by_key,
                cache_dir,
            )

            augmented_samples.append(augmented_sample)
            print(f"[OK] augmented topology variant cache: {sample_name}")

        except Exception as e:
            print(f"[ERROR] failed to augment {sample_name}: {e}")
            augmented_samples.append(sample_meta)

    index["samples"] = augmented_samples
    index["topology_variant_history_cache"] = "topology_variant_history"
    index["topology_variant_history_layout"] = (
        "surface_aware_fixed_main_nodes"
    )

    if output_index_input:
        out_path = Path(output_index_input).resolve()
    else:
        out_path = cache_dir / "samples_index.json"

    save_json(index, out_path)

    print(f"\nDone. Augmented index written to: {out_path}")


if __name__ == "__main__":
    main()