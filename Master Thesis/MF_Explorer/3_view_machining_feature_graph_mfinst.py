# analyze_mftrcad_subgraph_topologies.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism as iso


# -----------------------------
# Face categories
# -----------------------------
FACE_CATEGORIES = [
    'chamfer',                      # 0
    'through_hole',                 # 1
    'triangular_passage',           # 2
    'rectangular_passage',          # 3
    '6sides_passage',               # 4
    'triangular_through_slot',      # 5
    'rectangular_through_slot',     # 6
    'circular_through_slot',        # 7
    'rectangular_through_step',     # 8
    '2sides_through_step',          # 9
    'slanted_through_step',         # 10
    'Oring',                        # 11
    'blind_hole',                   # 12
    'triangular_pocket',            # 13
    'rectangular_pocket',           # 14
    '6sides_pocket',                # 15
    'circular_end_pocket',          # 16
    'rectangular_blind_slot',       # 17
    'v_circular_end_blind_slot',    # 18
    'h_circular_end_blind_slot',    # 19
    'triangular_blind_step',        # 20
    'circular_blind_step',          # 21
    'rectangular_blind_step',       # 22
    'round',                        # 23
    'stock',                        # 24
]

MACHINING_CATEGORY_IDS: Set[int] = set(range(0, 24))
STOCK_CATEGORY_ID = 24

LAYOUT_SEED = 42

# 是否为每一种 topology type 画一个代表图。
# 注意：这不是每个 instance 都画 png，而是每个 topology type 画一个代表 png，方便查看。
DRAW_TYPE_REPRESENTATIVE_PNG = True


# -----------------------------
# Basic IO helpers
# -----------------------------
def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s.strip("_")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def unwrap_label_json(obj: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Support label formats:

    1. [[sample_name, label_data]]
       Example:
       [
           [
               "20221123_142528_0",
               {
                   "seg": {"0": 24, "1": 3, ...},
                   "inst": [[...], ...],
                   "bottom": {"0": 0, ...}
               }
           ]
       ]

    2. [sample_name, label_data]

    3. label_data dict
    """
    if (
        isinstance(obj, list)
        and len(obj) == 1
        and isinstance(obj[0], list)
        and len(obj[0]) >= 2
        and isinstance(obj[0][1], dict)
    ):
        sample_name = obj[0][0] if isinstance(obj[0][0], str) else None
        return sample_name, obj[0][1]

    if (
        isinstance(obj, list)
        and len(obj) >= 2
        and isinstance(obj[0], str)
        and isinstance(obj[1], dict)
    ):
        return obj[0], obj[1]

    if isinstance(obj, dict):
        return obj.get("filename", None), obj

    raise ValueError(f"Unsupported label json format: {type(obj)}")


def unwrap_graph_json(obj: Any) -> Dict[str, Any]:
    """
    兼容 graph JSON 两种格式：
    1. {"graph": {...}, ...}
    2. ["sample_name", {"graph": {...}, ...}]
    """
    if isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[1], dict):
        return obj[1]

    if isinstance(obj, dict):
        return obj

    raise ValueError("Unsupported graph json format.")


def get_sample_name(graph_path: Path, graph_obj: Any) -> str:
    if isinstance(graph_obj, list) and len(graph_obj) >= 1 and isinstance(graph_obj[0], str):
        return safe_name(graph_obj[0])

    if isinstance(graph_obj, dict) and "filename" in graph_obj:
        return safe_name(graph_obj["filename"])

    return safe_name(graph_path.stem)


def resolve_data_subdir(data_dir: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    return None


def find_label_file(label_dir: Path, graph_path: Path, sample_name: str) -> Optional[Path]:
    candidates = [
        label_dir / graph_path.name,
        label_dir / f"{graph_path.stem}.json",
        label_dir / f"{sample_name}.json",
    ]

    for p in candidates:
        if p.exists() and not p.name.endswith("_rel.json"):
            return p

    matched = sorted(label_dir.glob(f"{graph_path.stem}*.json"))
    matched = [p for p in matched if not p.name.endswith("_rel.json")]
    if matched:
        return matched[0]

    matched = sorted(label_dir.glob(f"{sample_name}*.json"))
    matched = [p for p in matched if not p.name.endswith("_rel.json")]
    if matched:
        return matched[0]

    return None


# -----------------------------
# Parse graph / label
# -----------------------------
def build_graph(graph_data: Dict[str, Any]) -> nx.Graph:
    ginfo = graph_data["graph"]
    num_nodes = int(ginfo["num_nodes"])
    edges = ginfo["edges"]

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # AAGNet / MFInstSeg 常见格式：edges = [[src...], [dst...]]
    if (
        isinstance(edges, list)
        and len(edges) == 2
        and isinstance(edges[0], list)
        and isinstance(edges[1], list)
    ):
        for u, v in zip(edges[0], edges[1]):
            u, v = int(u), int(v)
            if u != v:
                G.add_edge(u, v)
    else:
        # 兼容 edges = [[u, v], [u, v], ...]
        for e in edges:
            if len(e) >= 2:
                u, v = int(e[0]), int(e[1])
                if u != v:
                    G.add_edge(u, v)

    return G


def parse_cls(label_data: Dict[str, Any]) -> Dict[int, int]:
    """
    当前 MFInstSeg / AAGNet label 格式：
        seg[face_id] = category_id

    兼容：
        cls[face_id] = category_id
    """
    if "seg" in label_data and isinstance(label_data["seg"], dict):
        cls_raw = label_data["seg"]
    elif "cls" in label_data and isinstance(label_data["cls"], dict):
        cls_raw = label_data["cls"]
    else:
        raise KeyError(
            "Cannot find face category map. Expected label_data['seg'] or label_data['cls'] as dict."
        )

    return {int(k): int(v) for k, v in cls_raw.items()}


def infer_feature_category(face_ids: List[int], cls_map: Dict[int, int]) -> Optional[int]:
    """
    根据一个 feature instance 的 face ids，从 cls_map 反推 category。
    若一个 instance 内存在多个 machining category，取出现次数最多的 category。
    """
    cats = [cls_map.get(int(fid)) for fid in face_ids]
    cats = [c for c in cats if c is not None and c in MACHINING_CATEGORY_IDS]

    if not cats:
        return None

    counts: Dict[int, int] = {}
    for c in cats:
        counts[c] = counts.get(c, 0) + 1

    return max(counts.items(), key=lambda x: x[1])[0]


def parse_feature_instances_from_inst_matrix(
    label_data: Dict[str, Any],
    cls_map: Dict[int, int],
    num_nodes: int,
) -> List[Dict[str, Any]]:
    """
    当前 MFInstSeg / AAGNet label 格式：

        seg[face_id] = category_id
        inst[i][j] = 1 表示 face i 和 face j 属于同一个 feature instance

    做法：
        1. 用 inst matrix 建 instance relation graph
        2. 只保留 machining category face，即 0..23，排除 stock=24
        3. connected_components 得到每个 instance 的 face set
    """
    inst_matrix = label_data.get("inst", None)
    if inst_matrix is None:
        raise KeyError("Cannot find label_data['inst'].")

    if not isinstance(inst_matrix, list):
        raise TypeError("label_data['inst'] should be a 2D list.")

    relation_graph = nx.Graph()

    machining_nodes = [
        int(n)
        for n in range(num_nodes)
        if cls_map.get(int(n), STOCK_CATEGORY_ID) in MACHINING_CATEGORY_IDS
    ]

    relation_graph.add_nodes_from(machining_nodes)

    rows = min(len(inst_matrix), num_nodes)

    for i in range(rows):
        row = inst_matrix[i]

        if not isinstance(row, list):
            continue

        cols = min(len(row), num_nodes)

        for j in range(cols):
            if i == j:
                continue

            try:
                related = int(row[j]) == 1
            except Exception:
                related = False

            if not related:
                continue

            ci = cls_map.get(i, STOCK_CATEGORY_ID)
            cj = cls_map.get(j, STOCK_CATEGORY_ID)

            if ci not in MACHINING_CATEGORY_IDS:
                continue

            if cj not in MACHINING_CATEGORY_IDS:
                continue

            relation_graph.add_edge(i, j)

    instances: List[Dict[str, Any]] = []

    for comp in nx.connected_components(relation_graph):
        face_ids = sorted(int(x) for x in comp)

        if not face_ids:
            continue

        category_id = infer_feature_category(face_ids, cls_map)

        if category_id is None:
            continue

        cats = [cls_map.get(fid) for fid in face_ids if fid in cls_map]
        unique_cats = sorted(set(cats))

        if len(unique_cats) > 1:
            print(
                f"[WARN] instance component has mixed categories: "
                f"faces={face_ids}, cats={unique_cats}. "
                f"Use majority category {category_id}."
            )

        instances.append({
            "instance_id": int(len(instances)),
            "category_id": int(category_id),
            "category_name": FACE_CATEGORIES[int(category_id)],
            "nodes": face_ids,
        })

    return instances


def parse_feature_instances(
    label_data: Dict[str, Any],
    cls_map: Dict[int, int],
    num_nodes: int,
) -> List[Dict[str, Any]]:
    """
    兼容两种格式：

    旧格式：
        seg[instance_id] = [face_id_1, face_id_2, ...]

    当前格式：
        seg[face_id] = category_id
        inst = NxN instance matrix
    """
    seg = label_data.get("seg", None)

    # 旧格式：seg 是 instance list
    if isinstance(seg, list):
        instances: List[Dict[str, Any]] = []

        for instance_id, face_ids in enumerate(seg):
            if not face_ids:
                continue

            face_ids = sorted({int(fid) for fid in face_ids})
            category_id = infer_feature_category(face_ids, cls_map)

            if category_id is None:
                print(f"[WARN] seg[{instance_id}] has no valid machining category: faces={face_ids}")
                continue

            if category_id not in MACHINING_CATEGORY_IDS:
                continue

            cats = [cls_map.get(fid) for fid in face_ids if fid in cls_map]
            unique_cats = sorted(set(cats))

            if len(unique_cats) > 1:
                print(
                    f"[WARN] seg[{instance_id}] has mixed categories: "
                    f"faces={face_ids}, cats={unique_cats}. "
                    f"Use majority category {category_id}."
                )

            instances.append({
                "instance_id": int(instance_id),
                "category_id": int(category_id),
                "category_name": FACE_CATEGORIES[category_id],
                "nodes": face_ids,
            })

        return instances

    # 当前格式：seg 是 face category dict，inst 是 NxN instance matrix
    if isinstance(seg, dict):
        return parse_feature_instances_from_inst_matrix(
            label_data=label_data,
            cls_map=cls_map,
            num_nodes=num_nodes,
        )

    raise ValueError("Unsupported label_data['seg'] format.")


# -----------------------------
# Build topology graph
# -----------------------------
def build_instance_topology_graph(
    G: nx.Graph,
    cls_map: Dict[int, int],
    inst: Dict[str, Any],
    include_one_hop_context: bool,
) -> nx.Graph:
    """
    生成一个 instance 的 topology graph。

    no_context:
        只包含当前 instance 的 feature nodes

    with_context:
        当前 instance nodes + 一跳外部邻居 nodes

    节点属性：
        original_face_id
        role: "feature" or "context"
        category_id
        category_name

    边属性：
        edge_role:
            internal: feature-feature
            boundary: feature-context
            context: context-context
    """
    feature_nodes: Set[int] = set(int(n) for n in inst["nodes"])

    if include_one_hop_context:
        selected_nodes = set(feature_nodes)
        for n in feature_nodes:
            if n in G:
                selected_nodes.update(int(x) for x in G.neighbors(n))
    else:
        selected_nodes = set(feature_nodes)

    SG_original = G.subgraph(sorted(selected_nodes)).copy()

    topo = nx.Graph()

    # 保留原 face id 作为节点编号，保存 json 时再 canonical relabel
    for n in sorted(SG_original.nodes()):
        cat_id = cls_map.get(int(n), -1)
        role = "feature" if n in feature_nodes else "context"

        cat_name = FACE_CATEGORIES[cat_id] if 0 <= cat_id < len(FACE_CATEGORIES) else "unknown"

        topo.add_node(
            int(n),
            original_face_id=int(n),
            role=role,
            category_id=int(cat_id),
            category_name=cat_name,
        )

    for u, v in sorted(SG_original.edges()):
        u_in = int(u) in feature_nodes
        v_in = int(v) in feature_nodes

        if u_in and v_in:
            edge_role = "internal"
        elif u_in or v_in:
            edge_role = "boundary"
        else:
            edge_role = "context"

        topo.add_edge(int(u), int(v), edge_role=edge_role)

    return topo


def topology_graph_to_json_data(
    topo: nx.Graph,
    sample_name: str,
    inst: Dict[str, Any],
    version: str,
) -> Dict[str, Any]:
    """
    把 topology graph 保存成稳定 JSON。

    注意：
    topology_nodes 是重新编号后的 0..N-1。
    original_face_id 保留原始 graph face/node id。
    """
    original_nodes_sorted = sorted(topo.nodes())
    node_mapping = {old: new for new, old in enumerate(original_nodes_sorted)}

    nodes_json = []
    for old_id in original_nodes_sorted:
        new_id = node_mapping[old_id]
        attr = topo.nodes[old_id]

        nodes_json.append({
            "id": int(new_id),
            "original_face_id": int(attr["original_face_id"]),
            "role": attr["role"],
            "category_id": int(attr["category_id"]),
            "category_name": attr["category_name"],
        })

    edges_json = []
    for u, v, attr in sorted(
        topo.edges(data=True),
        key=lambda x: (node_mapping[x[0]], node_mapping[x[1]])
    ):
        uu = node_mapping[u]
        vv = node_mapping[v]

        if uu > vv:
            uu, vv = vv, uu

        edges_json.append({
            "source": int(uu),
            "target": int(vv),
            "edge_role": attr.get("edge_role", "unknown"),
        })

    edges_json = sorted(edges_json, key=lambda e: (e["source"], e["target"], e["edge_role"]))

    return {
        "sample_name": sample_name,
        "version": version,
        "instance_id": int(inst["instance_id"]),
        "category_id": int(inst["category_id"]),
        "category_name": inst["category_name"],
        "feature_original_face_ids": [int(x) for x in inst["nodes"]],
        "num_nodes": topo.number_of_nodes(),
        "num_edges": topo.number_of_edges(),
        "nodes": nodes_json,
        "edges": edges_json,
    }


def json_data_to_topology_graph(data: Dict[str, Any]) -> nx.Graph:
    """
    从保存的 topology json 还原 NetworkX graph。
    用于后续 isomorphism 分组。
    """
    G = nx.Graph()

    for n in data["nodes"]:
        G.add_node(
            int(n["id"]),
            role=n["role"],
            category_id=int(n["category_id"]),
            category_name=n["category_name"],
        )

    for e in data["edges"]:
        G.add_edge(
            int(e["source"]),
            int(e["target"]),
            edge_role=e.get("edge_role", "unknown"),
        )

    return G


# -----------------------------
# Topology isomorphism
# -----------------------------
def topology_signature(G: nx.Graph) -> str:
    """
    一个快速初筛 signature。
    真正分 type 还是用 graph isomorphism。
    """
    node_labels = []

    for _, attr in G.nodes(data=True):
        node_labels.append((attr.get("role"), int(attr.get("category_id", -1))))

    node_labels = sorted(node_labels)
    edge_roles = sorted(attr.get("edge_role", "unknown") for _, _, attr in G.edges(data=True))
    degrees = sorted(dict(G.degree()).values())

    return json.dumps({
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "degrees": degrees,
        "node_labels": node_labels,
        "edge_roles": edge_roles,
    }, sort_keys=True)


def are_topologies_same(G1: nx.Graph, G2: nx.Graph) -> bool:
    """
    判断两个 topology 是否同构。

    节点匹配：
        role 相同
        category_id 相同

    边匹配：
        edge_role 相同
    """
    node_match = iso.categorical_node_match(["role", "category_id"], [None, None])
    edge_match = iso.categorical_edge_match("edge_role", None)

    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match)


def assign_topology_types(instance_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对同一个 category + version 下的一堆 instance topology 分 type。

    输入 instance_records:
        [
            {
                "sample_name": ...,
                "instance_id": ...,
                "json_path": ...,
                "topology_json": ...,
                "graph": nx.Graph,
            }
        ]

    返回 types:
        [
            {
                "type_id": 1,
                "representative": record,
                "members": [record, ...],
            }
        ]
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {}

    for rec in instance_records:
        sig = topology_signature(rec["graph"])
        buckets.setdefault(sig, []).append(rec)

    types = []
    next_type_id = 1

    for _, bucket_records in buckets.items():
        local_types = []

        for rec in bucket_records:
            matched_type = None

            for t in local_types:
                if are_topologies_same(rec["graph"], t["representative"]["graph"]):
                    matched_type = t
                    break

            if matched_type is None:
                local_types.append({
                    "type_id": None,
                    "representative": rec,
                    "members": [rec],
                })
            else:
                matched_type["members"].append(rec)

        for t in local_types:
            t["type_id"] = next_type_id
            next_type_id += 1
            types.append(t)

    # 按成员数量降序排列，重新编号：type001 是最常见拓扑
    types = sorted(types, key=lambda t: len(t["members"]), reverse=True)

    for idx, t in enumerate(types, start=1):
        t["type_id"] = idx

    return types


# -----------------------------
# Draw representative type png
# -----------------------------
def draw_topology_type_png(
    topo_graph: nx.Graph,
    out_path: Path,
    title: str,
):
    """
    给每种 topology type 画一个代表图，方便肉眼查看。
    不是每个 instance 都画，只画每个 type 的 representative。
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = topo_graph.number_of_nodes()
    plt.figure(figsize=(max(5.5, math.sqrt(max(n, 1)) * 2.2), 5.2))

    pos = nx.spring_layout(topo_graph, seed=LAYOUT_SEED)

    feature_nodes = [
        n for n, a in topo_graph.nodes(data=True)
        if a.get("role") == "feature"
    ]

    context_nodes = [
        n for n, a in topo_graph.nodes(data=True)
        if a.get("role") == "context"
    ]

    internal_edges = []
    boundary_edges = []
    context_edges = []

    for u, v, a in topo_graph.edges(data=True):
        role = a.get("edge_role", "unknown")

        if role == "internal":
            internal_edges.append((u, v))
        elif role == "boundary":
            boundary_edges.append((u, v))
        else:
            context_edges.append((u, v))

    if context_edges:
        nx.draw_networkx_edges(
            topo_graph,
            pos,
            edgelist=context_edges,
            edge_color=["#D0D0D0"] * len(context_edges),
            width=1.0,
            alpha=0.5,
        )

    if boundary_edges:
        nx.draw_networkx_edges(
            topo_graph,
            pos,
            edgelist=boundary_edges,
            edge_color=["#666666"] * len(boundary_edges),
            width=1.8,
            alpha=0.85,
            style="dashed",
        )

    if internal_edges:
        nx.draw_networkx_edges(
            topo_graph,
            pos,
            edgelist=internal_edges,
            edge_color=["#000000"] * len(internal_edges),
            width=2.5,
            alpha=0.95,
        )

    if context_nodes:
        nx.draw_networkx_nodes(
            topo_graph,
            pos,
            nodelist=context_nodes,
            node_color=["#E0E0E0"] * len(context_nodes),
            node_size=650,
            edgecolors=["#777777"] * len(context_nodes),
            linewidths=1.0,
        )

    if feature_nodes:
        nx.draw_networkx_nodes(
            topo_graph,
            pos,
            nodelist=feature_nodes,
            node_color=["#FFCC66"] * len(feature_nodes),
            node_size=900,
            edgecolors=["black"] * len(feature_nodes),
            linewidths=2.0,
        )

    labels = {}
    for node, attr in topo_graph.nodes(data=True):
        role = attr.get("role")
        cat_id = int(attr.get("category_id", -1))
        cat_name = FACE_CATEGORIES[cat_id] if 0 <= cat_id < len(FACE_CATEGORIES) else "unknown"
        prefix = "F" if role == "feature" else "C"
        labels[node] = f"{prefix}{node}\n{cat_id}:{cat_name}"

    nx.draw_networkx_labels(
        topo_graph,
        pos,
        labels=labels,
        font_size=7,
        font_color="black",
    )

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main processing
# -----------------------------
def process_one_sample(
    graph_path: Path,
    label_dir: Path,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """
    处理一个样本，生成该样本所有 instance 的 topology json。

    返回 records，用于后续统计 topology types。
    """
    raw_graph_obj = load_json(graph_path)
    graph_data = unwrap_graph_json(raw_graph_obj)
    sample_name = get_sample_name(graph_path, raw_graph_obj)

    label_path = find_label_file(label_dir, graph_path, sample_name)
    if label_path is None:
        print(f"[WARN] Cannot find label file for graph: {graph_path.name}")
        return []

    raw_label_obj = load_json(label_path)
    label_sample_name, label_data = unwrap_label_json(raw_label_obj)

    if label_sample_name:
        sample_name = safe_name(label_sample_name)

    G = build_graph(graph_data)

    cls_map = parse_cls(label_data)

    feature_instances = parse_feature_instances(
        label_data=label_data,
        cls_map=cls_map,
        num_nodes=G.number_of_nodes(),
    )

    records: List[Dict[str, Any]] = []

    for inst in feature_instances:
        category_name = safe_name(inst["category_name"])

        for version, include_context in [
            ("no_context", False),
            ("with_context", True),
        ]:
            topo = build_instance_topology_graph(
                G=G,
                cls_map=cls_map,
                inst=inst,
                include_one_hop_context=include_context,
            )

            topo_json = topology_graph_to_json_data(
                topo=topo,
                sample_name=sample_name,
                inst=inst,
                version=version,
            )

            nodes_str = "-".join(map(str, inst["nodes"]))

            file_name = (
                f"{sample_name}"
                f"__seg{int(inst['instance_id']):03d}"
                f"__cls{int(inst['category_id']):02d}_{safe_name(inst['category_name'])}"
                f"__nodes_{nodes_str}"
                f"__{version}.json"
            )

            instance_json_path = (
                output_dir
                / "topology_instances"
                / version
                / category_name
                / file_name
            )

            save_json(topo_json, instance_json_path)

            topo_for_iso = json_data_to_topology_graph(topo_json)

            records.append({
                "sample_name": sample_name,
                "instance_id": int(inst["instance_id"]),
                "category_id": int(inst["category_id"]),
                "category_name": inst["category_name"],
                "version": version,
                "feature_original_face_ids": [int(x) for x in inst["nodes"]],
                "json_path": str(instance_json_path),
                "topology_json": topo_json,
                "graph": topo_for_iso,
            })

    print(
        f"[OK] {sample_name}: "
        f"{G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, "
        f"{len(feature_instances)} instances, "
        f"{len(records)} topology json files"
    )

    return records


def write_type_outputs(
    grouped_records: Dict[Tuple[str, int, str], List[Dict[str, Any]]],
    output_dir: Path,
):
    """
    对每个 version + category 分析 topology types，并输出：
        output/topology_types/{version}/{category_name}/type001/type001.json
        output/topology_types/{version}/{category_name}/type001/type001.png
        output/topology_types/{version}/{category_name}/type001/members.txt

    同时打印最常见 type 和非主流样本。
    """
    summary = []

    for (version, category_id, category_name), records in sorted(
        grouped_records.items(),
        key=lambda x: (x[0][0], x[0][1], x[0][2])
    ):
        if not records:
            continue

        types = assign_topology_types(records)

        category_dir = (
            output_dir
            / "topology_types"
            / version
            / safe_name(category_name)
        )
        category_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 100)
        print(f"[CATEGORY] version={version} | {category_id}:{category_name}")
        print(f"Total instances: {len(records)}")
        print(f"Different topology types: {len(types)}")

        if types:
            most_common = types[0]
            print(
                f"Most common topology: type{most_common['type_id']:03d} "
                f"with {len(most_common['members'])}/{len(records)} instances"
            )

        for t in types:
            type_id = int(t["type_id"])
            type_name = f"type{type_id:03d}"
            type_dir = category_dir / type_name
            type_dir.mkdir(parents=True, exist_ok=True)

            rep = t["representative"]
            members = t["members"]

            # representative topology json
            rep_json = rep["topology_json"].copy()
            rep_json["topology_type"] = type_name
            rep_json["num_members"] = len(members)
            rep_json["members"] = [
                {
                    "sample_name": m["sample_name"],
                    "instance_id": m["instance_id"],
                    "category_id": m["category_id"],
                    "category_name": m["category_name"],
                    "feature_original_face_ids": m["feature_original_face_ids"],
                    "json_path": m["json_path"],
                }
                for m in members
            ]

            save_json(rep_json, type_dir / f"{type_name}.json")

            # members txt
            with (type_dir / "members.txt").open("w", encoding="utf-8") as f:
                for m in members:
                    f.write(
                        f"{m['sample_name']} | "
                        f"I{m['instance_id']} | "
                        f"faces={m['feature_original_face_ids']} | "
                        f"{m['json_path']}\n"
                    )

            # representative png
            if DRAW_TYPE_REPRESENTATIVE_PNG:
                draw_topology_type_png(
                    topo_graph=rep["graph"],
                    out_path=type_dir / f"{type_name}.png",
                    title=(
                        f"{version} | {category_id}:{category_name} | "
                        f"{type_name} | count={len(members)}"
                    ),
                )

            print(f"  {type_name}: {len(members)} instances")

        # 非主流 topology
        if len(types) > 1:
            print("\nNon-most-common topology instances:")
            for t in types[1:]:
                type_name = f"type{int(t['type_id']):03d}"
                print(f"  {type_name}:")
                for m in t["members"]:
                    print(
                        f"    {m['sample_name']} | "
                        f"I{m['instance_id']} | "
                        f"faces={m['feature_original_face_ids']}"
                    )
        else:
            print("All instances belong to the most common topology.")

        summary.append({
            "version": version,
            "category_id": category_id,
            "category_name": category_name,
            "total_instances": len(records),
            "num_topology_types": len(types),
            "most_common_type": "type001" if types else None,
            "most_common_count": len(types[0]["members"]) if types else 0,
            "types": [
                {
                    "type": f"type{int(t['type_id']):03d}",
                    "count": len(t["members"]),
                    "members": [
                        {
                            "sample_name": m["sample_name"],
                            "instance_id": m["instance_id"],
                            "faces": m["feature_original_face_ids"],
                        }
                        for m in t["members"]
                    ],
                }
                for t in types
            ],
        })

    save_json(summary, output_dir / "topology_type_summary.json")


def process_dataset(data_dir: str | Path, output_dir: str | Path = "output"):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    graph_dir = resolve_data_subdir(data_dir, ["graph", "graphs"])
    label_dir = resolve_data_subdir(data_dir, ["label", "labels"])

    if graph_dir is None:
        raise FileNotFoundError(
            f"graph folder not found under: {data_dir} "
            f"(tried graph/ and graphs/)"
        )

    if label_dir is None:
        raise FileNotFoundError(
            f"label folder not found under: {data_dir} "
            f"(tried label/ and labels/)"
        )

    graph_files = sorted(graph_dir.glob("*.json"))

    # 避免把 attr_stat.json 当成样本图
    graph_files = [p for p in graph_files if p.name != "attr_stat.json"]

    if not graph_files:
        print(f"[WARN] No graph json files found in {graph_dir}")
        return

    all_records: List[Dict[str, Any]] = []

    for graph_path in graph_files:
        try:
            records = process_one_sample(
                graph_path=graph_path,
                label_dir=label_dir,
                output_dir=output_dir,
            )
            all_records.extend(records)
        except Exception as e:
            print(f"[ERROR] Failed on {graph_path.name}: {e}")

    grouped_records: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = {}

    for rec in all_records:
        key = (
            rec["version"],
            int(rec["category_id"]),
            rec["category_name"],
        )
        grouped_records.setdefault(key, []).append(rec)

    print("\n" + "#" * 100)
    print("Topology type analysis")
    print(f"Total topology records: {len(all_records)}")
    print(f"Total groups: {len(grouped_records)}")

    write_type_outputs(
        grouped_records=grouped_records,
        output_dir=output_dir,
    )

    print("\n" + "#" * 100)
    print("Done.")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Instance topology JSONs: {output_dir / 'topology_instances'}")
    print(f"Topology type outputs:   {output_dir / 'topology_types'}")
    print(f"Summary JSON:             {output_dir / 'topology_type_summary.json'}")


if __name__ == "__main__":
    process_dataset(
        data_dir="data/mfinstseg",
        output_dir="output/mfinstseg",
    )