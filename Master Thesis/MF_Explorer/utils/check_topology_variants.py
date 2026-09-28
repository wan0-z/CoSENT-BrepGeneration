# verify_generalized_main_topology_hypothesis.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import itertools
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism as iso


# ============================================================
# Config
# ============================================================
DATASET_NAME = "mftrcad"
DATA_DIR = Path(f"data/{DATASET_NAME}")
GRAPH_DIR = DATA_DIR / "graphs"
LABEL_DIR = DATA_DIR / "labels"

# 这里读取你之前 analyze_mftrcad_subgraph_topologies.py 生成的 main type
TOPOLOGY_TYPES_DIR = Path(f"output/{DATASET_NAME}") / "topology_types"
MAIN_VERSION = "no_context"

RESULT_DIR = Path("generalized_main_topology_output")

# 如果某个 feature instance 的节点特别多，组合搜索可能爆炸。
# 一般 machining feature 子图节点数不会太大。
MAX_COMBINATIONS_PER_NODE = 200000

DRAW_FIGURES = True


# ============================================================
# Face categories
# ============================================================

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


# ============================================================
# IO helpers
# ============================================================

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


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# Parse graph / label
# ============================================================

def unwrap_graph_json(obj: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    支持：
        [sample_name, graph_data]
        graph_data
    """
    if isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[1], dict):
        sample_name = obj[0] if isinstance(obj[0], str) else None
        return sample_name, obj[1]

    if isinstance(obj, dict):
        return obj.get("filename", None), obj

    raise ValueError(f"Unsupported graph json format: {type(obj)}")


def unwrap_label_json(obj: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    支持：
        [[sample_name, label_data]]
        [sample_name, label_data]
        label_data
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


def build_full_graph(graph_data: Dict[str, Any]) -> nx.Graph:
    ginfo = graph_data["graph"]
    num_nodes = int(ginfo["num_nodes"])
    edges = ginfo["edges"]

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

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
        for e in edges:
            if len(e) >= 2:
                u, v = int(e[0]), int(e[1])
                if u != v:
                    G.add_edge(u, v)

    return G


def parse_cls(label_data: Dict[str, Any]) -> Dict[int, int]:
    """
    当前 AAGNet / MFInstSeg:
        seg[face_id] = category_id
    """
    if "seg" in label_data and isinstance(label_data["seg"], dict):
        raw = label_data["seg"]
    elif "cls" in label_data and isinstance(label_data["cls"], dict):
        raw = label_data["cls"]
    else:
        raise KeyError("Cannot find label_data['seg'] or label_data['cls'] as dict.")

    return {int(k): int(v) for k, v in raw.items()}


def infer_feature_category(face_ids: List[int], cls_map: Dict[int, int]) -> Optional[int]:
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
    inst[i][j] = 1 表示 face i 和 face j 属于同一个 feature instance。
    """
    inst_matrix = label_data.get("inst", None)

    if inst_matrix is None:
        raise KeyError("Cannot find label_data['inst'].")

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

    instances = []

    for comp in nx.connected_components(relation_graph):
        face_ids = sorted(int(x) for x in comp)

        if not face_ids:
            continue

        category_id = infer_feature_category(face_ids, cls_map)

        if category_id is None:
            continue

        instances.append({
            "instance_id": len(instances),
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
    seg = label_data.get("seg", None)

    if isinstance(seg, dict):
        return parse_feature_instances_from_inst_matrix(
            label_data=label_data,
            cls_map=cls_map,
            num_nodes=num_nodes,
        )

    if isinstance(seg, list):
        instances = []

        for instance_id, face_ids in enumerate(seg):
            if not face_ids:
                continue

            face_ids = sorted({int(fid) for fid in face_ids})
            category_id = infer_feature_category(face_ids, cls_map)

            if category_id is None:
                continue

            if category_id not in MACHINING_CATEGORY_IDS:
                continue

            instances.append({
                "instance_id": int(instance_id),
                "category_id": int(category_id),
                "category_name": FACE_CATEGORIES[int(category_id)],
                "nodes": face_ids,
            })

        return instances

    raise ValueError("Unsupported label_data['seg'] format.")


def find_label_file(label_dir: Path, sample_name: str, graph_path: Path) -> Optional[Path]:
    candidates = [
        label_dir / f"{sample_name}.json",
        label_dir / graph_path.name,
        label_dir / f"{graph_path.stem}.json",
    ]

    for p in candidates:
        if p.exists() and not p.name.endswith("_rel.json"):
            return p

    matched = sorted(label_dir.glob(f"{sample_name}*.json"))
    matched = [p for p in matched if not p.name.endswith("_rel.json")]

    if matched:
        return matched[0]

    return None


# ============================================================
# Main topology loading
# ============================================================

def topology_type_json_to_graph(data: Dict[str, Any]) -> nx.Graph:
    """
    读取 type001.json 里的 topology graph。

    这里的节点已经是 topology node id，不是原始 face id。
    """
    G = nx.Graph()

    for n in data["nodes"]:
        node_id = int(n["id"])
        role = str(n.get("role", "feature"))
        category_id = int(n.get("category_id", -1))

        G.add_node(
            node_id,
            role=role,
            category_id=category_id,
        )

    for e in data["edges"]:
        u = int(e["source"])
        v = int(e["target"])
        edge_role = str(e.get("edge_role", "internal"))

        if u != v:
            G.add_edge(
                u,
                v,
                edge_role=edge_role,
            )

    return G


def load_main_type_graphs(
    topology_types_dir: Path,
    version: str,
) -> Dict[int, Dict[str, Any]]:
    """
    返回：
        category_id -> {
            category_name,
            main_graph,
            main_json_path,
            main_json,
            main_count
        }
    """
    version_dir = topology_types_dir / version

    if not version_dir.exists():
        raise FileNotFoundError(f"Cannot find topology type version dir: {version_dir}")

    result: Dict[int, Dict[str, Any]] = {}

    for category_dir in sorted(version_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        type001_path = category_dir / "type001" / "type001.json"

        if not type001_path.exists():
            print(f"[WARN] Cannot find type001 for category dir: {category_dir}")
            continue

        data = load_json(type001_path)

        category_id = int(data["category_id"])
        category_name = str(data["category_name"])
        main_graph = topology_type_json_to_graph(data)

        result[category_id] = {
            "category_id": category_id,
            "category_name": category_name,
            "main_graph": main_graph,
            "main_json_path": str(type001_path),
            "main_json": data,
            "main_count_in_topology_types": int(data.get("num_members", 0)),
        }

    return result


# ============================================================
# Graph construction for current instance
# ============================================================

def node_label(G: nx.Graph, n: int) -> Tuple[str, int]:
    a = G.nodes[n]
    return str(a.get("role", "feature")), int(a.get("category_id", -1))


def edge_label(G: nx.Graph, u: int, v: int) -> str:
    return str(G.edges[u, v].get("edge_role", "internal"))


def labeled_isomorphic(G1: nx.Graph, G2: nx.Graph) -> bool:
    node_match = iso.categorical_node_match(["role", "category_id"], [None, None])
    edge_match = iso.categorical_edge_match("edge_role", None)
    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=edge_match)


def main_embeds_in_candidate(
    main_G: nx.Graph,
    candidate_G: nx.Graph,
) -> bool:
    """
    判断 main_G 是否可以嵌入 candidate_G。

    要求：
        1. main_G 的每个节点映射到 candidate_G 的一个不同节点；
        2. 节点 label 相同：role, category_id；
        3. main_G 的每条边，在 candidate_G 中也存在；
        4. candidate_G 允许有额外边。

    这比 graph isomorphism 宽松：
        isomorphism 要求边完全一致；
        embedding 只要求 main 的边被 candidate 包含。
    """
    if main_G.number_of_nodes() > candidate_G.number_of_nodes():
        return False

    if main_G.number_of_edges() > candidate_G.number_of_edges():
        return False

    main_nodes = list(main_G.nodes())
    cand_nodes = list(candidate_G.nodes())

    # 按 degree 从大到小排序，减少搜索量
    main_nodes = sorted(
        main_nodes,
        key=lambda n: main_G.degree(n),
        reverse=True,
    )

    candidates_by_main_node: Dict[int, List[int]] = {}

    for m in main_nodes:
        m_label = node_label(main_G, m)

        possible = [
            c for c in cand_nodes
            if node_label(candidate_G, c) == m_label
        ]

        if not possible:
            return False

        candidates_by_main_node[m] = possible

    mapping: Dict[int, int] = {}
    used_cand_nodes: Set[int] = set()

    def backtrack(idx: int) -> bool:
        if idx == len(main_nodes):
            return True

        m = main_nodes[idx]

        for c in candidates_by_main_node[m]:
            if c in used_cand_nodes:
                continue

            ok = True

            # 检查已经映射过的 main 邻居。
            # main 中有边，则 candidate 中必须有同 edge_role 的边。
            for m_prev, c_prev in mapping.items():
                if main_G.has_edge(m, m_prev):
                    if not candidate_G.has_edge(c, c_prev):
                        ok = False
                        break

                    main_edge_role = edge_label(main_G, m, m_prev)
                    cand_edge_role = edge_label(candidate_G, c, c_prev)

                    if main_edge_role != cand_edge_role:
                        ok = False
                        break

            if not ok:
                continue

            mapping[m] = c
            used_cand_nodes.add(c)

            if backtrack(idx + 1):
                return True

            used_cand_nodes.remove(c)
            del mapping[m]

        return False

    return backtrack(0)


def build_direct_feature_graph(
    full_G: nx.Graph,
    feature_nodes: List[int],
    cls_map: Dict[int, int],
) -> nx.Graph:
    """
    只保留当前 feature nodes 之间的直接边。
    """
    feature_set = set(int(x) for x in feature_nodes)
    H = nx.Graph()

    for n in sorted(feature_set):
        H.add_node(
            int(n),
            role="feature",
            category_id=int(cls_map.get(int(n), -1)),
        )

    for u, v in full_G.subgraph(feature_set).edges():
        H.add_edge(int(u), int(v), edge_role="internal")

    return nx.convert_node_labels_to_integers(H, ordering="sorted")


def has_generalized_connection(
    full_G: nx.Graph,
    instance_node_set: Set[int],
    a: int,
    b: int,
) -> bool:
    """
    判断当前 instance 内部节点 a 和 b 是否广义连接。

    定义：
        a 和 b 广义相连，当且仅当 full graph 中存在路径：

            a = p0, p1, ..., pk = b

        并且所有中间节点都不属于当前 instance：

            p1, ..., p{k-1} ∉ instance_node_set

    注意：
        中间节点可以是：
            - stock face
            - 其他 machining feature instance 的 face
            - 其他同类/不同类 feature 的 face

        中间节点唯一不能是：
            - 当前正在判断的这个 instance 的其他内部节点
    """
    if a == b:
        return False

    # 直接连接当然算广义连接
    if full_G.has_edge(a, b):
        return True

    visited: Set[int] = set()
    queue: List[int] = []

    # 从 a 出发，只允许进入当前 instance 外部节点，或者直接到达 b
    for nb in full_G.neighbors(a):
        nb = int(nb)

        if nb == b:
            return True

        # 不能经过当前 instance 内部的其他节点
        if nb in instance_node_set:
            continue

        visited.add(nb)
        queue.append(nb)

    head = 0

    while head < len(queue):
        x = queue[head]
        head += 1

        for nb in full_G.neighbors(x):
            nb = int(nb)

            if nb == b:
                return True

            # 只禁止经过当前 instance 内部节点；
            # 其他 instance 的 feature 面、stock 面都允许经过。
            if nb in instance_node_set:
                continue

            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    return False


def build_generalized_feature_graph(
    full_G: nx.Graph,
    feature_nodes: List[int],
    cls_map: Dict[int, int],
) -> nx.Graph:
    """
    构造当前 instance 的广义 feature graph。

    节点：
        当前 instance 的 feature nodes。

    边：
        对任意两个当前 instance 内部节点 a, b，
        如果它们在 full graph 中直接相连，
        或者可以通过当前 instance 外部节点间接相连，
        则在 generalized graph 中添加边 a-b。

    这里的“外部节点”定义为：
        full graph 中不属于当前 instance 的所有节点。

    它可以是：
        - stock 面
        - 其他 machining feature instance 的面
        - 其他同类 feature instance 的面
        - 其他不同类 feature instance 的面
    """
    feature_nodes = sorted(int(x) for x in feature_nodes)
    instance_node_set = set(feature_nodes)

    H = nx.Graph()

    for n in feature_nodes:
        H.add_node(
            int(n),
            role="feature",
            category_id=int(cls_map.get(int(n), -1)),
        )

    for i, a in enumerate(feature_nodes):
        for b in feature_nodes[i + 1:]:
            if has_generalized_connection(
                full_G=full_G,
                instance_node_set=instance_node_set,
                a=a,
                b=b,
            ):
                H.add_edge(int(a), int(b), edge_role="internal")

    return nx.convert_node_labels_to_integers(H, ordering="sorted")


# ============================================================
# Generalized main type coverage
# ============================================================

def category_multiset(G: nx.Graph) -> List[int]:
    return sorted(int(a.get("category_id", -1)) for _, a in G.nodes(data=True))


def graph_induced_by_nodes_relabel(G: nx.Graph, nodes: List[int]) -> nx.Graph:
    H = G.subgraph(nodes).copy()
    return nx.convert_node_labels_to_integers(H, ordering="sorted")


def subset_can_form_main_type(
    generalized_G: nx.Graph,
    subset_nodes: List[int],
    main_G: nx.Graph,
) -> bool:
    """
    判断 subset_nodes 在 generalized_G 中是否能组成一个 main type。

    注意：
        这里不是要求 induced subgraph 与 main_G 完全同构；
        而是要求 main_G 能嵌入这个 induced subgraph。

    也就是说：
        main_G 的边必须存在；
        但 subset 内部允许有额外边。
    """
    H = graph_induced_by_nodes_relabel(generalized_G, subset_nodes)

    if H.number_of_nodes() != main_G.number_of_nodes():
        return False

    # H 可以有比 main_G 更多的边
    if H.number_of_edges() < main_G.number_of_edges():
        return False

    # 节点 category 必须能匹配
    if category_multiset(H) != category_multiset(main_G):
        return False

    return main_embeds_in_candidate(
        main_G=main_G,
        candidate_G=H,
    )


def node_is_covered_by_generalized_main(
    generalized_G: nx.Graph,
    node: int,
    main_G: nx.Graph,
    max_combinations: int = MAX_COMBINATIONS_PER_NODE,
) -> Tuple[bool, Optional[List[int]], str]:
    """
    判断当前 node 是否能参与组成一个 generalized main type。

    即存在一个节点子集 S：
        node in S
        |S| = |V(main_G)|
        generalized_G[S] 包含 main_G。
    """
    k = main_G.number_of_nodes()
    n = generalized_G.number_of_nodes()

    if k == 1:
        # 单节点 main type：
        # 当前节点只要 label 与 main 节点一致，即认为被覆盖。
        main_node = next(iter(main_G.nodes()))
        if node_label(generalized_G, node) == node_label(main_G, main_node):
            return True, [node], "single_node_main"
        return False, None, "single_node_label_mismatch"

    if n < k:
        return False, None, "candidate_smaller_than_main"

    other_nodes = [x for x in generalized_G.nodes() if x != node]

    checked = 0

    for comb in itertools.combinations(other_nodes, k - 1):
        checked += 1

        if checked > max_combinations:
            return False, None, f"combination_limit_exceeded_{max_combinations}"

        subset = [node] + list(comb)

        if subset_can_form_main_type(
            generalized_G=generalized_G,
            subset_nodes=subset,
            main_G=main_G,
        ):
            return True, sorted(int(x) for x in subset), f"covered_after_{checked}_checks"

    return False, None, f"not_covered_after_{checked}_checks"


def all_nodes_covered_by_generalized_main(
    generalized_G: nx.Graph,
    main_G: nx.Graph,
) -> Tuple[bool, Dict[int, Dict[str, Any]], str]:
    """
    对当前 instance 的每个 feature node 检查：
        是否都能找到一个包含它的 generalized main type。
    """
    coverage: Dict[int, Dict[str, Any]] = {}

    for node in sorted(generalized_G.nodes()):
        ok, subset, reason = node_is_covered_by_generalized_main(
            generalized_G=generalized_G,
            node=node,
            main_G=main_G,
        )

        coverage[int(node)] = {
            "covered": bool(ok),
            "covering_subset": subset,
            "reason": reason,
        }

        if not ok:
            return False, coverage, f"node_{node}_not_covered:{reason}"

    return True, coverage, "all_nodes_covered"


def classify_instance_against_main(
    full_G: nx.Graph,
    cls_map: Dict[int, int],
    inst: Dict[str, Any],
    main_G: nx.Graph,
) -> Dict[str, Any]:
    """
    按新的定义分类：

    main:
        当前 instance 的 direct feature graph 与 main type 完全同构。

    variant_edge_split:
        节点数与 main type 相同；
        direct graph 不是 main；
        generalized graph 中包含 main type。
        注意 generalized graph 允许有额外边。

    variant_node_duplication:
        节点数多于 main type；
        当前 instance 的每一个 feature node，
        都能找到一个包含该 node 的节点子集，
        该子集的 generalized graph 中包含 main type。

    non_variant:
        不满足以上条件。
    """
    feature_nodes = sorted(int(x) for x in inst["nodes"])

    direct_G = build_direct_feature_graph(
        full_G=full_G,
        feature_nodes=feature_nodes,
        cls_map=cls_map,
    )

    generalized_G = build_generalized_feature_graph(
        full_G=full_G,
        feature_nodes=feature_nodes,
        cls_map=cls_map,
    )

    main_n = main_G.number_of_nodes()
    cand_n = generalized_G.number_of_nodes()

    direct_iso = (
        cand_n == main_n
        and direct_G.number_of_edges() == main_G.number_of_edges()
        and category_multiset(direct_G) == category_multiset(main_G)
        and labeled_isomorphic(direct_G, main_G)
    )

    generalized_contains_main = (
        cand_n == main_n
        and generalized_G.number_of_edges() >= main_G.number_of_edges()
        and category_multiset(generalized_G) == category_multiset(main_G)
        and main_embeds_in_candidate(
            main_G=main_G,
            candidate_G=generalized_G,
        )
    )

    if direct_iso:
        return {
            "relation": "main",
            "reason": "direct_graph_isomorphic_to_main",
            "main_num_nodes": main_n,
            "candidate_num_nodes": cand_n,
            "direct_num_edges": direct_G.number_of_edges(),
            "generalized_num_edges": generalized_G.number_of_edges(),
            "main_num_edges": main_G.number_of_edges(),
            "coverage": {},
        }

    if cand_n == main_n and generalized_contains_main:
        return {
            "relation": "variant",
            "variant_type": "edge_split",
            "reason": "same_node_count_generalized_graph_contains_main",
            "main_num_nodes": main_n,
            "candidate_num_nodes": cand_n,
            "direct_num_edges": direct_G.number_of_edges(),
            "generalized_num_edges": generalized_G.number_of_edges(),
            "main_num_edges": main_G.number_of_edges(),
            "coverage": {},
        }

    if cand_n > main_n:
        covered, coverage, reason = all_nodes_covered_by_generalized_main(
            generalized_G=generalized_G,
            main_G=main_G,
        )

        if covered:
            return {
                "relation": "variant",
                "variant_type": "node_duplication_or_extended_cover",
                "reason": reason,
                "main_num_nodes": main_n,
                "candidate_num_nodes": cand_n,
                "direct_num_edges": direct_G.number_of_edges(),
                "generalized_num_edges": generalized_G.number_of_edges(),
                "main_num_edges": main_G.number_of_edges(),
                "coverage": coverage,
            }

        return {
            "relation": "non_variant",
            "reason": reason,
            "main_num_nodes": main_n,
            "candidate_num_nodes": cand_n,
            "direct_num_edges": direct_G.number_of_edges(),
            "generalized_num_edges": generalized_G.number_of_edges(),
            "main_num_edges": main_G.number_of_edges(),
            "coverage": coverage,
        }

    return {
        "relation": "non_variant",
        "reason": (
            "same_node_count_but_generalized_graph_does_not_contain_main"
            if cand_n == main_n
            else "candidate_has_fewer_nodes_than_main"
        ),
        "main_num_nodes": main_n,
        "candidate_num_nodes": cand_n,
        "direct_num_edges": direct_G.number_of_edges(),
        "generalized_num_edges": generalized_G.number_of_edges(),
        "main_num_edges": main_G.number_of_edges(),
        "coverage": {},
    }


# ============================================================
# Dataset processing
# ============================================================

def collect_graph_files(graph_dir: Path) -> List[Path]:
    files = sorted(graph_dir.glob("*.json"))
    files = [p for p in files if p.name != "attr_stat.json"]
    return files


def process_dataset_instances(
    graph_dir: Path,
    label_dir: Path,
    main_type_graphs: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    graph_files = collect_graph_files(graph_dir)

    if not graph_files:
        raise RuntimeError(f"No graph json files found in {graph_dir}")

    records: List[Dict[str, Any]] = []

    for graph_path in graph_files:
        try:
            raw_graph_obj = load_json(graph_path)
            graph_sample_name, graph_data = unwrap_graph_json(raw_graph_obj)
            sample_name = safe_name(graph_sample_name or graph_path.stem)

            label_path = find_label_file(
                label_dir=label_dir,
                sample_name=sample_name,
                graph_path=graph_path,
            )

            if label_path is None:
                print(f"[WARN] label not found for {sample_name}")
                continue

            raw_label_obj = load_json(label_path)
            label_sample_name, label_data = unwrap_label_json(raw_label_obj)

            if label_sample_name:
                sample_name = safe_name(label_sample_name)

            full_G = build_full_graph(graph_data)
            cls_map = parse_cls(label_data)

            instances = parse_feature_instances(
                label_data=label_data,
                cls_map=cls_map,
                num_nodes=full_G.number_of_nodes(),
            )

            for inst in instances:
                category_id = int(inst["category_id"])

                if category_id not in main_type_graphs:
                    print(
                        f"[WARN] no main type for category "
                        f"{category_id}:{inst['category_name']}, skip instance"
                    )
                    continue

                main_info = main_type_graphs[category_id]
                main_G = main_info["main_graph"]

                cls_result = classify_instance_against_main(
                    full_G=full_G,
                    cls_map=cls_map,
                    inst=inst,
                    main_G=main_G,
                )

                rec = {
                    "sample_name": sample_name,
                    "graph_json": str(graph_path),
                    "label_json": str(label_path),
                    "instance_id": int(inst["instance_id"]),
                    "category_id": category_id,
                    "category_name": inst["category_name"],
                    "feature_original_face_ids": [int(x) for x in inst["nodes"]],
                    "main_type_json": main_info["main_json_path"],
                    **cls_result,
                }

                records.append(rec)

            print(
                f"[OK] {sample_name}: "
                f"nodes={full_G.number_of_nodes()}, "
                f"edges={full_G.number_of_edges()}, "
                f"instances={len(instances)}"
            )

        except Exception as e:
            print(f"[ERROR] failed on {graph_path.name}: {e}")

    return records


# ============================================================
# Summaries
# ============================================================

def summarize_by_category(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}

    for r in records:
        key = (int(r["category_id"]), r["category_name"])
        grouped.setdefault(key, []).append(r)

    rows = []

    for (category_id, category_name), items in sorted(grouped.items()):
        total = len(items)
        main_count = sum(1 for x in items if x["relation"] == "main")
        variant_count = sum(1 for x in items if x["relation"] == "variant")
        non_variant_count = sum(1 for x in items if x["relation"] == "non_variant")

        edge_split_count = sum(
            1 for x in items
            if x["relation"] == "variant" and x.get("variant_type") == "edge_split"
        )

        node_dup_count = sum(
            1 for x in items
            if x["relation"] == "variant"
            and x.get("variant_type") == "node_duplication_or_extended_cover"
        )

        rows.append({
            "category_id": category_id,
            "category_name": category_name,
            "total": total,
            "main_count": main_count,
            "variant_count": variant_count,
            "non_variant_count": non_variant_count,
            "edge_split_variant_count": edge_split_count,
            "node_duplication_variant_count": node_dup_count,
            "main_ratio": main_count / total if total else 0.0,
            "variant_ratio": variant_count / total if total else 0.0,
            "non_variant_ratio": non_variant_count / total if total else 0.0,
            "explained_ratio": (main_count + variant_count) / total if total else 0.0,
        })

    return rows


def summarize_overall(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    main_count = sum(1 for x in records if x["relation"] == "main")
    variant_count = sum(1 for x in records if x["relation"] == "variant")
    non_variant_count = sum(1 for x in records if x["relation"] == "non_variant")

    edge_split_count = sum(
        1 for x in records
        if x["relation"] == "variant" and x.get("variant_type") == "edge_split"
    )

    node_dup_count = sum(
        1 for x in records
        if x["relation"] == "variant"
        and x.get("variant_type") == "node_duplication_or_extended_cover"
    )

    return {
        "total": total,
        "main_count": main_count,
        "variant_count": variant_count,
        "non_variant_count": non_variant_count,
        "edge_split_variant_count": edge_split_count,
        "node_duplication_variant_count": node_dup_count,
        "main_ratio": main_count / total if total else 0.0,
        "variant_ratio": variant_count / total if total else 0.0,
        "non_variant_ratio": non_variant_count / total if total else 0.0,
        "explained_ratio": (main_count + variant_count) / total if total else 0.0,
    }


# ============================================================
# Output writers
# ============================================================

def write_records_csv(records: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "sample_name",
        "instance_id",
        "category_id",
        "category_name",
        "relation",
        "variant_type",
        "reason",
        "feature_original_face_ids",
        "main_num_nodes",
        "candidate_num_nodes",
        "main_num_edges",
        "direct_num_edges",
        "generalized_num_edges",
        "graph_json",
        "label_json",
        "main_type_json",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in records:
            writer.writerow({
                "sample_name": r["sample_name"],
                "instance_id": r["instance_id"],
                "category_id": r["category_id"],
                "category_name": r["category_name"],
                "relation": r["relation"],
                "variant_type": r.get("variant_type", ""),
                "reason": r["reason"],
                "feature_original_face_ids": json.dumps(r["feature_original_face_ids"]),
                "main_num_nodes": r["main_num_nodes"],
                "candidate_num_nodes": r["candidate_num_nodes"],
                "main_num_edges": r["main_num_edges"],
                "direct_num_edges": r["direct_num_edges"],
                "generalized_num_edges": r["generalized_num_edges"],
                "graph_json": r["graph_json"],
                "label_json": r["label_json"],
                "main_type_json": r["main_type_json"],
            })


def write_category_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "category_id",
        "category_name",
        "total",
        "main_count",
        "variant_count",
        "non_variant_count",
        "edge_split_variant_count",
        "node_duplication_variant_count",
        "main_ratio",
        "variant_ratio",
        "non_variant_ratio",
        "explained_ratio",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in rows:
            writer.writerow(r)


def write_member_lists(records: List[Dict[str, Any]], out_dir: Path):
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}

    for r in records:
        key = (int(r["category_id"]), r["category_name"])
        grouped.setdefault(key, []).append(r)

    for (category_id, category_name), items in grouped.items():
        category_dir = out_dir / f"{category_id:02d}_{safe_name(category_name)}"
        category_dir.mkdir(parents=True, exist_ok=True)

        for relation in ["main", "variant", "non_variant"]:
            path = category_dir / f"{relation}_members.txt"

            with path.open("w", encoding="utf-8") as f:
                for r in items:
                    if r["relation"] != relation:
                        continue

                    f.write(
                        f"{r['sample_name']} | "
                        f"I{r['instance_id']} | "
                        f"faces={r['feature_original_face_ids']} | "
                        f"reason={r['reason']} | "
                        f"variant_type={r.get('variant_type', '')}\n"
                    )


# ============================================================
# Plotting
# ============================================================

def plot_category_ratios(rows: List[Dict[str, Any]], out_path: Path):
    rows = sorted(rows, key=lambda r: int(r["category_id"]))

    labels = [
        f"{r['category_id']}:{r['category_name']}"
        for r in rows
    ]

    main_vals = [r["main_ratio"] for r in rows]
    variant_vals = [r["variant_ratio"] for r in rows]
    non_variant_vals = [r["non_variant_ratio"] for r in rows]

    x = list(range(len(rows)))

    plt.figure(figsize=(max(12, len(rows) * 0.7), 7))

    plt.bar(x, main_vals, label="main type")
    plt.bar(x, variant_vals, bottom=main_vals, label="variant")

    bottoms = [m + v for m, v in zip(main_vals, variant_vals)]
    plt.bar(x, non_variant_vals, bottom=bottoms, label="non-variant")

    plt.xticks(x, labels, rotation=65, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Instance ratio")
    plt.title("Generalized main topology hypothesis by machining feature")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_category_counts(rows: List[Dict[str, Any]], out_path: Path):
    rows = sorted(rows, key=lambda r: int(r["category_id"]))

    labels = [
        f"{r['category_id']}:{r['category_name']}"
        for r in rows
    ]

    main_vals = [r["main_count"] for r in rows]
    variant_vals = [r["variant_count"] for r in rows]
    non_variant_vals = [r["non_variant_count"] for r in rows]

    x = list(range(len(rows)))

    plt.figure(figsize=(max(12, len(rows) * 0.7), 7))

    plt.bar(x, main_vals, label="main type")
    plt.bar(x, variant_vals, bottom=main_vals, label="variant")

    bottoms = [m + v for m, v in zip(main_vals, variant_vals)]
    plt.bar(x, non_variant_vals, bottom=bottoms, label="non-variant")

    plt.xticks(x, labels, rotation=65, ha="right")
    plt.ylabel("Instance count")
    plt.title("Generalized main topology hypothesis counts by machining feature")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_overall(overall: Dict[str, Any], out_path: Path):
    labels = ["all features"]

    main_vals = [overall["main_ratio"]]
    variant_vals = [overall["variant_ratio"]]
    non_variant_vals = [overall["non_variant_ratio"]]

    x = [0]

    plt.figure(figsize=(6, 6))

    plt.bar(x, main_vals, label="main type")
    plt.bar(x, variant_vals, bottom=main_vals, label="variant")

    bottoms = [main_vals[0] + variant_vals[0]]
    plt.bar(x, non_variant_vals, bottom=bottoms, label="non-variant")

    plt.xticks(x, labels)
    plt.ylim(0, 1.0)
    plt.ylabel("Instance ratio")
    plt.title("Overall generalized main topology hypothesis")
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ensure_dir(RESULT_DIR)

    print("Loading main topology types...")
    main_type_graphs = load_main_type_graphs(
        topology_types_dir=TOPOLOGY_TYPES_DIR,
        version=MAIN_VERSION,
    )

    print(f"Loaded main types: {len(main_type_graphs)} categories")

    print("\nProcessing dataset instances...")
    records = process_dataset_instances(
        graph_dir=GRAPH_DIR,
        label_dir=LABEL_DIR,
        main_type_graphs=main_type_graphs,
    )

    print(f"\nTotal processed feature instances: {len(records)}")

    category_summary = summarize_by_category(records)
    overall_summary = summarize_overall(records)

    save_json(records, RESULT_DIR / "instance_classification_details.json")
    save_json(category_summary, RESULT_DIR / "category_summary.json")
    save_json(overall_summary, RESULT_DIR / "overall_summary.json")

    write_records_csv(records, RESULT_DIR / "instance_classification_details.csv")
    write_category_csv(category_summary, RESULT_DIR / "category_summary.csv")

    write_member_lists(records, RESULT_DIR / "member_lists")

    if DRAW_FIGURES:
        plot_category_ratios(
            category_summary,
            RESULT_DIR / "figures" / "category_ratios.png",
        )

        plot_category_counts(
            category_summary,
            RESULT_DIR / "figures" / "category_counts.png",
        )

        plot_overall(
            overall_summary,
            RESULT_DIR / "figures" / "overall_ratio.png",
        )

    print("\n" + "#" * 100)
    print("Done.")
    print(f"Output folder: {RESULT_DIR.resolve()}")

    print("\nOverall:")
    print(
        f"total={overall_summary['total']} | "
        f"main={overall_summary['main_count']} ({overall_summary['main_ratio']:.2%}) | "
        f"variant={overall_summary['variant_count']} ({overall_summary['variant_ratio']:.2%}) | "
        f"non_variant={overall_summary['non_variant_count']} ({overall_summary['non_variant_ratio']:.2%}) | "
        f"explained={overall_summary['explained_ratio']:.2%}"
    )

    print("\nMain outputs:")
    print(f"  {RESULT_DIR / 'category_summary.csv'}")
    print(f"  {RESULT_DIR / 'instance_classification_details.csv'}")
    print(f"  {RESULT_DIR / 'overall_summary.json'}")
    print(f"  {RESULT_DIR / 'figures' / 'category_ratios.png'}")
    print(f"  {RESULT_DIR / 'figures' / 'overall_ratio.png'}")


if __name__ == "__main__":
    main()