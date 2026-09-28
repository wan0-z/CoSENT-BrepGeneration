# prepare_mftrcad_ui_cache_single_or_dir.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx

# =============================
# 需要 pythonocc-core
# =============================
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods


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
    'stock'                         # 24
]

CATEGORY_TO_COLOR = {
    0: (0.0, 1.0, 0.0),
    1: (0.0, 0.0, 1.0),
    2: (1.0, 1.0, 0.0),
    3: (1.0, 0.5, 0.0),
    4: (0.0, 1.0, 1.0),
    5: (1.0, 0.0, 1.0),
    6: (0.5, 0.5, 0.5),
    7: (0.6, 0.4, 0.2),
    8: (0.5, 0.0, 0.5),
    9: (0.4, 0.8, 0.8),
    10: (0.9, 0.7, 0.5),
    11: (0.7, 0.3, 0.0),
    12: (0.6, 0.6, 0.0),
    13: (0.8, 0.0, 0.1),
    14: (0.0, 0.5, 0.9),
    15: (0.8, 0.4, 0.7),
    16: (1.0, 0.8, 0.0),
    17: (0.3, 0.9, 0.3),
    18: (0.7, 0.0, 1.0),
    19: (0.5, 1.0, 0.5),
    20: (0.2, 0.8, 1.0),
    21: (0.8, 0.5, 0.0),
    22: (0.6, 0.8, 0.4),
    23: (0.9, 0.5, 0.5),
    24: (0.5, 0.7, 0.9)
}

MACHINING_CATEGORY_IDS: Set[int] = set(range(0, 24))
STOCK_CATEGORY_ID = 24
LAYOUT_SEED = 42

LIGHT_GRAY = "#E0E0E0"
MID_GRAY = "#888888"


def rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)),
        int(round(g * 255)),
        int(round(b * 255)),
    )


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


# ============================================================
# graph json parsing
# 支持：
#   1. [sample_name, graph_data]
#   2. graph_data
# ============================================================

def unwrap_graph_json(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[1], dict):
        return obj[1]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported graph json format.")


def get_instance_name_from_graph(graph_path: Path, graph_obj: Any) -> str:
    if isinstance(graph_obj, list) and len(graph_obj) >= 1 and isinstance(graph_obj[0], str):
        return safe_name(graph_obj[0])
    if isinstance(graph_obj, dict) and "filename" in graph_obj:
        return safe_name(graph_obj["filename"])
    return safe_name(graph_path.stem)


def build_graph(graph_data: Dict[str, Any]) -> nx.Graph:
    ginfo = graph_data["graph"]
    num_nodes = int(ginfo["num_nodes"])
    edges = ginfo["edges"]

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    # AAGNet 格式：edges = [[src...], [dst...]]
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


# ============================================================
# label json parsing
# 支持你的上传格式：
#   [
#     [
#       sample_name,
#       {
#         "seg": {"0": 24, "1": 3, ...},
#         "inst": [[...], [...], ...],
#         "bottom": {"0": 0, ...}
#       }
#     ]
#   ]
#
# 同时兼容：
#   [sample_name, label_data]
#   label_data
# ============================================================

def unwrap_label_json(obj: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Return:
        sample_name, label_data
    """
    # Your uploaded format:
    # [[sample_name, label_data]]
    if (
        isinstance(obj, list)
        and len(obj) == 1
        and isinstance(obj[0], list)
        and len(obj[0]) >= 2
        and isinstance(obj[0][1], dict)
    ):
        sample_name = obj[0][0] if isinstance(obj[0][0], str) else None
        return sample_name, obj[0][1]

    # Common pair format:
    # [sample_name, label_data]
    if (
        isinstance(obj, list)
        and len(obj) >= 2
        and isinstance(obj[0], str)
        and isinstance(obj[1], dict)
    ):
        return obj[0], obj[1]

    # Direct dict:
    # {"seg": ..., "inst": ..., "bottom": ...}
    if isinstance(obj, dict):
        return obj.get("filename", None), obj

    raise ValueError("Unsupported label json format.")


def parse_cls(label_data: Dict[str, Any]) -> Dict[int, int]:
    """
    你的 label 文件里没有 cls 字段。
    seg 才是 face_id -> category_id。

    兼容：
        label_data["seg"] = {"0": 24, "1": 3, ...}
        label_data["cls"] = {"0": 24, "1": 3, ...}
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
    你的 inst 是 NxN 矩阵：
        inst[i][j] == 1 表示 face i 和 face j 属于同一个 feature instance。

    做法：
        1. 建一个 instance relation graph
        2. 对 inst[i][j] == 1 的 machining faces 加边
        3. connected_components 得到 feature instances
        4. 对没有边但属于 machining category 的单面特征，补成 singleton instance

    stock, category_id=24，不作为 machining feature instance 输出。
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

    if not isinstance(inst_matrix, list):
        raise TypeError("label_data['inst'] should be a 2D list.")

    rows = min(len(inst_matrix), num_nodes)

    for i in range(rows):
        row = inst_matrix[i]
        if not isinstance(row, list):
            continue

        cols = min(len(row), num_nodes)

        for j in range(cols):
            if i == j:
                continue

            if int(row[j]) != 1:
                continue

            # 只处理 machining feature face，排除 stock face
            ci = cls_map.get(i, STOCK_CATEGORY_ID)
            cj = cls_map.get(j, STOCK_CATEGORY_ID)

            if ci not in MACHINING_CATEGORY_IDS:
                continue
            if cj not in MACHINING_CATEGORY_IDS:
                continue

            relation_graph.add_edge(i, j)

    feature_instances = []
    used_face_sets: Set[Tuple[int, ...]] = set()

    for comp in nx.connected_components(relation_graph):
        face_ids = sorted(int(x) for x in comp)

        if not face_ids:
            continue

        category_id = infer_feature_category(face_ids, cls_map)
        if category_id is None:
            continue

        face_tuple = tuple(face_ids)
        if face_tuple in used_face_sets:
            continue

        used_face_sets.add(face_tuple)

        feature_instances.append({
            "instance_id": len(feature_instances),
            "category_id": int(category_id),
            "category_name": FACE_CATEGORIES[int(category_id)],
            "nodes": face_ids,
        })

    return feature_instances


def parse_feature_instances(
    label_data: Dict[str, Any],
    cls_map: Dict[int, int],
    num_nodes: int,
) -> List[Dict[str, Any]]:
    """
    统一入口。

    如果 label_data['seg'] 是 list，兼容你原来写的：
        seg[instance_id] = [face ids]

    如果 label_data['seg'] 是 dict，使用新的 MFInstSeg/AAGNet 格式：
        seg[face_id] = category_id
        inst = NxN instance matrix
    """
    seg = label_data.get("seg", None)

    # 旧格式：seg 是 instance list
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
                "category_name": FACE_CATEGORIES[category_id],
                "nodes": face_ids,
            })

        return instances

    # 新格式：seg 是 face category dict，inst 是 NxN instance matrix
    if isinstance(seg, dict):
        return parse_feature_instances_from_inst_matrix(
            label_data=label_data,
            cls_map=cls_map,
            num_nodes=num_nodes,
        )

    raise ValueError("Unsupported label_data['seg'] format.")


def build_node_to_instance(feature_instances: List[Dict[str, Any]]) -> Dict[int, int]:
    node_to_instance = {}

    for inst in feature_instances:
        iid = inst["instance_id"]
        for n in inst["nodes"]:
            if n not in node_to_instance:
                node_to_instance[int(n)] = int(iid)

    return node_to_instance


# ============================================================
# drawing functions
# ============================================================

def make_full_graph_node_labels(
    G: nx.Graph,
    cls_map: Dict[int, int],
    node_to_instance: Dict[int, int],
) -> Dict[int, str]:
    labels = {}

    for n in G.nodes:
        cat_id = cls_map.get(int(n), None)
        cat_name = FACE_CATEGORIES[cat_id] if cat_id is not None and 0 <= cat_id < len(FACE_CATEGORIES) else "unknown"

        if n in node_to_instance:
            labels[n] = f"{n}\nI{node_to_instance[n]}\n{cat_name}"
        else:
            labels[n] = f"{n}\n{cat_name}"

    return labels


def make_subgraph_node_labels(
    G: nx.Graph,
    cls_map: Dict[int, int],
    feature_nodes: Set[int],
    instance_id: int,
) -> Dict[int, str]:
    labels = {}

    for n in G.nodes:
        cat_id = cls_map.get(int(n), None)
        cat_name = FACE_CATEGORIES[cat_id] if cat_id is not None and 0 <= cat_id < len(FACE_CATEGORIES) else "unknown"

        if n in feature_nodes:
            labels[n] = f"{n}\nI{instance_id}\n{cat_name}"
        else:
            labels[n] = f"{n}\n{cat_name}"

    return labels


def full_graph_node_colors_by_category(
    G: nx.Graph,
    cls_map: Dict[int, int],
    node_to_instance: Dict[int, int],
) -> List[str]:
    colors = []

    for n in G.nodes:
        cat_id = cls_map.get(int(n), None)

        if n in node_to_instance and cat_id in CATEGORY_TO_COLOR:
            colors.append(rgb01_to_hex(CATEGORY_TO_COLOR[cat_id]))
        else:
            colors.append(LIGHT_GRAY)

    return colors


def draw_full_graph(
    G: nx.Graph,
    cls_map: Dict[int, int],
    feature_instances: List[Dict[str, Any]],
    out_path: Path,
    title: str,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    node_to_instance = build_node_to_instance(feature_instances)
    labels = make_full_graph_node_labels(G, cls_map, node_to_instance)
    node_colors = full_graph_node_colors_by_category(G, cls_map, node_to_instance)

    plt.figure(figsize=(max(12, math.sqrt(G.number_of_nodes()) * 2.4), 10))
    pos = nx.spring_layout(G, seed=LAYOUT_SEED)

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=["#A8A8A8"] * G.number_of_edges(),
        width=1.2,
        alpha=0.8,
    )

    node_sizes = [1000 if n in node_to_instance else 650 for n in G.nodes]
    node_linewidths = [2.0 if n in node_to_instance else 0.8 for n in G.nodes]

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=node_linewidths,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=6.8,
        font_color="black",
    )

    handles = []
    used_cats = sorted({inst["category_id"] for inst in feature_instances})

    for cat_id in used_cats:
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=f"{cat_id}: {FACE_CATEGORIES[cat_id]}",
                markerfacecolor=rgb01_to_hex(CATEGORY_TO_COLOR[cat_id]),
                markeredgecolor="black",
                markersize=9,
            )
        )

    if handles:
        plt.legend(handles=handles, loc="best", fontsize=8, frameon=True)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def draw_feature_subgraph(
    G: nx.Graph,
    cls_map: Dict[int, int],
    inst: Dict[str, Any],
    out_path: Path,
    title: str,
    include_one_hop_context: bool,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    instance_id = inst["instance_id"]
    category_id = inst["category_id"]
    feature_color = rgb01_to_hex(CATEGORY_TO_COLOR[category_id])

    feature_nodes: Set[int] = set(inst["nodes"])

    if include_one_hop_context:
        context_nodes = set(feature_nodes)
        for n in feature_nodes:
            context_nodes.update(G.neighbors(n))
    else:
        context_nodes = set(feature_nodes)

    SG = G.subgraph(sorted(context_nodes)).copy()

    internal_edges = []
    boundary_edges = []
    external_edges = []

    for u, v in SG.edges:
        u_in = u in feature_nodes
        v_in = v in feature_nodes

        if u_in and v_in:
            internal_edges.append((u, v))
        elif u_in or v_in:
            boundary_edges.append((u, v))
        else:
            external_edges.append((u, v))

    plt.figure(figsize=(7.8, 6.8))
    pos = nx.spring_layout(SG, seed=LAYOUT_SEED)

    if external_edges:
        nx.draw_networkx_edges(
            SG,
            pos,
            edgelist=external_edges,
            edge_color=["#D0D0D0"] * len(external_edges),
            width=1.0,
            alpha=0.5,
        )

    if internal_edges:
        nx.draw_networkx_edges(
            SG,
            pos,
            edgelist=internal_edges,
            edge_color=["#000000"] * len(internal_edges),
            width=2.6,
            alpha=0.95,
        )

    if boundary_edges:
        nx.draw_networkx_edges(
            SG,
            pos,
            edgelist=boundary_edges,
            edge_color=["#666666"] * len(boundary_edges),
            width=1.8,
            alpha=0.8,
            style="dashed",
        )

    feature_node_list = sorted(feature_nodes)
    context_node_list = sorted(set(SG.nodes) - feature_nodes)

    if context_node_list:
        nx.draw_networkx_nodes(
            SG,
            pos,
            nodelist=context_node_list,
            node_color=[LIGHT_GRAY] * len(context_node_list),
            node_size=560,
            edgecolors=[MID_GRAY] * len(context_node_list),
            linewidths=1.0,
            alpha=0.95,
        )

    nx.draw_networkx_nodes(
        SG,
        pos,
        nodelist=feature_node_list,
        node_color=[feature_color] * len(feature_node_list),
        node_size=980,
        edgecolors=["black"] * len(feature_node_list),
        linewidths=2.2,
    )

    labels = make_subgraph_node_labels(SG, cls_map, feature_nodes, instance_id)

    nx.draw_networkx_labels(
        SG,
        pos,
        labels=labels,
        font_size=7,
        font_color="black",
    )

    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label=f"I{instance_id} | {category_id}:{FACE_CATEGORIES[category_id]}",
            markerfacecolor=feature_color,
            markeredgecolor="black",
            markersize=10,
        )
    ]

    if include_one_hop_context:
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                label="one-hop external faces",
                markerfacecolor=LIGHT_GRAY,
                markeredgecolor=MID_GRAY,
                markersize=9,
            )
        )

        legend_handles.append(
            plt.Line2D(
                [0], [0],
                linestyle="--",
                color="#666666",
                label="instance-external connection",
            )
        )

    plt.legend(handles=legend_handles, loc="best", fontsize=8)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# STEP mesh export
# ============================================================

def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Cannot read STEP file: {step_path}")

    ok = reader.TransferRoots()

    if ok == 0:
        raise RuntimeError(f"Cannot transfer STEP roots: {step_path}")

    shape = reader.OneShape()

    return shape


def export_face_meshes(
    step_path: Path,
    out_json_path: Path,
    cls_map: Dict[int, int],
    node_to_instance: Dict[int, int],
):
    """
    将 STEP 按 face 导出三角网格。
    每个 face 对应一个 mesh trace，前端 hover/pick 就能知道 face node id。
    """
    shape = read_step_shape(step_path)

    BRepMesh_IncrementalMesh(shape, 0.8, False, 0.5, True)

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_idx = 0
    face_meshes = []

    while explorer.More():
        face = topods.Face(explorer.Current())

        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        cat_id = int(cls_map.get(face_idx, -1))

        if cat_id in CATEGORY_TO_COLOR:
            color_hex = rgb01_to_hex(CATEGORY_TO_COLOR[cat_id])
            cat_name = FACE_CATEGORIES[cat_id]
        else:
            color_hex = LIGHT_GRAY
            cat_name = "unknown"

        if triangulation is None:
            face_meshes.append({
                "face_id": face_idx,
                "category_id": cat_id,
                "category_name": cat_name,
                "instance_id": node_to_instance.get(face_idx, None),
                "color": color_hex,
                "x": [],
                "y": [],
                "z": [],
                "i": [],
                "j": [],
                "k": [],
            })

            face_idx += 1
            explorer.Next()
            continue

        transform = loc.Transformation()

        x, y, z = [], [], []

        for i_node in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i_node)
            p.Transform(transform)

            x.append(float(p.X()))
            y.append(float(p.Y()))
            z.append(float(p.Z()))

        ii, jj, kk = [], [], []
        is_reversed = face.Orientation() == TopAbs_REVERSED

        for i_tri in range(1, triangulation.NbTriangles() + 1):
            t = triangulation.Triangle(i_tri)
            n1, n2, n3 = t.Get()

            n1 -= 1
            n2 -= 1
            n3 -= 1

            if is_reversed:
                n2, n3 = n3, n2

            ii.append(int(n1))
            jj.append(int(n2))
            kk.append(int(n3))

        face_meshes.append({
            "face_id": face_idx,
            "category_id": cat_id,
            "category_name": cat_name,
            "instance_id": node_to_instance.get(face_idx, None),
            "color": color_hex,
            "x": x,
            "y": y,
            "z": z,
            "i": ii,
            "j": jj,
            "k": kk,
        })

        face_idx += 1
        explorer.Next()

    save_json(face_meshes, out_json_path)

    return len(face_meshes)


# ============================================================
# file pairing
# ============================================================

def find_same_stem_file(folder: Path, stem: str, suffixes: List[str]) -> Optional[Path]:
    for suffix in suffixes:
        p = folder / f"{stem}{suffix}"
        if p.exists():
            return p

    for suffix in suffixes:
        matched = sorted(folder.glob(f"{stem}*{suffix}"))
        if matched:
            return matched[0]

    return None


def collect_samples_from_explicit_files(
    graph_path: Path,
    label_path: Path,
    step_path: Path,
) -> List[Tuple[Path, Path, Path]]:
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if not step_path.exists():
        raise FileNotFoundError(f"STEP file not found: {step_path}")

    return [(graph_path, label_path, step_path)]


def collect_samples_from_dirs(
    graph_dir: Path,
    label_dir: Path,
    step_dir: Path,
) -> List[Tuple[Path, Path, Path]]:
    graph_files = sorted(graph_dir.glob("*.json"))

    samples = []

    for graph_path in graph_files:
        if graph_path.name == "attr_stat.json":
            continue

        raw_graph_obj = load_json(graph_path)
        sample_name = get_instance_name_from_graph(graph_path, raw_graph_obj)

        label_path = find_same_stem_file(label_dir, sample_name, [".json"])
        step_path = find_same_stem_file(step_dir, sample_name, [".step", ".stp", ".STEP", ".STP"])

        if label_path is None:
            print(f"[WARN] label not found for {sample_name}")
            continue

        if step_path is None:
            print(f"[WARN] step not found for {sample_name}")
            continue

        samples.append((graph_path, label_path, step_path))

    return samples


# ============================================================
# process
# ============================================================

def process_one_sample(
    graph_path: Path,
    label_path: Path,
    step_path: Path,
    cache_dir: Path,
) -> Dict[str, Any]:
    full_graph_dir = cache_dir / "graphs" / "full"
    sub_with_dir = cache_dir / "subgraphs" / "with_context"
    sub_no_dir = cache_dir / "subgraphs" / "no_context"
    mesh_dir = cache_dir / "meshes"

    full_graph_dir.mkdir(parents=True, exist_ok=True)
    sub_with_dir.mkdir(parents=True, exist_ok=True)
    sub_no_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir.mkdir(parents=True, exist_ok=True)

    raw_graph_obj = load_json(graph_path)
    graph_data = unwrap_graph_json(raw_graph_obj)

    graph_sample_name = get_instance_name_from_graph(graph_path, raw_graph_obj)

    raw_label_obj = load_json(label_path)
    label_sample_name, label_data = unwrap_label_json(raw_label_obj)

    if label_sample_name:
        sample_name = safe_name(label_sample_name)
    else:
        sample_name = graph_sample_name

    G = build_graph(graph_data)

    cls_map = parse_cls(label_data)

    feature_instances = parse_feature_instances(
        label_data=label_data,
        cls_map=cls_map,
        num_nodes=G.number_of_nodes(),
    )

    node_to_instance = build_node_to_instance(feature_instances)

    # ---- full graph png
    full_graph_png = full_graph_dir / f"{sample_name}_graph.png"

    draw_full_graph(
        G=G,
        cls_map=cls_map,
        feature_instances=feature_instances,
        out_path=full_graph_png,
        title=f"{sample_name} | full graph (category colors)",
    )

    # ---- subgraphs
    subgraph_entries = []

    for inst in feature_instances:
        cat_name = inst["category_name"]
        nodes_str = "-".join(map(str, inst["nodes"]))

        sub_file_name = (
            f"{sample_name}"
            f"__seg{inst['instance_id']:03d}"
            f"__cls{inst['category_id']:02d}_{safe_name(cat_name)}"
            f"__nodes_{nodes_str}.png"
        )

        sub_with_path = sub_with_dir / safe_name(cat_name) / sub_file_name
        sub_no_path = sub_no_dir / safe_name(cat_name) / sub_file_name

        draw_feature_subgraph(
            G=G,
            cls_map=cls_map,
            inst=inst,
            out_path=sub_with_path,
            title=(
                f"{sample_name} | instance I{inst['instance_id']} | "
                f"{inst['category_id']}:{cat_name} | faces={inst['nodes']} | one-hop context"
            ),
            include_one_hop_context=True,
        )

        draw_feature_subgraph(
            G=G,
            cls_map=cls_map,
            inst=inst,
            out_path=sub_no_path,
            title=(
                f"{sample_name} | instance I{inst['instance_id']} | "
                f"{inst['category_id']}:{cat_name} | faces={inst['nodes']} | no context"
            ),
            include_one_hop_context=False,
        )

        subgraph_entries.append({
            "instance_id": inst["instance_id"],
            "category_id": inst["category_id"],
            "category_name": inst["category_name"],
            "nodes": inst["nodes"],
            "image_with_context": str(sub_with_path.relative_to(cache_dir)).replace("\\", "/"),
            "image_no_context": str(sub_no_path.relative_to(cache_dir)).replace("\\", "/"),
        })

    # ---- 3d mesh cache
    mesh_json_path = mesh_dir / f"{sample_name}_faces.json"

    num_step_faces = export_face_meshes(
        step_path=step_path,
        out_json_path=mesh_json_path,
        cls_map=cls_map,
        node_to_instance=node_to_instance,
    )

    entry = {
        "sample_name": sample_name,
        "graph_json": str(graph_path.resolve()),
        "label_json": str(label_path.resolve()),
        "step_file": str(step_path.resolve()),
        "num_graph_nodes": G.number_of_nodes(),
        "num_graph_edges": G.number_of_edges(),
        "num_step_faces": num_step_faces,
        "full_graph_image": str(full_graph_png.relative_to(cache_dir)).replace("\\", "/"),
        "mesh_json": str(mesh_json_path.relative_to(cache_dir)).replace("\\", "/"),
        "subgraphs": subgraph_entries,
    }

    print(
        f"[OK] {sample_name}: "
        f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, "
        f"instances={len(feature_instances)}, step_faces={num_step_faces}"
    )

    return entry


def process_explicit_files(
    graph_path: Path,
    label_path: Path,
    step_path: Path,
    cache_dir: Path,
):
    samples = collect_samples_from_explicit_files(
        graph_path=graph_path,
        label_path=label_path,
        step_path=step_path,
    )

    sample_entries = []

    for graph_path, label_path, step_path in samples:
        try:
            entry = process_one_sample(
                graph_path=graph_path,
                label_path=label_path,
                step_path=step_path,
                cache_dir=cache_dir,
            )
            sample_entries.append(entry)
        except Exception as e:
            print(f"[ERROR] Failed on {graph_path.name}: {e}")

    index = {
        "cache_dir": str(cache_dir.resolve()),
        "samples": sample_entries,
    }

    save_json(index, cache_dir / "samples_index.json")

    print(f"\nDone. Cache written to: {cache_dir.resolve()}")


def process_dataset_dirs(
    graph_dir: Path,
    label_dir: Path,
    step_dir: Path,
    cache_dir: Path,
):
    samples = collect_samples_from_dirs(
        graph_dir=graph_dir,
        label_dir=label_dir,
        step_dir=step_dir,
    )

    if not samples:
        raise RuntimeError("No valid graph/label/step triples found.")

    sample_entries = []

    for graph_path, label_path, step_path in samples:
        try:
            entry = process_one_sample(
                graph_path=graph_path,
                label_path=label_path,
                step_path=step_path,
                cache_dir=cache_dir,
            )
            sample_entries.append(entry)
        except Exception as e:
            print(f"[ERROR] Failed on {graph_path.name}: {e}")

    index = {
        "cache_dir": str(cache_dir.resolve()),
        "samples": sample_entries,
    }

    save_json(index, cache_dir / "samples_index.json")

    print(f"\nDone. Cache written to: {cache_dir.resolve()}")


# ============================================================
# main config
# 你只需要改这里
# ============================================================

def main():
    # --------------------------------------------------------
    # 模式 1：处理单个样本
    # --------------------------------------------------------
    USE_SINGLE_SAMPLE = False
    DATASET_NAME = "mfinstseg"

    GRAPH_PATH = Path(fr"data\{DATASET_NAME}\graphs\20221123_142528_0.json")
    LABEL_PATH = Path(fr"data\{DATASET_NAME}\labels\20221123_142528_0.json")
    STEP_PATH = Path(fr"data\{DATASET_NAME}\steps\20221123_142528_0.step")

    # --------------------------------------------------------
    # 模式 2：处理整个文件夹
    # 如果 USE_SINGLE_SAMPLE = False，则使用下面三个文件夹
    # --------------------------------------------------------
    GRAPH_DIR = Path(fr"data\{DATASET_NAME}\graphs")
    LABEL_DIR = Path(fr"data\{DATASET_NAME}\labels")
    STEP_DIR = Path(fr"data\{DATASET_NAME}\steps")

    # 输出 cache 目录，保持你原来的输出结构
    CACHE_DIR = Path(fr"web_cache\{DATASET_NAME}")

    if USE_SINGLE_SAMPLE:
        process_explicit_files(
            graph_path=GRAPH_PATH,
            label_path=LABEL_PATH,
            step_path=STEP_PATH,
            cache_dir=CACHE_DIR,
        )
    else:
        process_dataset_dirs(
            graph_dir=GRAPH_DIR,
            label_dir=LABEL_DIR,
            step_dir=STEP_DIR,
            cache_dir=CACHE_DIR,
        )


if __name__ == "__main__":
    main()