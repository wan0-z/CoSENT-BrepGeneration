# check_topology_variants_surface_aware.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import isomorphism as iso


# ============================================================
# Config
# ============================================================
DATASET_NAME = "mfinstseg"
DATA_DIR = Path(f"data\{DATASET_NAME}")
GRAPH_DIR = DATA_DIR / "graphs"
LABEL_DIR = DATA_DIR / "labels"

# 原始 STEP 文件夹。你只需要改这里。
# 脚本会在 STEP_DIR 下递归搜索 .stp / .step / .STP / .STEP。
STEP_DIR = Path(r"C:\Users\go36sal\Downloads\data2\steps")
STEP_EXTENSIONS = [".stp", ".step", ".STP", ".STEP"]

# 这里读取你之前 analyze_mftrcad_subgraph_topologies.py 生成的 main type
TOPOLOGY_TYPES_DIR = Path(f"output\{DATASET_NAME}") / "topology_types"
MAIN_VERSION = "no_context"

RESULT_DIR = Path(f"generalized_main_topology_surface_aware_output\{DATASET_NAME}")

# 如果某个 feature instance 的节点特别多，组合搜索可能爆炸。
# 一般 machining feature 子图节点数不会太大。
MAX_COMBINATIONS_PER_NODE = 200000

DRAW_FIGURES = True

# OpenCascade surface equivalence tolerance.
SURFACE_TOL = 1e-6

# 如果 STEP 文件找不到 / OCC 不可用 / face 数量不一致：
# True  = 直接把该 sample 记为 error 并跳过；
# False = 保守 fallback：每个 face 自己成为一个 surface group。
#         这样不会把无法确认共面的 face 错判成 variant，但可能低估 variant 数量。
STRICT_STEP_SURFACE = False


# ============================================================
# Optional pythonOCC imports
# ============================================================

OCC_AVAILABLE = True
OCC_IMPORT_ERROR: Optional[Exception] = None

try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.TopAbs import TopAbs_FACE
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import topods
    from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
    from OCC.Core.GeomAbs import (
        GeomAbs_Plane,
        GeomAbs_Cylinder,
        GeomAbs_Cone,
        GeomAbs_Sphere,
        GeomAbs_Torus,
    )
except Exception as e:  # pragma: no cover - depends on local environment
    OCC_AVAILABLE = False
    OCC_IMPORT_ERROR = e


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
# STEP file search + OpenCascade surface extraction
# ============================================================

def build_step_index(step_dir: Path) -> Dict[str, List[Path]]:
    """
    递归扫描 STEP_DIR，建立 stem/name 的索引。
    后续每个 sample 自动查找所属 STEP 文件。
    """
    index: Dict[str, List[Path]] = {}

    if not step_dir.exists():
        print(f"[WARN] STEP_DIR does not exist: {step_dir}")
        return index

    files: List[Path] = []
    for ext in STEP_EXTENSIONS:
        files.extend(step_dir.rglob(f"*{ext}"))

    for p in sorted(set(files)):
        keys = {
            p.name.lower(),
            p.stem.lower(),
            safe_name(p.stem).lower(),
        }
        for k in keys:
            index.setdefault(k, []).append(p)

    print(f"[INFO] Indexed STEP files: {len(set(files))} from {step_dir}")
    return index


def find_step_file(
    step_dir: Path,
    step_index: Dict[str, List[Path]],
    sample_name: str,
    graph_path: Path,
) -> Optional[Path]:
    """
    自动搜索当前 graph / label 所属的 STEP 文件。

    优先级：
      1. sample_name + .stp/.step 精确匹配
      2. graph_path.stem + .stp/.step 精确匹配
      3. STEP_DIR 递归索引中的 stem/name 匹配
      4. STEP_DIR 递归搜索包含 sample_name 或 graph stem 的文件
    """
    if not step_dir.exists():
        return None

    raw_names = [
        sample_name,
        Path(sample_name).stem,
        graph_path.stem,
        safe_name(sample_name),
        safe_name(graph_path.stem),
    ]

    # Direct exact paths.
    for raw in raw_names:
        if not raw:
            continue
        raw_stem = Path(str(raw)).stem
        for ext in STEP_EXTENSIONS:
            p = step_dir / f"{raw_stem}{ext}"
            if p.exists():
                return p

    # Indexed exact match.
    for raw in raw_names:
        if not raw:
            continue
        keys = {
            str(raw).lower(),
            Path(str(raw)).stem.lower(),
            safe_name(Path(str(raw)).stem).lower(),
        }
        for k in keys:
            hits = step_index.get(k, [])
            if hits:
                return sorted(hits, key=lambda x: len(str(x)))[0]

    # Loose contains match.
    needles = [safe_name(Path(str(x)).stem).lower() for x in raw_names if x]
    needles = [x for x in dict.fromkeys(needles) if x]

    candidates: List[Path] = []
    for paths in step_index.values():
        for p in paths:
            stem = safe_name(p.stem).lower()
            if any(n in stem or stem in n for n in needles):
                candidates.append(p)

    if candidates:
        return sorted(set(candidates), key=lambda x: (len(x.stem), str(x)))[0]

    return None


def _q(x: float, tol: float = SURFACE_TOL) -> int:
    return int(round(float(x) / tol))


def _q_tuple(vals: Tuple[float, ...], tol: float = SURFACE_TOL) -> Tuple[int, ...]:
    return tuple(_q(v, tol) for v in vals)


def _dir_tuple(d: Any) -> Tuple[float, float, float]:
    return (float(d.X()), float(d.Y()), float(d.Z()))


def _point_tuple(p: Any) -> Tuple[float, float, float]:
    return (float(p.X()), float(p.Y()), float(p.Z()))


def _canonical_dir(dx: float, dy: float, dz: float) -> Tuple[float, float, float, int]:
    """
    统一方向符号，使同一无向轴线得到一致 key。
    返回 canonical direction 和 sign。
    """
    vals = [dx, dy, dz]
    sign = 1
    for v in vals:
        if abs(v) > SURFACE_TOL:
            if v < 0:
                sign = -1
            break
    return (dx * sign, dy * sign, dz * sign, sign)


def _cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _line_key(location: Any, direction: Any) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    用 canonical direction + moment(point x direction) 表示空间直线。
    对 cylinder/cone/torus axis 很实用。
    """
    dx, dy, dz = _dir_tuple(direction)
    cdx, cdy, cdz, sign = _canonical_dir(dx, dy, dz)
    p = _point_tuple(location)
    # 如果方向翻转，moment 也按 canonical direction 计算。
    moment = _cross(p, (cdx, cdy, cdz))
    return _q_tuple((cdx, cdy, cdz)), _q_tuple(moment)


def surface_key_from_occ_face(face: Any) -> Tuple[Any, ...]:
    """
    把 OCC face 的 underlying surface 转成可比较的几何 key。

    目标不是比较 trimmed face 边界，而是比较底层 surface。
    对常见机械特征最重要的是 plane/cylinder/cone/sphere/torus。
    非解析面 fallback 为 type + 参数范围；这类面一般不会大量出现在 MFInstSeg machining feature。
    """
    adaptor = BRepAdaptor_Surface(face, True)
    stype = adaptor.GetType()

    if stype == GeomAbs_Plane:
        pln = adaptor.Plane()
        normal = pln.Axis().Direction()
        loc = pln.Location()
        nx_, ny_, nz_ = _dir_tuple(normal)
        cnx, cny, cnz, sign = _canonical_dir(nx_, ny_, nz_)
        p = _point_tuple(loc)
        d = _dot((cnx, cny, cnz), p)
        return ("plane", _q_tuple((cnx, cny, cnz)), _q(d))

    if stype == GeomAbs_Cylinder:
        cyl = adaptor.Cylinder()
        axis = cyl.Axis()
        return (
            "cylinder",
            _line_key(axis.Location(), axis.Direction()),
            _q(float(cyl.Radius())),
        )

    if stype == GeomAbs_Cone:
        cone = adaptor.Cone()
        axis = cone.Axis()
        return (
            "cone",
            _line_key(axis.Location(), axis.Direction()),
            _q(float(cone.RefRadius())),
            _q(float(cone.SemiAngle())),
        )

    if stype == GeomAbs_Sphere:
        sph = adaptor.Sphere()
        return (
            "sphere",
            _q_tuple(_point_tuple(sph.Location())),
            _q(float(sph.Radius())),
        )

    if stype == GeomAbs_Torus:
        tor = adaptor.Torus()
        axis = tor.Axis()
        return (
            "torus",
            _line_key(axis.Location(), axis.Direction()),
            _q(float(tor.MajorRadius())),
            _q(float(tor.MinorRadius())),
        )

    # Fallback: 不轻易把自由曲面合并，避免误判。
    # 参数范围只是辅助信息；同一曲面被不同 trimming 后范围可能不同。
    # 所以这里保守地返回 surface type + rounded bounds。
    try:
        bounds = (
            float(adaptor.FirstUParameter()),
            float(adaptor.LastUParameter()),
            float(adaptor.FirstVParameter()),
            float(adaptor.LastVParameter()),
        )
    except Exception:
        bounds = (0.0, 0.0, 0.0, 0.0)

    return ("other_surface", int(stype), _q_tuple(bounds))


def read_step_face_surface_keys(step_path: Path) -> Tuple[Dict[int, Tuple[Any, ...]], Dict[str, Any]]:
    """
    返回：
      face_id/index -> underlying surface key

    重要假设：graph node id 与 STEP face 遍历顺序一致。
    如果你的 graph cache 生成时用了同一套 OCC face explorer，这通常成立。
    """
    if not OCC_AVAILABLE:
        raise RuntimeError(f"pythonOCC import failed: {OCC_IMPORT_ERROR}")

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"STEP read failed: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()

    face_to_key: Dict[int, Tuple[Any, ...]] = {}
    type_counts: Dict[str, int] = {}

    exp = TopExp_Explorer(shape, TopAbs_FACE)
    idx = 0

    while exp.More():
        face = topods.Face(exp.Current())
        key = surface_key_from_occ_face(face)
        face_to_key[idx] = key
        type_name = str(key[0])
        type_counts[type_name] = type_counts.get(type_name, 0) + 1
        idx += 1
        exp.Next()

    meta = {
        "step_path": str(step_path),
        "num_step_faces": idx,
        "surface_type_counts": type_counts,
        "occ_available": True,
    }
    return face_to_key, meta


def fallback_unique_surface_keys(num_nodes: int, reason: str) -> Tuple[Dict[int, Tuple[Any, ...]], Dict[str, Any]]:
    """
    保守 fallback：每个 face 都是独立 surface。
    这样不会把无法确认的 face 合并成 variant。
    """
    return (
        {i: ("fallback_unique_face", i) for i in range(num_nodes)},
        {
            "step_path": "",
            "num_step_faces": None,
            "surface_type_counts": {},
            "occ_available": OCC_AVAILABLE,
            "fallback_reason": reason,
        },
    )


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
    """
    if main_G.number_of_nodes() > candidate_G.number_of_nodes():
        return False

    if main_G.number_of_edges() > candidate_G.number_of_edges():
        return False

    main_nodes = list(main_G.nodes())
    cand_nodes = list(candidate_G.nodes())

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
    注意：返回图会按 feature_nodes sorted 顺序 relabel 为 0..n-1。
    因此 local node i 对应 sorted(feature_nodes)[i]。
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
    """
    if a == b:
        return False

    if full_G.has_edge(a, b):
        return True

    visited: Set[int] = set()
    queue: List[int] = []

    for nb in full_G.neighbors(a):
        nb = int(nb)

        if nb == b:
            return True

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
    注意：返回图会按 feature_nodes sorted 顺序 relabel 为 0..n-1。
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
# Surface-aware generalized main type coverage
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
    允许 subset 内部有额外边；main_G 的边必须存在。
    """
    H = graph_induced_by_nodes_relabel(generalized_G, subset_nodes)

    if H.number_of_nodes() != main_G.number_of_nodes():
        return False

    if H.number_of_edges() < main_G.number_of_edges():
        return False

    if category_multiset(H) != category_multiset(main_G):
        return False

    return main_embeds_in_candidate(
        main_G=main_G,
        candidate_G=H,
    )


def make_local_surface_groups(
    feature_nodes: List[int],
    face_to_surface_key: Dict[int, Tuple[Any, ...]],
) -> Tuple[Dict[int, int], Dict[int, List[int]], Dict[int, int]]:
    """
    当前 instance 内部：
      original face id -> surface key -> surface group id

    返回：
      local_node_to_group: relabeled local node id -> group id
      group_to_local_nodes: group id -> local node ids
      local_to_original: local node id -> original face id
    """
    feature_nodes = sorted(int(x) for x in feature_nodes)
    key_to_gid: Dict[Tuple[Any, ...], int] = {}
    local_node_to_group: Dict[int, int] = {}
    group_to_local_nodes: Dict[int, List[int]] = {}
    local_to_original: Dict[int, int] = {}

    for local_id, original_face_id in enumerate(feature_nodes):
        key = face_to_surface_key.get(original_face_id, ("missing_face_key", original_face_id))
        if key not in key_to_gid:
            key_to_gid[key] = len(key_to_gid)
        gid = key_to_gid[key]
        local_node_to_group[local_id] = gid
        group_to_local_nodes.setdefault(gid, []).append(local_id)
        local_to_original[local_id] = original_face_id

    return local_node_to_group, group_to_local_nodes, local_to_original


def node_is_covered_by_surface_aware_main(
    generalized_G: nx.Graph,
    node: int,
    main_G: nx.Graph,
    local_node_to_group: Dict[int, int],
    group_to_local_nodes: Dict[int, List[int]],
    max_combinations: int = MAX_COMBINATIONS_PER_NODE,
) -> Tuple[bool, Optional[List[int]], str]:
    """
    新 variant 定义的核心：

    当前 node 必须能与“其他 surface group”中的 face node 组成 main type。
    每个候选 subset 中，每个 surface group 最多选一个 face。
    """
    k = main_G.number_of_nodes()

    if k == 1:
        main_node = next(iter(main_G.nodes()))
        if node_label(generalized_G, node) == node_label(main_G, main_node):
            return True, [node], "single_node_main_surface_aware"
        return False, None, "single_node_label_mismatch"

    node_group = local_node_to_group[node]
    all_groups = sorted(group_to_local_nodes.keys())

    if len(all_groups) != k:
        return False, None, f"surface_group_count_{len(all_groups)}_not_equal_main_nodes_{k}"

    other_groups = [g for g in all_groups if g != node_group]

    checked = 0
    choices_per_group = [group_to_local_nodes[g] for g in other_groups]

    for picked in itertools.product(*choices_per_group):
        checked += 1

        if checked > max_combinations:
            return False, None, f"surface_aware_combination_limit_exceeded_{max_combinations}"

        subset = [node] + list(picked)

        # 理论上 product 已保证跨 group，这里再防御检查一次。
        subset_groups = [local_node_to_group[x] for x in subset]
        if len(set(subset_groups)) != len(subset):
            continue

        if subset_can_form_main_type(
            generalized_G=generalized_G,
            subset_nodes=subset,
            main_G=main_G,
        ):
            return True, sorted(int(x) for x in subset), f"surface_aware_covered_after_{checked}_checks"

    return False, None, f"surface_aware_not_covered_after_{checked}_checks"


def all_nodes_covered_by_surface_aware_main(
    generalized_G: nx.Graph,
    main_G: nx.Graph,
    local_node_to_group: Dict[int, int],
    group_to_local_nodes: Dict[int, List[int]],
    local_to_original: Dict[int, int],
) -> Tuple[bool, Dict[int, Dict[str, Any]], str]:
    """
    对当前 instance 每个 face node 检查：
      它是否能与其他 surface group 的 face node 组成 main type。
    """
    main_n = main_G.number_of_nodes()
    surface_group_count = len(group_to_local_nodes)

    if surface_group_count > main_n:
        return (
            False,
            {},
            f"surface_group_count_{surface_group_count}_greater_than_main_nodes_{main_n}",
        )

    if surface_group_count < main_n:
        return (
            False,
            {},
            f"surface_group_count_{surface_group_count}_less_than_main_nodes_{main_n}",
        )

    coverage: Dict[int, Dict[str, Any]] = {}

    for node in sorted(generalized_G.nodes()):
        ok, subset, reason = node_is_covered_by_surface_aware_main(
            generalized_G=generalized_G,
            node=node,
            main_G=main_G,
            local_node_to_group=local_node_to_group,
            group_to_local_nodes=group_to_local_nodes,
        )

        original_node = int(local_to_original[node])
        original_subset = None
        if subset is not None:
            original_subset = [int(local_to_original[x]) for x in subset]

        coverage[original_node] = {
            "covered": bool(ok),
            "local_node": int(node),
            "surface_group_id": int(local_node_to_group[node]),
            "covering_subset_local": subset,
            "covering_subset_original_face_ids": original_subset,
            "reason": reason,
        }

        if not ok:
            return False, coverage, f"face_{original_node}_not_covered:{reason}"

    return True, coverage, "all_faces_covered_by_surface_aware_main"


def classify_instance_against_main(
    full_G: nx.Graph,
    cls_map: Dict[int, int],
    inst: Dict[str, Any],
    main_G: nx.Graph,
    face_to_surface_key: Dict[int, Tuple[Any, ...]],
    step_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    按 surface-aware 新定义分类：

    main:
        当前 instance 的 direct feature graph 与 main type 完全同构。

    variant / edge_split:
        节点数与 main type 相同；direct graph 不是 main；
        generalized graph 中包含 main type。

    variant / surface_aware_node_split:
        face 节点数多于 main type；
        先按 STEP underlying surface 把 face 分组；
        surface group 数量必须等于 main type 节点数；
        每个 face node 都必须能和其他 surface group 的 face node 组成 main type。

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

    local_node_to_group, group_to_local_nodes, local_to_original = make_local_surface_groups(
        feature_nodes=feature_nodes,
        face_to_surface_key=face_to_surface_key,
    )

    main_n = main_G.number_of_nodes()
    cand_n = generalized_G.number_of_nodes()
    surface_group_count = len(group_to_local_nodes)

    surface_groups_as_original_faces = {
        int(g): [int(local_to_original[x]) for x in sorted(nodes)]
        for g, nodes in group_to_local_nodes.items()
    }

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

    common = {
        "main_num_nodes": main_n,
        "candidate_num_nodes": cand_n,
        "surface_group_count": surface_group_count,
        "surface_groups_as_original_faces": surface_groups_as_original_faces,
        "direct_num_edges": direct_G.number_of_edges(),
        "generalized_num_edges": generalized_G.number_of_edges(),
        "main_num_edges": main_G.number_of_edges(),
        "step_path": step_meta.get("step_path", ""),
        "step_num_faces": step_meta.get("num_step_faces", None),
        "step_surface_type_counts": step_meta.get("surface_type_counts", {}),
        "step_fallback_reason": step_meta.get("fallback_reason", ""),
    }

    if direct_iso:
        return {
            "relation": "main",
            "reason": "direct_graph_isomorphic_to_main",
            "coverage": {},
            **common,
        }

    if cand_n == main_n and generalized_contains_main:
        return {
            "relation": "variant",
            "variant_type": "edge_split",
            "reason": "same_node_count_generalized_graph_contains_main",
            "coverage": {},
            **common,
        }

    if cand_n > main_n:
        covered, coverage, reason = all_nodes_covered_by_surface_aware_main(
            generalized_G=generalized_G,
            main_G=main_G,
            local_node_to_group=local_node_to_group,
            group_to_local_nodes=group_to_local_nodes,
            local_to_original=local_to_original,
        )

        if covered:
            return {
                "relation": "variant",
                "variant_type": "surface_aware_node_split",
                "reason": reason,
                "coverage": coverage,
                **common,
            }

        return {
            "relation": "non_variant",
            "reason": reason,
            "coverage": coverage,
            **common,
        }

    return {
        "relation": "non_variant",
        "reason": (
            "same_node_count_but_generalized_graph_does_not_contain_main"
            if cand_n == main_n
            else "candidate_has_fewer_nodes_than_main"
        ),
        "coverage": {},
        **common,
    }


# ============================================================
# Dataset processing
# ============================================================

def collect_graph_files(graph_dir: Path) -> List[Path]:
    files = sorted(graph_dir.glob("*.json"))
    files = [p for p in files if p.name != "attr_stat.json"]
    return files


def get_surface_keys_for_sample(
    sample_name: str,
    graph_path: Path,
    full_G: nx.Graph,
    step_dir: Path,
    step_index: Dict[str, List[Path]],
    cache: Dict[str, Tuple[Dict[int, Tuple[Any, ...]], Dict[str, Any]]],
) -> Tuple[Dict[int, Tuple[Any, ...]], Dict[str, Any]]:
    step_path = find_step_file(
        step_dir=step_dir,
        step_index=step_index,
        sample_name=sample_name,
        graph_path=graph_path,
    )

    if step_path is None:
        msg = f"step_file_not_found_for_sample_{sample_name}"
        if STRICT_STEP_SURFACE:
            raise FileNotFoundError(msg)
        print(f"[WARN] {msg}; fallback each face as unique surface")
        return fallback_unique_surface_keys(full_G.number_of_nodes(), msg)

    key = str(step_path.resolve())
    if key in cache:
        return cache[key]

    try:
        face_to_surface_key, meta = read_step_face_surface_keys(step_path)

        if len(face_to_surface_key) != full_G.number_of_nodes():
            msg = (
                f"step_face_count_mismatch: step_faces={len(face_to_surface_key)}, "
                f"graph_nodes={full_G.number_of_nodes()}, step={step_path}"
            )
            if STRICT_STEP_SURFACE:
                raise RuntimeError(msg)
            print(f"[WARN] {msg}; fallback each face as unique surface")
            return fallback_unique_surface_keys(full_G.number_of_nodes(), msg)

        cache[key] = (face_to_surface_key, meta)
        return face_to_surface_key, meta

    except Exception as e:
        msg = f"step_surface_extraction_failed: {e}"
        if STRICT_STEP_SURFACE:
            raise
        print(f"[WARN] {msg}; fallback each face as unique surface")
        return fallback_unique_surface_keys(full_G.number_of_nodes(), msg)


def process_dataset_instances(
    graph_dir: Path,
    label_dir: Path,
    step_dir: Path,
    main_type_graphs: Dict[int, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    graph_files = collect_graph_files(graph_dir)

    if not graph_files:
        raise RuntimeError(f"No graph json files found in {graph_dir}")

    step_index = build_step_index(step_dir)
    step_surface_cache: Dict[str, Tuple[Dict[int, Tuple[Any, ...]], Dict[str, Any]]] = {}

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

            face_to_surface_key, step_meta = get_surface_keys_for_sample(
                sample_name=sample_name,
                graph_path=graph_path,
                full_G=full_G,
                step_dir=step_dir,
                step_index=step_index,
                cache=step_surface_cache,
            )

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
                    face_to_surface_key=face_to_surface_key,
                    step_meta=step_meta,
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
                f"instances={len(instances)}, "
                f"step={step_meta.get('step_path', '')}, "
                f"fallback={step_meta.get('fallback_reason', '')}"
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

        surface_aware_count = sum(
            1 for x in items
            if x["relation"] == "variant"
            and x.get("variant_type") == "surface_aware_node_split"
        )

        fallback_count = sum(1 for x in items if x.get("step_fallback_reason"))

        rows.append({
            "category_id": category_id,
            "category_name": category_name,
            "total": total,
            "main_count": main_count,
            "variant_count": variant_count,
            "non_variant_count": non_variant_count,
            "edge_split_variant_count": edge_split_count,
            "surface_aware_node_split_variant_count": surface_aware_count,
            "step_fallback_count": fallback_count,
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

    surface_aware_count = sum(
        1 for x in records
        if x["relation"] == "variant"
        and x.get("variant_type") == "surface_aware_node_split"
    )

    fallback_count = sum(1 for x in records if x.get("step_fallback_reason"))

    return {
        "total": total,
        "main_count": main_count,
        "variant_count": variant_count,
        "non_variant_count": non_variant_count,
        "edge_split_variant_count": edge_split_count,
        "surface_aware_node_split_variant_count": surface_aware_count,
        "step_fallback_count": fallback_count,
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
        "surface_group_count",
        "surface_groups_as_original_faces",
        "main_num_edges",
        "direct_num_edges",
        "generalized_num_edges",
        "step_path",
        "step_num_faces",
        "step_fallback_reason",
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
                "surface_group_count": r.get("surface_group_count", ""),
                "surface_groups_as_original_faces": json.dumps(r.get("surface_groups_as_original_faces", {}), ensure_ascii=False),
                "main_num_edges": r["main_num_edges"],
                "direct_num_edges": r["direct_num_edges"],
                "generalized_num_edges": r["generalized_num_edges"],
                "step_path": r.get("step_path", ""),
                "step_num_faces": r.get("step_num_faces", ""),
                "step_fallback_reason": r.get("step_fallback_reason", ""),
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
        "surface_aware_node_split_variant_count",
        "step_fallback_count",
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
                        f"surface_groups={r.get('surface_groups_as_original_faces', {})} | "
                        f"reason={r['reason']} | "
                        f"variant_type={r.get('variant_type', '')} | "
                        f"step={r.get('step_path', '')} | "
                        f"fallback={r.get('step_fallback_reason', '')}\n"
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
    plt.title("Surface-aware generalized main topology hypothesis by machining feature")
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
    plt.title("Surface-aware generalized main topology hypothesis counts by machining feature")
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
    plt.title("Overall surface-aware generalized main topology hypothesis")
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
    print(f"OCC available: {OCC_AVAILABLE}")
    if not OCC_AVAILABLE:
        print(f"[WARN] OCC import error: {OCC_IMPORT_ERROR}")
        if STRICT_STEP_SURFACE:
            raise RuntimeError(f"pythonOCC is required but unavailable: {OCC_IMPORT_ERROR}")

    print("\nProcessing dataset instances...")
    records = process_dataset_instances(
        graph_dir=GRAPH_DIR,
        label_dir=LABEL_DIR,
        step_dir=STEP_DIR,
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
        f"explained={overall_summary['explained_ratio']:.2%} | "
        f"step_fallback={overall_summary['step_fallback_count']}"
    )

    print("\nVariant details:")
    print(
        f"edge_split={overall_summary['edge_split_variant_count']} | "
        f"surface_aware_node_split={overall_summary['surface_aware_node_split_variant_count']}"
    )

    print("\nMain outputs:")
    print(f"  {RESULT_DIR / 'category_summary.csv'}")
    print(f"  {RESULT_DIR / 'instance_classification_details.csv'}")
    print(f"  {RESULT_DIR / 'overall_summary.json'}")
    print(f"  {RESULT_DIR / 'figures' / 'category_ratios.png'}")
    print(f"  {RESULT_DIR / 'figures' / 'overall_ratio.png'}")


if __name__ == "__main__":
    main()
