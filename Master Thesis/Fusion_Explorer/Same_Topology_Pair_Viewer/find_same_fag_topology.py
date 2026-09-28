# find_fag_topology_groups_cache_only_random_verbose.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
import itertools
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import networkx as nx
import pandas as pd
from tqdm import tqdm

# pythonocc-core / OpenCascade
# USE_CACHE_ONLY=True 时不会读取 STEP，不会调用 OCC。
# 但保留这些 import，方便 USE_CACHE_ONLY=False 时仍然可以从 STEP 提取 FAG。
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import (
    TopTools_IndexedMapOfShape,
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListIteratorOfListOfShape,
)


# ============================================================
# Config
# 你只需要改这里
# ============================================================

CONFIG = {
    # ========================================================
    # 数据来源模式
    # ========================================================
    # True:
    #   完全不扫描 STEP_ROOT，不读取 source STEP file。
    #   直接从 GRAPH_CACHE_DIR 读取之前保存好的 FAG json。
    #
    # False:
    #   从 STEP_ROOT 扫描 STEP，优先读 cache；
    #   cache 不存在或失效时，从 STEP 重新提取。
    "USE_CACHE_ONLY": True,

    # Fusion Gallery STEP 文件所在目录
    # USE_CACHE_ONLY=True 时不会使用这个路径
    "STEP_ROOT": Path(r"E:\fusion gallery 360 segmentation\breps\step"),

    # 输出目录
    "OUT_DIR": Path("output/fusion_fag_topology_groups_cached"),

    # FAG 拓扑缓存目录
    # 这里必须是你之前完整跑 MAX_FILES=None 时保存 graph cache 的目录
    "GRAPH_CACHE_DIR": Path("output/fusion_fag_topology_groups_cached/fag_cache"),

    # 查找范围
    # USE_CACHE_ONLY=True:
    #   从 GRAPH_CACHE_DIR 里随机抽取 MAX_FILES 个 graph cache。
    #
    # USE_CACHE_ONLY=False:
    #   从 STEP_ROOT 中按文件名排序后取前 MAX_FILES 个 STEP。
    #
    # 全部设为 None
    "MAX_FILES": 5000,

    # cache_only 模式下随机抽样
    # True:
    #   从全部 graph cache 里随机选 MAX_FILES 个
    #
    # False:
    #   按 cache 文件名顺序取前 MAX_FILES 个
    "RANDOM_SAMPLE_CACHE": True,

    # 随机种子，保证每次随机结果可复现
    # 想每次都不同就设为 None
    "RANDOM_SEED": 42,

    # 只保留 face 数 >= MIN_FACES 的模型
    "MIN_FACES": 20,

    # 一个 topology type 至少包含几个样本才保存
    "MIN_SAMPLES_PER_TOPOLOGY": 2,

    # 是否把 shared edge number 也作为拓扑等价条件
    # False: 只比较 FAG，也就是 face adjacency graph
    # True : 比较 FAG + 每对相邻 face 的 shared_edges 数量
    "USE_SHARED_EDGE_NUMBER": False,

    # 是否强制重新提取 FAG，忽略已有缓存
    # USE_CACHE_ONLY=True 时这个参数无效，因为不会重新提取
    "REBUILD_GRAPH_CACHE": False,

    # 打印更多查找过程信息
    "VERBOSE_PROGRESS": True,

    # 每个 signature bucket 最多允许多少次 exact isomorphism 比较
    # 防止某个大 bucket 看起来卡死
    # 设为 None 表示不限制
    "MAX_EXACT_CHECKS_PER_BUCKET": 20000,
}


# ============================================================
# STEP loading
# ============================================================

def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()
    return shape


# ============================================================
# pythonocc compatibility helpers
# ============================================================

def occ_size(obj) -> int:
    if hasattr(obj, "Size"):
        return obj.Size()

    if hasattr(obj, "Extent"):
        return obj.Extent()

    raise AttributeError(f"{type(obj)} has neither Size() nor Extent()")


def occ_list_to_indices(
    shape_list,
    face_map: TopTools_IndexedMapOfShape,
) -> List[int]:
    indices = []

    it = TopTools_ListIteratorOfListOfShape(shape_list)
    while it.More():
        face = it.Value()
        idx = face_map.FindIndex(face)
        if idx > 0:
            indices.append(idx)
        it.Next()

    return indices


# ============================================================
# FAG cache helpers
# ============================================================

def make_graph_cache_name(step_path: Path) -> str:
    abs_path = str(step_path.resolve())
    h = hashlib.md5(abs_path.encode("utf-8")).hexdigest()
    return f"{step_path.stem}_{h}.json"


def get_graph_cache_path(step_path: Path, cache_dir: Path) -> Path:
    return cache_dir / make_graph_cache_name(step_path)


def graph_to_cache_dict(G: nx.Graph, step_path: Path) -> dict:
    """
    从 STEP 新提取图时使用。
    会记录 STEP 文件的 mtime/size，方便普通 STEP 模式判断 cache 是否过期。
    """
    stat = step_path.stat()

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append(
            {
                "u": int(u),
                "v": int(v),
                "shared_edges": int(data.get("shared_edges", 1)),
            }
        )

    return {
        "step_path": str(step_path),
        "step_mtime_ns": int(stat.st_mtime_ns),
        "step_size": int(stat.st_size),
        "n_faces": int(G.number_of_nodes()),
        "n_adjacencies": int(G.number_of_edges()),
        "nodes": [int(n) for n in G.nodes()],
        "edges": edges,
    }


def cache_dict_to_graph(data: dict) -> nx.Graph:
    G = nx.Graph()

    for n in data["nodes"]:
        G.add_node(int(n))

    for e in data["edges"]:
        G.add_edge(
            int(e["u"]),
            int(e["v"]),
            shared_edges=int(e.get("shared_edges", 1)),
        )

    G.graph["step_path"] = data.get("step_path", "")
    G.graph["n_faces"] = G.number_of_nodes()
    G.graph["n_adjacencies"] = G.number_of_edges()

    return G


def is_cache_valid(cache_data: dict, step_path: Path) -> bool:
    """
    普通 STEP 模式下使用。
    USE_CACHE_ONLY=True 时不会调用这个函数，因为那时不依赖 source STEP。
    """
    if not step_path.exists():
        return False

    stat = step_path.stat()

    return (
        cache_data.get("step_mtime_ns") == int(stat.st_mtime_ns)
        and cache_data.get("step_size") == int(stat.st_size)
    )


def save_graph_cache(G: nx.Graph, step_path: Path, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = get_graph_cache_path(step_path, cache_dir)
    data = graph_to_cache_dict(G, step_path)

    cache_path.write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def load_graph_cache_for_step(
    step_path: Path,
    cache_dir: Path,
    validate_against_step: bool = True,
) -> Optional[nx.Graph]:
    """
    根据 STEP path 查找对应 cache。

    validate_against_step=True:
      检查 STEP 文件 mtime/size。

    validate_against_step=False:
      不碰 STEP 文件，只要 cache json 能读，就返回图。
    """
    cache_path = get_graph_cache_path(step_path, cache_dir)

    if not cache_path.exists():
        return None

    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))

        if validate_against_step:
            if not is_cache_valid(data, step_path):
                return None

        return cache_dict_to_graph(data)

    except Exception:
        return None


def load_graph_cache_file(cache_path: Path) -> Tuple[nx.Graph, dict]:
    """
    直接从一个 cache json 文件读取图。
    不检查 source STEP 文件是否存在。
    这是 USE_CACHE_ONLY=True 的核心。
    """
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    G = cache_dict_to_graph(data)
    return G, data


def find_graph_cache_files(cache_dir: Path) -> List[Path]:
    """
    找到所有已保存的 graph cache json 文件。

    不读取 json 内容，不按照原始 STEP path 排序。
    只按 cache 文件名排序，速度快。
    后续如果 RANDOM_SAMPLE_CACHE=True，会从这些 cache 中随机抽样。
    """
    print(f"[INFO] Scanning graph cache directory: {cache_dir}")

    cache_files = sorted(cache_dir.glob("*.json"))

    print(f"[INFO] Found graph cache json files: {len(cache_files)}")

    return cache_files


# ============================================================
# B-Rep Face Adjacency Graph extraction
# ============================================================

def build_face_adjacency_graph(step_path: Path) -> nx.Graph:
    """
    Build FAG:
      node = B-Rep face
      edge = two faces share at least one B-Rep edge

    shared_edges 会保存。
    之后是否用于拓扑等价判断，由 USE_SHARED_EDGE_NUMBER 决定。
    """
    shape = read_step_shape(step_path)

    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    edge_to_faces = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_to_faces)

    G = nx.Graph()

    n_faces = occ_size(face_map)

    for i in range(1, n_faces + 1):
        G.add_node(i)

    n_topo_edges = occ_size(edge_to_faces)

    for edge_idx in range(1, n_topo_edges + 1):
        faces = edge_to_faces.FindFromIndex(edge_idx)

        face_indices = occ_list_to_indices(faces, face_map)
        face_indices = sorted(set(face_indices))

        if len(face_indices) < 2:
            continue

        for u, v in itertools.combinations(face_indices, 2):
            if G.has_edge(u, v):
                G[u][v]["shared_edges"] += 1
            else:
                G.add_edge(u, v, shared_edges=1)

    G.graph["step_path"] = str(step_path)
    G.graph["n_faces"] = G.number_of_nodes()
    G.graph["n_adjacencies"] = G.number_of_edges()

    return G


def load_or_build_face_adjacency_graph(
    step_path: Path,
    cache_dir: Path,
    rebuild_cache: bool = False,
) -> Tuple[nx.Graph, str]:
    """
    普通 STEP 模式：
      优先读 cache。
      cache 不存在或失效时，重新从 STEP 提取。

    Returns:
      G, source
      source in {"cache", "extracted"}
    """
    if not rebuild_cache:
        G = load_graph_cache_for_step(
            step_path=step_path,
            cache_dir=cache_dir,
            validate_against_step=True,
        )
        if G is not None:
            return G, "cache"

    G = build_face_adjacency_graph(step_path)
    save_graph_cache(G, step_path, cache_dir)

    return G, "extracted"


# ============================================================
# Topology signature and exact isomorphism
# ============================================================

def topology_prefilter_signature(
    G: nx.Graph,
    use_shared_edge_number: bool,
) -> Tuple[Any, ...]:
    """
    Cheap pre-filter before exact isomorphism.

    use_shared_edge_number = False:
      pure FAG:
        - node count
        - adjacency count
        - degree sequence
        - unlabeled WL hash

    use_shared_edge_number = True:
      FAG + shared edge number:
        - node count
        - adjacency count
        - degree sequence
        - sorted shared_edges values
        - edge-attributed WL hash
    """
    degrees = tuple(sorted(dict(G.degree()).values()))

    if use_shared_edge_number:
        shared_edge_values = tuple(
            sorted(
                int(data.get("shared_edges", 1))
                for _, _, data in G.edges(data=True)
            )
        )

        try:
            wl_hash = nx.weisfeiler_lehman_graph_hash(
                G,
                edge_attr="shared_edges",
                iterations=3,
            )
        except Exception:
            wl_hash = None

        return (
            "fag_shared_edges",
            G.number_of_nodes(),
            G.number_of_edges(),
            degrees,
            shared_edge_values,
            wl_hash,
        )

    else:
        try:
            wl_hash = nx.weisfeiler_lehman_graph_hash(
                G,
                iterations=3,
            )
        except Exception:
            wl_hash = None

        return (
            "pure_fag",
            G.number_of_nodes(),
            G.number_of_edges(),
            degrees,
            wl_hash,
        )


def graphs_are_topologically_same(
    G1: nx.Graph,
    G2: nx.Graph,
    use_shared_edge_number: bool,
) -> bool:
    """
    Exact graph isomorphism.

    use_shared_edge_number = False:
      only face adjacency

    use_shared_edge_number = True:
      face adjacency + shared_edges attribute
    """
    if use_shared_edge_number:
        edge_match = nx.algorithms.isomorphism.categorical_edge_match(
            "shared_edges",
            1,
        )
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            G1,
            G2,
            edge_match=edge_match,
        )
    else:
        matcher = nx.algorithms.isomorphism.GraphMatcher(G1, G2)

    return matcher.is_isomorphic()


# ============================================================
# Dataset utilities
# ============================================================

def find_step_files(root: Path) -> List[Path]:
    exts = {".step", ".stp", ".STEP", ".STP"}
    return sorted(p for p in root.rglob("*") if p.suffix in exts)


def maybe_unzip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


# ============================================================
# Graph collection
# ============================================================

def collect_graph_infos_from_cache_only(
    graph_cache_dir: Path,
    max_files: Optional[int],
    min_faces: int,
    use_shared_edge_number: bool,
    random_sample_cache: bool = True,
    random_seed: Optional[int] = 42,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """
    只从 graph cache 读取图。
    不扫描 STEP_ROOT。
    不读取 STEP。
    不检查 source STEP 是否存在。
    """
    cache_files = find_graph_cache_files(graph_cache_dir)

    total_cache_files = len(cache_files)

    if max_files is not None and max_files < total_cache_files:
        if random_sample_cache:
            print(
                f"[INFO] Randomly sampling cache files: "
                f"{max_files} / {total_cache_files} | seed={random_seed}"
            )

            rng = random.Random(random_seed)
            cache_files = rng.sample(cache_files, max_files)

            # 排序只是为了输出稳定，不影响随机抽样结果
            cache_files = sorted(cache_files, key=lambda p: p.name)

        else:
            print(
                f"[INFO] Taking first cache files by filename: "
                f"{max_files} / {total_cache_files}"
            )
            cache_files = cache_files[:max_files]
    else:
        print(f"[INFO] Using all cache files: {total_cache_files}")

    print(f"[INFO] Cache files selected for this run: {len(cache_files)}")

    graph_infos = []
    all_metadata_records = []
    kept_metadata_records = []
    failed = []

    skipped_too_few_faces = 0
    loaded_from_cache = 0

    mode_name = "fag_shared_edges" if use_shared_edge_number else "pure_fag"

    for sample_index, cache_path in enumerate(
        tqdm(cache_files, desc="Loading FAGs from graph cache only"),
        start=1,
    ):
        try:
            G, cache_data = load_graph_cache_file(cache_path)

            loaded_from_cache += 1

            step_path_str = cache_data.get("step_path", G.graph.get("step_path", ""))
            step_path = Path(step_path_str) if step_path_str else Path(cache_path.stem)

            n_faces = G.number_of_nodes()
            n_edges = G.number_of_edges()

            sig = topology_prefilter_signature(
                G,
                use_shared_edge_number=use_shared_edge_number,
            )

            all_rec = {
                "sample_index": sample_index,
                "path": str(step_path),
                "file_name": step_path.name,
                "cache_path": str(cache_path),
                "n_faces": n_faces,
                "n_adjacencies": n_edges,
                "signature": str(sig),
                "graph_source": "cache_only",
                "topology_mode": mode_name,
            }
            all_metadata_records.append(all_rec)

            if n_faces < min_faces:
                skipped_too_few_faces += 1
                continue

            kept_metadata_records.append(all_rec)

            graph_infos.append(
                {
                    "sample_index": sample_index,
                    "path": step_path,
                    "file_name": step_path.name,
                    "cache_path": cache_path,
                    "graph": G,
                    "signature": sig,
                    "n_faces": n_faces,
                    "n_edges": n_edges,
                    "graph_source": "cache_only",
                }
            )

        except Exception as e:
            failed.append(
                {
                    "sample_index": sample_index,
                    "cache_path": str(cache_path),
                    "error": repr(e),
                }
            )

    stats = {
        "num_cache_files_selected": len(cache_files),
        "loaded_from_cache": loaded_from_cache,
        "extracted_now": 0,
        "skipped_too_few_faces": skipped_too_few_faces,
        "failed": failed,
    }

    return graph_infos, all_metadata_records, kept_metadata_records, stats


def collect_graph_infos_from_step_root(
    step_root: Path,
    graph_cache_dir: Path,
    max_files: Optional[int],
    min_faces: int,
    use_shared_edge_number: bool,
    rebuild_graph_cache: bool,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """
    普通模式：
      从 STEP_ROOT 扫描 STEP。
      优先读 cache。
      cache 缺失或失效时，从 STEP 提取。
    """
    step_files = find_step_files(step_root)

    if max_files is not None:
        step_files = step_files[:max_files]

    graph_infos = []
    all_metadata_records = []
    kept_metadata_records = []
    failed = []

    skipped_too_few_faces = 0
    loaded_from_cache = 0
    extracted_now = 0

    mode_name = "fag_shared_edges" if use_shared_edge_number else "pure_fag"

    for sample_index, step_path in enumerate(
        tqdm(step_files, desc="Loading / extracting FAGs from STEP root"),
        start=1,
    ):
        try:
            G, source = load_or_build_face_adjacency_graph(
                step_path=step_path,
                cache_dir=graph_cache_dir,
                rebuild_cache=rebuild_graph_cache,
            )

            if source == "cache":
                loaded_from_cache += 1
            else:
                extracted_now += 1

            n_faces = G.number_of_nodes()
            n_edges = G.number_of_edges()

            sig = topology_prefilter_signature(
                G,
                use_shared_edge_number=use_shared_edge_number,
            )

            all_rec = {
                "sample_index": sample_index,
                "path": str(step_path),
                "file_name": step_path.name,
                "cache_path": str(get_graph_cache_path(step_path, graph_cache_dir)),
                "n_faces": n_faces,
                "n_adjacencies": n_edges,
                "signature": str(sig),
                "graph_source": source,
                "topology_mode": mode_name,
            }
            all_metadata_records.append(all_rec)

            if n_faces < min_faces:
                skipped_too_few_faces += 1
                continue

            kept_metadata_records.append(all_rec)

            graph_infos.append(
                {
                    "sample_index": sample_index,
                    "path": step_path,
                    "file_name": step_path.name,
                    "cache_path": get_graph_cache_path(step_path, graph_cache_dir),
                    "graph": G,
                    "signature": sig,
                    "n_faces": n_faces,
                    "n_edges": n_edges,
                    "graph_source": source,
                }
            )

        except Exception as e:
            failed.append(
                {
                    "sample_index": sample_index,
                    "path": str(step_path),
                    "error": repr(e),
                }
            )

    stats = {
        "num_step_files_selected": len(step_files),
        "loaded_from_cache": loaded_from_cache,
        "extracted_now": extracted_now,
        "skipped_too_few_faces": skipped_too_few_faces,
        "failed": failed,
    }

    return graph_infos, all_metadata_records, kept_metadata_records, stats


# ============================================================
# Grouping logic
# ============================================================

def split_bucket_into_exact_topology_groups(
    bucket_items: List[dict],
    use_shared_edge_number: bool,
    max_exact_checks: Optional[int] = None,
    verbose: bool = True,
    bucket_index: int = 0,
    total_buckets: int = 0,
) -> Tuple[List[List[dict]], int, bool]:
    """
    一个 signature bucket 里可能仍然有非同构图。
    这里用 exact isomorphism 把它继续拆成真正的 topology groups。

    Returns:
      groups
      exact_check_count
      stopped_early
    """
    groups: List[List[dict]] = []
    exact_check_count = 0
    stopped_early = False

    bucket_start = time.perf_counter()

    if verbose:
        print(
            f"\n[BUCKET {bucket_index}/{total_buckets}] "
            f"items={len(bucket_items)} | "
            f"max_exact_checks={max_exact_checks}"
        )

    for item_idx, item in enumerate(bucket_items, start=1):
        placed = False
        G = item["graph"]

        if verbose and (
            item_idx == 1
            or item_idx % 100 == 0
            or item_idx == len(bucket_items)
        ):
            elapsed = time.perf_counter() - bucket_start
            print(
                f"  [bucket {bucket_index}] item {item_idx}/{len(bucket_items)} | "
                f"current_groups={len(groups)} | "
                f"exact_checks={exact_check_count} | "
                f"elapsed={elapsed:.1f}s"
            )

        for group_idx, group in enumerate(groups, start=1):
            if max_exact_checks is not None and exact_check_count >= max_exact_checks:
                stopped_early = True
                if verbose:
                    print(
                        f"  [WARN] bucket {bucket_index} stopped early: "
                        f"exact_checks reached {max_exact_checks}"
                    )
                break

            representative_G = group[0]["graph"]
            exact_check_count += 1

            if graphs_are_topologically_same(
                G,
                representative_G,
                use_shared_edge_number=use_shared_edge_number,
            ):
                group.append(item)
                placed = True
                break

        if stopped_early:
            break

        if not placed:
            groups.append([item])

    elapsed = time.perf_counter() - bucket_start

    if verbose:
        print(
            f"  [DONE BUCKET {bucket_index}] "
            f"groups={len(groups)} | "
            f"exact_checks={exact_check_count} | "
            f"elapsed={elapsed:.1f}s | "
            f"stopped_early={stopped_early}"
        )

    return groups, exact_check_count, stopped_early


def find_topology_groups(
    step_root: Path,
    out_dir: Path,
    graph_cache_dir: Path,
    max_files: Optional[int] = 100,
    min_faces: int = 10,
    min_samples_per_topology: int = 2,
    use_shared_edge_number: bool = False,
    rebuild_graph_cache: bool = False,
    use_cache_only: bool = False,
    verbose_progress: bool = True,
    max_exact_checks_per_bucket: Optional[int] = 20000,
    random_sample_cache: bool = True,
    random_seed: Optional[int] = 42,
):
    """
    在指定范围内寻找所有 topology groups。

    use_cache_only=False:
      从 STEP_ROOT 找 STEP 文件，然后读/建 graph cache。

    use_cache_only=True:
      直接从 GRAPH_CACHE_DIR 读取已有 graph cache。
      不需要 source STEP 文件存在。
    """
    total_start = time.perf_counter()

    out_dir.mkdir(parents=True, exist_ok=True)
    graph_cache_dir.mkdir(parents=True, exist_ok=True)

    mode_name = "fag_shared_edges" if use_shared_edge_number else "pure_fag"
    source_mode_name = "cache_only" if use_cache_only else "step_root"

    print(f"[INFO] Source mode: {source_mode_name}")
    print(f"[INFO] STEP root: {step_root}")
    print(f"[INFO] Graph cache dir: {graph_cache_dir}")
    print(f"[INFO] Output dir: {out_dir}")
    print(f"[INFO] MAX_FILES: {max_files}")
    print(f"[INFO] Random sample cache: {random_sample_cache}")
    print(f"[INFO] Random seed: {random_seed}")
    print(f"[INFO] Minimum faces required: {min_faces}")
    print(f"[INFO] Minimum samples per topology: {min_samples_per_topology}")
    print(f"[INFO] Topology mode: {mode_name}")
    print(f"[INFO] Rebuild graph cache: {rebuild_graph_cache}")
    print(f"[INFO] Verbose progress: {verbose_progress}")
    print(f"[INFO] Max exact checks per bucket: {max_exact_checks_per_bucket}")

    # --------------------------------------------------------
    # Load graph infos
    # --------------------------------------------------------
    load_start = time.perf_counter()

    if use_cache_only:
        graph_infos, all_metadata_records, kept_metadata_records, stats = (
            collect_graph_infos_from_cache_only(
                graph_cache_dir=graph_cache_dir,
                max_files=max_files,
                min_faces=min_faces,
                use_shared_edge_number=use_shared_edge_number,
                random_sample_cache=random_sample_cache,
                random_seed=random_seed,
            )
        )
    else:
        graph_infos, all_metadata_records, kept_metadata_records, stats = (
            collect_graph_infos_from_step_root(
                step_root=step_root,
                graph_cache_dir=graph_cache_dir,
                max_files=max_files,
                min_faces=min_faces,
                use_shared_edge_number=use_shared_edge_number,
                rebuild_graph_cache=rebuild_graph_cache,
            )
        )

    load_elapsed = time.perf_counter() - load_start

    print(f"[INFO] Loaded FAGs from cache: {stats.get('loaded_from_cache', 0)}")
    print(f"[INFO] Extracted FAGs now: {stats.get('extracted_now', 0)}")
    print(f"[INFO] Skipped models with faces < {min_faces}: {stats.get('skipped_too_few_faces', 0)}")
    print(f"[INFO] Kept models with faces >= {min_faces}: {len(graph_infos)}")
    print(f"[TIME] Loading / filtering graphs: {load_elapsed:.1f}s")

    failed = stats.get("failed", [])
    if failed:
        failed_path = out_dir / "failed_files.json"
        failed_path.write_text(
            json.dumps(failed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[WARN] Failed files/caches: {len(failed)}. See {failed_path}")

    if len(graph_infos) == 0:
        print("[WARN] No graphs kept after filtering. No topology groups will be produced.")

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------
    metadata_start = time.perf_counter()

    all_meta_csv = (
        out_dir
        / f"graph_metadata_all_{mode_name}_{source_mode_name}_max_{max_files}_min_faces_{min_faces}.csv"
    )
    pd.DataFrame(all_metadata_records).to_csv(
        all_meta_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[OK] Wrote all metadata: {all_meta_csv}")

    kept_meta_csv = (
        out_dir
        / f"graph_metadata_kept_{mode_name}_{source_mode_name}_max_{max_files}_min_faces_{min_faces}.csv"
    )
    pd.DataFrame(kept_metadata_records).to_csv(
        kept_meta_csv,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"[OK] Wrote kept metadata: {kept_meta_csv}")

    metadata_elapsed = time.perf_counter() - metadata_start
    print(f"[TIME] Writing metadata: {metadata_elapsed:.1f}s")

    # --------------------------------------------------------
    # Coarse bucketing
    # --------------------------------------------------------
    bucket_start = time.perf_counter()

    signature_buckets: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for info in graph_infos:
        signature_buckets[info["signature"]].append(info)

    candidate_buckets = {
        sig: items
        for sig, items in signature_buckets.items()
        if len(items) >= min_samples_per_topology
    }

    bucket_sizes = sorted(
        [len(items) for items in candidate_buckets.values()],
        reverse=True,
    )

    print(f"[INFO] Total signature buckets: {len(signature_buckets)}")
    print(f"[INFO] Candidate signature buckets: {len(candidate_buckets)}")

    if bucket_sizes:
        print(f"[INFO] Largest candidate buckets: {bucket_sizes[:10]}")
    else:
        print("[INFO] No candidate buckets found.")

    bucket_elapsed = time.perf_counter() - bucket_start
    print(f"[TIME] Coarse bucketing: {bucket_elapsed:.1f}s")

    # --------------------------------------------------------
    # Exact isomorphism grouping
    # --------------------------------------------------------
    exact_start = time.perf_counter()

    exact_groups: List[List[dict]] = []
    total_exact_checks = 0
    stopped_buckets = 0

    bucket_items_list = list(candidate_buckets.items())

    for bucket_idx, (sig, items) in enumerate(
        tqdm(
            bucket_items_list,
            desc="Splitting buckets by exact isomorphism",
        ),
        start=1,
    ):
        groups_in_bucket, checks, stopped_early = split_bucket_into_exact_topology_groups(
            bucket_items=items,
            use_shared_edge_number=use_shared_edge_number,
            max_exact_checks=max_exact_checks_per_bucket,
            verbose=verbose_progress,
            bucket_index=bucket_idx,
            total_buckets=len(bucket_items_list),
        )

        total_exact_checks += checks

        if stopped_early:
            stopped_buckets += 1

        for group in groups_in_bucket:
            if len(group) >= min_samples_per_topology:
                exact_groups.append(group)

    exact_elapsed = time.perf_counter() - exact_start

    print(f"[INFO] Total exact isomorphism checks: {total_exact_checks}")
    print(f"[INFO] Buckets stopped early: {stopped_buckets}")
    print(f"[TIME] Exact isomorphism grouping: {exact_elapsed:.1f}s")

    # 排序：样本数多的拓扑类型排前面
    exact_groups.sort(
        key=lambda g: (
            -len(g),
            g[0]["n_faces"],
            g[0]["n_edges"],
            str(g[0]["path"]),
        )
    )

    print(f"[DONE] Topology groups with >= {min_samples_per_topology} samples: {len(exact_groups)}")

    # --------------------------------------------------------
    # Save group outputs
    # --------------------------------------------------------
    output_start = time.perf_counter()

    group_summary_records = []
    group_member_records = []
    group_json = []

    for idx, group in enumerate(exact_groups, start=1):
        topology_id = f"topology_{idx:04d}"

        sample_indices = [int(item["sample_index"]) for item in group]
        file_names = [item["file_name"] for item in group]
        paths = [str(item["path"]) for item in group]
        cache_paths = [str(item.get("cache_path", "")) for item in group]

        representative = group[0]

        group_summary_records.append(
            {
                "topology_id": topology_id,
                "num_samples": len(group),
                "n_faces": representative["n_faces"],
                "n_adjacencies": representative["n_edges"],
                "topology_mode": mode_name,
                "source_mode": source_mode_name,
                "sample_indices": ";".join(map(str, sample_indices)),
                "file_names": ";".join(file_names),
                "paths": ";".join(paths),
                "cache_paths": ";".join(cache_paths),
                "signature": str(representative["signature"]),
            }
        )

        group_json.append(
            {
                "topology_id": topology_id,
                "num_samples": len(group),
                "n_faces": representative["n_faces"],
                "n_adjacencies": representative["n_edges"],
                "topology_mode": mode_name,
                "source_mode": source_mode_name,
                "sample_indices": sample_indices,
                "file_names": file_names,
                "paths": paths,
                "cache_paths": cache_paths,
                "signature": str(representative["signature"]),
            }
        )

        for item in group:
            group_member_records.append(
                {
                    "topology_id": topology_id,
                    "sample_index": int(item["sample_index"]),
                    "file_name": item["file_name"],
                    "path": str(item["path"]),
                    "cache_path": str(item.get("cache_path", "")),
                    "n_faces": item["n_faces"],
                    "n_adjacencies": item["n_edges"],
                    "topology_mode": mode_name,
                    "source_mode": source_mode_name,
                    "graph_source": item["graph_source"],
                    "signature": str(item["signature"]),
                }
            )

    groups_csv = (
        out_dir
        / f"topology_groups_{mode_name}_{source_mode_name}_max_{max_files}_min_faces_{min_faces}.csv"
    )
    members_csv = (
        out_dir
        / f"topology_group_members_{mode_name}_{source_mode_name}_max_{max_files}_min_faces_{min_faces}.csv"
    )
    groups_json_path = (
        out_dir
        / f"topology_groups_{mode_name}_{source_mode_name}_max_{max_files}_min_faces_{min_faces}.json"
    )

    pd.DataFrame(group_summary_records).to_csv(
        groups_csv,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(group_member_records).to_csv(
        members_csv,
        index=False,
        encoding="utf-8-sig",
    )

    groups_json_path.write_text(
        json.dumps(group_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 写 fixed latest 文件名，方便 viewer 固定读取
    latest_groups_csv = out_dir / "topology_groups_latest.csv"
    latest_members_csv = out_dir / "topology_group_members_latest.csv"
    latest_groups_json = out_dir / "topology_groups_latest.json"

    pd.DataFrame(group_summary_records).to_csv(
        latest_groups_csv,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(group_member_records).to_csv(
        latest_members_csv,
        index=False,
        encoding="utf-8-sig",
    )

    latest_groups_json.write_text(
        json.dumps(group_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    output_elapsed = time.perf_counter() - output_start

    print(f"[OK] Wrote topology groups CSV: {groups_csv}")
    print(f"[OK] Wrote topology members CSV: {members_csv}")
    print(f"[OK] Wrote topology groups JSON: {groups_json_path}")
    print(f"[OK] Wrote latest groups CSV: {latest_groups_csv}")
    print(f"[OK] Wrote latest members CSV: {latest_members_csv}")
    print(f"[OK] Wrote latest groups JSON: {latest_groups_json}")
    print(f"[TIME] Writing group outputs: {output_elapsed:.1f}s")

    print("\n[SUMMARY]")
    for rec in group_summary_records:
        print(
            f"{rec['topology_id']}: "
            f"samples={rec['sample_indices']} "
            f"files={rec['file_names']}"
        )

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[TIME] Total elapsed: {total_elapsed:.1f}s")


def main():
    find_topology_groups(
        step_root=CONFIG["STEP_ROOT"],
        out_dir=CONFIG["OUT_DIR"],
        graph_cache_dir=CONFIG["GRAPH_CACHE_DIR"],
        max_files=CONFIG["MAX_FILES"],
        min_faces=CONFIG["MIN_FACES"],
        min_samples_per_topology=CONFIG["MIN_SAMPLES_PER_TOPOLOGY"],
        use_shared_edge_number=CONFIG["USE_SHARED_EDGE_NUMBER"],
        rebuild_graph_cache=CONFIG["REBUILD_GRAPH_CACHE"],
        use_cache_only=CONFIG["USE_CACHE_ONLY"],
        verbose_progress=CONFIG["VERBOSE_PROGRESS"],
        max_exact_checks_per_bucket=CONFIG["MAX_EXACT_CHECKS_PER_BUCKET"],
        random_sample_cache=CONFIG["RANDOM_SAMPLE_CACHE"],
        random_seed=CONFIG["RANDOM_SEED"],
    )


if __name__ == "__main__":
    main()