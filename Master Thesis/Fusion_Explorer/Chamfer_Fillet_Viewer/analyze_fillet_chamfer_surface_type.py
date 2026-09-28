import json
from pathlib import Path

import numpy as np


# ============================================================
# 0. 你只需要改这里
# ============================================================

CONFIG = {
    # 这里填你的缓存根目录
    # 下面应该有 compare_mesh_cache/manifest.json
    "CACHE_ROOT": r"E:\fusion_cache_50",

    # 是否把每个模型的统计也打印出来
    "PRINT_PER_MODEL": True,

    # 输出 JSON 统计文件
    "OUT_JSON": r"E:\fusion_cache_50\compare_mesh_cache\original_fillet_chamfer_surface_type_stats.json",
}


# ============================================================
# 1. 读取 manifest
# ============================================================

def load_manifest(cache_root: Path):
    manifest_path = cache_root / "compare_mesh_cache" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cannot find manifest.json: {manifest_path}\n"
            f"请先运行 prepare_defeature_compare_cache.py"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_map(manifest):
    segment_names = manifest.get("segment_names", {})

    if isinstance(segment_names, list):
        return {i: str(v) for i, v in enumerate(segment_names)}

    if isinstance(segment_names, dict):
        return {int(k): str(v) for k, v in segment_names.items()}

    return {
        0: "ExtrudeSide",
        1: "ExtrudeEnd",
        2: "CutSide",
        3: "CutEnd",
        4: "Fillet",
        5: "Chamfer",
        6: "RevolveSide",
        7: "RevolveEnd",
        8: "RecoveredPatch",
    }


def build_name_to_label(label_map):
    return {name: idx for idx, name in label_map.items()}


def build_surface_type_map(manifest):
    surface_type_names = manifest.get("surface_type_names", {})

    if isinstance(surface_type_names, dict):
        surface_type_map = {int(k): str(v) for k, v in surface_type_names.items()}
        surface_type_map[-1] = "UnknownSurface"
        return surface_type_map

    return {-1: "UnknownSurface"}


# ============================================================
# 2. 读取 before mesh cache
# ============================================================

def load_before_npz(cache_root: Path, before_mesh_rel_path: str):
    npz_path = cache_root / before_mesh_rel_path

    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find before mesh cache: {npz_path}")

    data = np.load(npz_path)

    face_label = data["face_label"]

    if "face_surface_type" not in data:
        raise RuntimeError(
            f"{npz_path} 里面没有 face_surface_type。\n"
            f"说明你还没有用最新版 prepare_defeature_compare_cache.py 重新生成 cache。"
        )

    face_surface_type = data["face_surface_type"]

    if len(face_label) != len(face_surface_type):
        raise RuntimeError(
            f"face_label 和 face_surface_type 长度不一致: {npz_path}\n"
            f"face_label={len(face_label)}, face_surface_type={len(face_surface_type)}"
        )

    return face_label.astype(np.int32), face_surface_type.astype(np.int32)


# ============================================================
# 3. 统计函数
# ============================================================

def add_count(counter: dict, key: str, value: int = 1):
    counter[key] = counter.get(key, 0) + value


def count_feature_surface_types(
    face_label,
    face_surface_type,
    fillet_label: int,
    chamfer_label: int,
    surface_type_map: dict,
):
    result = {
        "Fillet": {},
        "Chamfer": {},
    }

    for label, surface_type_id in zip(face_label, face_surface_type):
        label = int(label)
        surface_type_id = int(surface_type_id)
        surface_type_name = surface_type_map.get(
            surface_type_id,
            f"UnknownSurfaceType_{surface_type_id}",
        )

        if label == fillet_label:
            add_count(result["Fillet"], surface_type_name)

        elif label == chamfer_label:
            add_count(result["Chamfer"], surface_type_name)

    return result


def merge_nested_counts(total: dict, item: dict):
    for feature_name, sub in item.items():
        if feature_name not in total:
            total[feature_name] = {}

        for surface_name, count in sub.items():
            total[feature_name][surface_name] = (
                total[feature_name].get(surface_name, 0) + int(count)
            )


def nested_total_count(d: dict):
    return sum(int(v) for v in d.values())


# ============================================================
# 4. 主流程
# ============================================================

def main():
    cache_root = Path(CONFIG["CACHE_ROOT"]).expanduser().resolve()

    manifest = load_manifest(cache_root)

    label_map = build_label_map(manifest)
    name_to_label = build_name_to_label(label_map)
    surface_type_map = build_surface_type_map(manifest)

    if "Fillet" not in name_to_label:
        raise RuntimeError(f"segment_names 里面找不到 Fillet: {label_map}")

    if "Chamfer" not in name_to_label:
        raise RuntimeError(f"segment_names 里面找不到 Chamfer: {label_map}")

    fillet_label = int(name_to_label["Fillet"])
    chamfer_label = int(name_to_label["Chamfer"])

    items = manifest.get("items", [])

    if not items:
        raise RuntimeError("manifest 里面没有成功处理的模型 items。")

    global_counts = {
        "Fillet": {},
        "Chamfer": {},
    }

    per_model = []

    print("=" * 80)
    print("[INFO] Analyze original / before mesh cache")
    print(f"CACHE_ROOT: {cache_root}")
    print(f"Fillet label : {fillet_label}")
    print(f"Chamfer label: {chamfer_label}")
    print(f"Models       : {len(items)}")
    print("=" * 80)

    for idx, item in enumerate(items, start=1):
        stem = item["stem"]
        before_mesh = item["before_mesh"]

        face_label, face_surface_type = load_before_npz(
            cache_root,
            before_mesh,
        )

        model_counts = count_feature_surface_types(
            face_label=face_label,
            face_surface_type=face_surface_type,
            fillet_label=fillet_label,
            chamfer_label=chamfer_label,
            surface_type_map=surface_type_map,
        )

        merge_nested_counts(global_counts, model_counts)

        model_summary = {
            "stem": stem,
            "before_mesh": before_mesh,
            "Fillet": model_counts["Fillet"],
            "Chamfer": model_counts["Chamfer"],
            "num_fillet_faces": nested_total_count(model_counts["Fillet"]),
            "num_chamfer_faces": nested_total_count(model_counts["Chamfer"]),
        }

        per_model.append(model_summary)

        if CONFIG["PRINT_PER_MODEL"]:
            print(f"\n[{idx}/{len(items)}] {stem}")
            print(f"  Fillet : {model_counts['Fillet']}")
            print(f"  Chamfer: {model_counts['Chamfer']}")

    payload = {
        "cache_root": str(cache_root),
        "fillet_label": fillet_label,
        "chamfer_label": chamfer_label,
        "global_counts": global_counts,
        "global_num_fillet_faces": nested_total_count(global_counts["Fillet"]),
        "global_num_chamfer_faces": nested_total_count(global_counts["Chamfer"]),
        "per_model": per_model,
    }

    out_json = Path(CONFIG["OUT_JSON"]).expanduser().resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 80)
    print("[GLOBAL RESULT]")
    print(f"Fillet total faces : {payload['global_num_fillet_faces']}")
    print(f"Chamfer total faces: {payload['global_num_chamfer_faces']}")
    print()
    print("Fillet surface types:")
    for surface_name, count in sorted(global_counts["Fillet"].items()):
        print(f"  {surface_name}: {count}")
    print()
    print("Chamfer surface types:")
    for surface_name, count in sorted(global_counts["Chamfer"].items()):
        print(f"  {surface_name}: {count}")

    print("\n" + "=" * 80)
    print(f"[SAVED] {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    main()