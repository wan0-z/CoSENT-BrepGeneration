# check_valid_mftrcad_labels.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Set, List


# 0~23 是 machining feature
# 24 plane, 25 cylinder, 26 cone 不是 machining feature
MACHINING_CATEGORY_IDS = set(range(0, 24))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_relation_file(path: Path) -> bool:
    """
    排除 *_rel.json。
    """
    return path.name.endswith("_rel.json")


def get_sample_name(path: Path) -> str:
    """
    样本名默认用文件 stem。
    例如：
    20240116_231044_0_result.json -> 20240116_231044_0_result
    """
    return path.stem


def parse_cls_feature_faces(label_data: Dict[str, Any]) -> Set[int]:
    """
    从 cls 中取出所有 machining feature faces。

    cls:
    {
        "0": 24,
        "1": 23,
        "2": 9,
        ...
    }

    返回：
    {face_id | cls[face_id] in 0~23}
    """
    cls_map = label_data.get("cls", {})

    feature_faces = set()

    for face_id_str, cat_id in cls_map.items():
        face_id = int(face_id_str)
        cat_id = int(cat_id)

        if cat_id in MACHINING_CATEGORY_IDS:
            feature_faces.add(face_id)

    return feature_faces


def parse_seg_faces(label_data: Dict[str, Any]) -> Set[int]:
    """
    从 seg 中取出所有被任意 instance 覆盖到的 face。

    seg:
    [
        [8],
        [],
        [2, 6, 20],
        ...
    ]

    返回：
    {8, 2, 6, 20, ...}
    """
    seg = label_data.get("seg", [])

    seg_faces = set()

    for instance_faces in seg:
        if not instance_faces:
            continue

        for face_id in instance_faces:
            seg_faces.add(int(face_id))

    return seg_faces


def check_one_label(path: Path) -> Dict[str, Any]:
    """
    检查一个 label json。

    valid 标准：
    cls_feature_faces <= seg_faces

    即：
    cls 中所有 machining feature faces 都必须出现在 seg 里。
    """
    label_data = load_json(path)

    cls_feature_faces = parse_cls_feature_faces(label_data)
    seg_faces = parse_seg_faces(label_data)

    missing_in_seg = sorted(cls_feature_faces - seg_faces)
    extra_in_seg = sorted(seg_faces - cls_feature_faces)

    valid = len(missing_in_seg) == 0

    return {
        "sample_name": get_sample_name(path),
        "path": str(path),
        "valid": valid,
        "num_cls_feature_faces": len(cls_feature_faces),
        "num_seg_faces": len(seg_faces),
        "missing_in_seg": missing_in_seg,
        "extra_in_seg": extra_in_seg,
    }


def check_labels_folder(
    labels_dir: str | Path,
    output_txt: str | Path = "valid_label_samples.txt",
    output_invalid_report: str | Path = "invalid_label_report.txt",
):
    labels_dir = Path(labels_dir)
    output_txt = Path(output_txt)
    output_invalid_report = Path(output_invalid_report)

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels folder not found: {labels_dir}")

    label_files = sorted(
        p for p in labels_dir.glob("*.json")
        if not is_relation_file(p)
    )

    valid_samples: List[str] = []
    invalid_results: List[Dict[str, Any]] = []

    total_checked = 0

    for path in label_files:
        try:
            result = check_one_label(path)
            total_checked += 1

            if result["valid"]:
                valid_samples.append(result["sample_name"])
            else:
                invalid_results.append(result)

        except Exception as e:
            print(f"[ERROR] Failed to check {path.name}: {e}")

    # 输出 valid 样本名
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as f:
        for name in valid_samples:
            f.write(name + "\n")

    # 输出 invalid 详情，方便你排查
    with output_invalid_report.open("w", encoding="utf-8") as f:
        for r in invalid_results:
            f.write(f"Sample: {r['sample_name']}\n")
            f.write(f"Path: {r['path']}\n")
            f.write(f"cls feature faces count: {r['num_cls_feature_faces']}\n")
            f.write(f"seg faces count: {r['num_seg_faces']}\n")
            f.write(f"missing_in_seg: {r['missing_in_seg']}\n")
            f.write(f"extra_in_seg: {r['extra_in_seg']}\n")
            f.write("-" * 80 + "\n")

    print("Done.")
    print(f"Labels folder: {labels_dir}")
    print(f"Total checked samples: {total_checked}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Invalid samples: {len(invalid_results)}")
    print(f"Valid sample list saved to: {output_txt.resolve()}")
    print(f"Invalid report saved to: {output_invalid_report.resolve()}")


if __name__ == "__main__":
    # 改成你的 labels 文件夹路径
    check_labels_folder(
        labels_dir="C:\\Users\\go36sal\\Downloads\\archive\\labels",
        output_txt="output/mftrcad/valid_label_samples.txt",
        output_invalid_report="output/mftrcad/invalid_label_report.txt",
    )