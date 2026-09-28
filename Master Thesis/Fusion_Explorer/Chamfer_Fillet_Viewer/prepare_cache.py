import json
import math
from pathlib import Path

import numpy as np

from OCC.Core.STEPControl import (
    STEPControl_Reader,
    STEPControl_Writer,
    STEPControl_AsIs,
)
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopTools import (
    TopTools_ListOfShape,
    TopTools_ListIteratorOfListOfShape,
)

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Defeaturing
from OCC.Core.BRepCheck import BRepCheck_Analyzer
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
)


# ============================================================
# 0. 你只需要改这里
# ============================================================

CONFIG = {
    # 这里填你已经 copy 好 40 个样本的目录
    # 下面应该有：
    # CACHE_ROOT/data/step
    # CACHE_ROOT/data/seg
    "CACHE_ROOT": r"E:\fusion_cache_50",

    # mesh 精度
    # 400 快，600 中等，800 更细但慢
    "LINEAR_DEFLECTION_RATIO": 600.0,

    # 是否覆盖已经存在的 cache 和 defeatured step
    "OVERWRITE": True,

    # 是否只处理前 N 个；None 表示全部
    "LIMIT": None,

    # 如果某个模型 defeaturing 失败，是否继续处理下一个
    "CONTINUE_ON_FAIL": True,

    # after 模型中没有原始 label 的新面使用这个 label
    "RECOVERED_PATCH_LABEL": 8,
}


# ============================================================
# 1. feature 颜色和默认 label
# ============================================================

FEATURE_COLORS = {
    "Chamfer": "#39FF14",          # 荧光绿
    "Fillet": "#FFFF33",           # 荧光黄
    "ExtrudeSide": "#1f77b4",
    "ExtrudeEnd": "#ff7f0e",
    "CutSide": "#d62728",
    "CutEnd": "#9467bd",
    "RevolveSide": "#17becf",
    "RevolveEnd": "#e377c2",
    "RecoveredPatch": "#BDBDBD",   # OCC 去特征后新生成/无法映射的面
    "Unknown": "#808080",
}

DEFAULT_SEGMENT_NAMES = [
    "ExtrudeSide",
    "ExtrudeEnd",
    "CutSide",
    "CutEnd",
    "Fillet",
    "Chamfer",
    "RevolveSide",
    "RevolveEnd",
]


# ============================================================
# 2. OpenCascade surface type 映射
# ============================================================

SURFACE_TYPE_TO_NAME = {
    int(GeomAbs_Plane): "Plane",
    int(GeomAbs_Cylinder): "Cylinder",
    int(GeomAbs_Cone): "Cone",
    int(GeomAbs_Sphere): "Sphere",
    int(GeomAbs_Torus): "Torus",
    int(GeomAbs_BezierSurface): "BezierSurface",
    int(GeomAbs_BSplineSurface): "BSplineSurface",
    int(GeomAbs_SurfaceOfRevolution): "SurfaceOfRevolution",
    int(GeomAbs_SurfaceOfExtrusion): "SurfaceOfExtrusion",
    int(GeomAbs_OffsetSurface): "OffsetSurface",
    int(GeomAbs_OtherSurface): "OtherSurface",
}


def get_face_surface_type(face):
    """
    提取一个 TopoDS_Face 对应的 OCC 几何曲面类型。

    返回：
        surface_type_id: int
        surface_type_name: str

    常见结果：
        Plane
        Cylinder
        Cone
        Sphere
        Torus
        BSplineSurface
        BezierSurface
        SurfaceOfRevolution
        SurfaceOfExtrusion
        OffsetSurface
        OtherSurface
    """
    adaptor = BRepAdaptor_Surface(face)
    surface_type = adaptor.GetType()
    surface_type_id = int(surface_type)
    surface_type_name = SURFACE_TYPE_TO_NAME.get(
        surface_type_id,
        f"UnknownSurfaceType_{surface_type_id}",
    )
    return surface_type_id, surface_type_name


# ============================================================
# 3. segment_names
# ============================================================

def normalize_segment_names(obj):
    if isinstance(obj, list):
        index_to_name = {i: str(name) for i, name in enumerate(obj)}
    elif isinstance(obj, dict):
        index_to_name = {int(k): str(v) for k, v in obj.items()}
    else:
        raise ValueError("segment_names.json must be list or dict")

    name_to_index = {v: k for k, v in index_to_name.items()}
    return index_to_name, name_to_index


def load_segment_names(cache_root: Path):
    path = cache_root / "segment_names.json"

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        print("[WARN] segment_names.json not found, use default 8 classes.")
        obj = DEFAULT_SEGMENT_NAMES

    index_to_name, name_to_index = normalize_segment_names(obj)

    recovered_label = int(CONFIG["RECOVERED_PATCH_LABEL"])
    index_to_name[recovered_label] = "RecoveredPatch"
    name_to_index["RecoveredPatch"] = recovered_label

    return index_to_name, name_to_index


# ============================================================
# 4. 文件扫描
# ============================================================

def scan_pairs(cache_root: Path):
    step_dir = cache_root / "data" / "step"
    seg_dir = cache_root / "data" / "seg"

    if not step_dir.exists():
        raise FileNotFoundError(f"Cannot find step dir: {step_dir}")

    if not seg_dir.exists():
        raise FileNotFoundError(f"Cannot find seg dir: {seg_dir}")

    step_files = []
    for ext in ["*.step", "*.stp", "*.STEP", "*.STP"]:
        step_files.extend(step_dir.glob(ext))

    pairs = []

    for step_path in sorted(step_files):
        seg_path = seg_dir / f"{step_path.stem}.seg"
        if seg_path.exists():
            pairs.append((step_path, seg_path))
        else:
            print(f"[WARN] no seg for {step_path.name}")

    limit = CONFIG["LIMIT"]
    if limit is not None:
        pairs = pairs[: int(limit)]

    return pairs


def read_seg(seg_path: Path):
    labels = np.loadtxt(seg_path, dtype=np.int32)
    return np.atleast_1d(labels).astype(np.int32)


# ============================================================
# 5. STEP I/O
# ============================================================

def read_step_shape(step_path: Path):
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP: {step_path}")

    reader.TransferRoots()
    return reader.OneShape()


def write_step_shape(shape, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    status = writer.Write(str(out_path))

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to write STEP: {out_path}")


def iter_faces(shape):
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)

    while exp.More():
        faces.append(exp.Current())
        exp.Next()

    return faces


# ============================================================
# 6. mesh cache 基础函数
# ============================================================

def shape_diagonal(shape):
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()

    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    diag = math.sqrt(dx * dx + dy * dy + dz * dz)

    if not math.isfinite(diag) or diag <= 0:
        return 1.0

    return diag


def triangulate_shape(shape, linear_deflection_ratio: float):
    diag = shape_diagonal(shape)
    linear_deflection = max(diag / linear_deflection_ratio, 1e-4)
    angular_deflection = 0.35

    BRepMesh_IncrementalMesh(
        shape,
        linear_deflection,
        False,
        angular_deflection,
        True,
    )

    return linear_deflection, angular_deflection


def triangulate_face(face):
    loc = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, loc)

    if triangulation is None:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32)

    trsf = loc.Transformation()

    vertices = np.empty((triangulation.NbNodes(), 3), dtype=np.float32)

    for node_idx in range(1, triangulation.NbNodes() + 1):
        p = triangulation.Node(node_idx).Transformed(trsf)
        vertices[node_idx - 1, 0] = p.X()
        vertices[node_idx - 1, 1] = p.Y()
        vertices[node_idx - 1, 2] = p.Z()

    triangles = np.empty((triangulation.NbTriangles(), 3), dtype=np.int32)
    reversed_face = face.Orientation() == TopAbs_REVERSED

    for tri_idx in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(tri_idx)

        try:
            n1, n2, n3 = tri.Get()
        except Exception:
            n1, n2, n3 = tri

        i = int(n1) - 1
        j = int(n2) - 1
        k = int(n3) - 1

        if reversed_face:
            triangles[tri_idx - 1] = [i, k, j]
        else:
            triangles[tri_idx - 1] = [i, j, k]

    return vertices, triangles


def build_npz_mesh_cache(shape, face_labels, out_npz_path: Path, linear_deflection_ratio: float):
    """
    生成二进制 mesh cache：

    vertices:          (N, 3) float32
    triangles:         (M, 3) int32
    triangle_face:     (M,) int32，每个三角形属于哪个 B-Rep face
    face_label:        (F,) int32，每个 B-Rep face 的 feature label
    face_surface_type: (F,) int32，每个 B-Rep face 的 OCC surface type id
    """
    faces = iter_faces(shape)

    if len(faces) != len(face_labels):
        raise RuntimeError(
            f"Face label mismatch: shape faces={len(faces)}, labels={len(face_labels)}"
        )

    linear_deflection, angular_deflection = triangulate_shape(
        shape,
        linear_deflection_ratio,
    )

    all_vertices = []
    all_triangles = []
    all_triangle_face = []

    face_surface_type = np.zeros((len(faces),), dtype=np.int32)

    vertex_offset = 0

    for face_idx, face in enumerate(faces):
        surface_type_id, _ = get_face_surface_type(face)
        face_surface_type[face_idx] = surface_type_id

        v, tri = triangulate_face(face)

        if v.shape[0] == 0 or tri.shape[0] == 0:
            continue

        tri_global = tri + vertex_offset

        all_vertices.append(v)
        all_triangles.append(tri_global)
        all_triangle_face.append(
            np.full((tri_global.shape[0],), face_idx, dtype=np.int32)
        )

        vertex_offset += v.shape[0]

    if all_vertices:
        vertices = np.vstack(all_vertices).astype(np.float32)
        triangles = np.vstack(all_triangles).astype(np.int32)
        triangle_face = np.concatenate(all_triangle_face).astype(np.int32)
    else:
        vertices = np.zeros((0, 3), dtype=np.float32)
        triangles = np.zeros((0, 3), dtype=np.int32)
        triangle_face = np.zeros((0,), dtype=np.int32)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_npz_path,
        vertices=vertices,
        triangles=triangles,
        triangle_face=triangle_face,
        face_label=np.asarray(face_labels, dtype=np.int32),
        face_surface_type=face_surface_type,
    )

    return {
        "num_faces": int(len(faces)),
        "num_vertices": int(vertices.shape[0]),
        "num_triangles": int(triangles.shape[0]),
        "linear_deflection": float(linear_deflection),
        "angular_deflection": float(angular_deflection),
    }


# ============================================================
# 7. OCC defeaturing
# ============================================================

def top_tools_list_to_python_list(toptools_list):
    result = []
    it = TopTools_ListIteratorOfListOfShape(toptools_list)

    while it.More():
        result.append(it.Value())
        it.Next()

    return result


def make_remove_face_list(faces, labels, fillet_idx, chamfer_idx):
    remove_list = TopTools_ListOfShape()
    remove_indices = []

    for idx, (face, label) in enumerate(zip(faces, labels)):
        label = int(label)

        if label in {fillet_idx, chamfer_idx}:
            remove_list.Append(face)
            remove_indices.append(idx)

    return remove_list, remove_indices


def run_defeaturing(shape, faces_to_remove):
    algo = BRepAlgoAPI_Defeaturing()
    algo.SetShape(shape)

    try:
        algo.SetFacesToRemove(faces_to_remove)
    except Exception:
        it = TopTools_ListIteratorOfListOfShape(faces_to_remove)
        while it.More():
            try:
                algo.AddFaceToRemove(it.Value())
            except Exception as e:
                raise RuntimeError(
                    "当前 pythonocc 的 BRepAlgoAPI_Defeaturing 绑定不支持 "
                    "SetFacesToRemove 或 AddFaceToRemove。"
                ) from e
            it.Next()

    algo.Build()

    if not algo.IsDone():
        raise RuntimeError("BRepAlgoAPI_Defeaturing failed: IsDone() is False")

    result = algo.Shape()

    checker = BRepCheck_Analyzer(result)
    valid = bool(checker.IsValid())

    return algo, result, valid


def find_same_face_index(target_face, result_faces):
    for idx, rf in enumerate(result_faces):
        try:
            if rf.IsSame(target_face):
                return idx
        except Exception:
            pass
    return None


def map_after_face_labels(
    algo,
    original_faces,
    original_labels,
    result_faces,
    remove_indices,
    recovered_label,
):
    """
    after 模型 face 的 label 来源：

    1. 如果 result face 和某个未删除 original face 是同一个 shape，继承原 label
    2. 如果 OCC history Modified(original_face) 指向 result face，继承原 label
    3. 其余新生成/无法映射的 face 标记为 RecoveredPatch
    """
    remove_set = set(remove_indices)

    after_labels = np.full(
        (len(result_faces),),
        int(recovered_label),
        dtype=np.int32,
    )

    mapped = set()

    # A. IsSame 直接匹配
    for orig_idx, orig_face in enumerate(original_faces):
        if orig_idx in remove_set:
            continue

        label = int(original_labels[orig_idx])

        res_idx = find_same_face_index(orig_face, result_faces)
        if res_idx is not None:
            after_labels[res_idx] = label
            mapped.add(res_idx)

    # B. OCC history Modified 匹配
    for orig_idx, orig_face in enumerate(original_faces):
        if orig_idx in remove_set:
            continue

        label = int(original_labels[orig_idx])

        try:
            modified_list = algo.Modified(orig_face)
            modified_faces = top_tools_list_to_python_list(modified_list)
        except Exception:
            modified_faces = []

        for mf in modified_faces:
            res_idx = find_same_face_index(mf, result_faces)
            if res_idx is not None:
                after_labels[res_idx] = label
                mapped.add(res_idx)

    return after_labels, sorted(mapped)


# ============================================================
# 8. 统计函数
# ============================================================

def feature_count_from_labels(labels, index_to_name):
    count = {}

    for label in labels:
        name = index_to_name.get(int(label), "Unknown")
        count[name] = count.get(name, 0) + 1

    return count


def surface_type_count_from_shape(shape):
    faces = iter_faces(shape)

    result = {}

    for face in faces:
        _, surface_type_name = get_face_surface_type(face)
        result[surface_type_name] = result.get(surface_type_name, 0) + 1

    return result


def feature_surface_type_count_from_shape(shape, labels, index_to_name):
    """
    统计 feature × surface type：

    Fillet:
        Cylinder: 10
        Torus: 2
        BSplineSurface: 1

    Chamfer:
        Plane: 8
    """
    faces = iter_faces(shape)

    if len(faces) != len(labels):
        raise RuntimeError(
            f"Cannot count surface types: faces={len(faces)}, labels={len(labels)}"
        )

    result = {}

    for face, label in zip(faces, labels):
        feature_name = index_to_name.get(int(label), "Unknown")
        _, surface_type_name = get_face_surface_type(face)

        if feature_name not in result:
            result[feature_name] = {}

        result[feature_name][surface_type_name] = (
            result[feature_name].get(surface_type_name, 0) + 1
        )

    return result


# ============================================================
# 9. 单模型完整处理
# ============================================================

def process_one_model(
    step_path: Path,
    seg_path: Path,
    cache_root: Path,
    index_to_name: dict,
    name_to_index: dict,
):
    stem = step_path.stem

    compare_cache_dir = cache_root / "compare_mesh_cache"
    defeatured_step_dir = cache_root / "data" / "defeatured_step"
    log_dir = compare_cache_dir / "logs"

    compare_cache_dir.mkdir(parents=True, exist_ok=True)
    defeatured_step_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    before_npz = compare_cache_dir / f"{stem}.before.mesh.npz"
    after_npz = compare_cache_dir / f"{stem}.after.mesh.npz"
    out_step = defeatured_step_dir / f"{stem}.defeatured.step"
    log_path = log_dir / f"{stem}.json"

    overwrite = bool(CONFIG["OVERWRITE"])
    linear_deflection_ratio = float(CONFIG["LINEAR_DEFLECTION_RATIO"])
    recovered_label = int(CONFIG["RECOVERED_PATCH_LABEL"])

    if (
        before_npz.exists()
        and after_npz.exists()
        and out_step.exists()
        and log_path.exists()
        and not overwrite
    ):
        print("[SKIP] cache exists")
        return None

    if "Fillet" not in name_to_index:
        raise RuntimeError("Fillet not found in segment names")

    if "Chamfer" not in name_to_index:
        raise RuntimeError("Chamfer not found in segment names")

    fillet_idx = int(name_to_index["Fillet"])
    chamfer_idx = int(name_to_index["Chamfer"])

    labels = read_seg(seg_path)

    shape = read_step_shape(step_path)
    original_faces = iter_faces(shape)

    if len(original_faces) != len(labels):
        raise RuntimeError(
            f"Original face count mismatch: faces={len(original_faces)}, labels={len(labels)}"
        )

    remove_list, remove_indices = make_remove_face_list(
        original_faces,
        labels,
        fillet_idx,
        chamfer_idx,
    )

    if not remove_indices:
        raise RuntimeError("No Fillet/Chamfer faces to remove")

    # before cache：原模型直接用原始 labels
    before_info = build_npz_mesh_cache(
        shape=shape,
        face_labels=labels,
        out_npz_path=before_npz,
        linear_deflection_ratio=linear_deflection_ratio,
    )

    # 统计 before
    before_count = feature_count_from_labels(labels, index_to_name)
    before_surface_type_count = surface_type_count_from_shape(shape)
    before_feature_surface_count = feature_surface_type_count_from_shape(
        shape,
        labels,
        index_to_name,
    )

    # defeature
    algo, result_shape, is_valid = run_defeaturing(shape, remove_list)

    # 写去特征后的 STEP
    write_step_shape(result_shape, out_step)

    result_faces = iter_faces(result_shape)

    after_labels, mapped_result_faces = map_after_face_labels(
        algo=algo,
        original_faces=original_faces,
        original_labels=labels,
        result_faces=result_faces,
        remove_indices=remove_indices,
        recovered_label=recovered_label,
    )

    # after cache：result shape 用 history 映射后的 labels
    after_info = build_npz_mesh_cache(
        shape=result_shape,
        face_labels=after_labels,
        out_npz_path=after_npz,
        linear_deflection_ratio=linear_deflection_ratio,
    )

    # 统计 after
    after_count = feature_count_from_labels(after_labels, index_to_name)
    after_surface_type_count = surface_type_count_from_shape(result_shape)
    after_feature_surface_count = feature_surface_type_count_from_shape(
        result_shape,
        after_labels,
        index_to_name,
    )

    num_recovered = int(np.sum(after_labels == recovered_label))

    log = {
        "stem": stem,
        "step": str(step_path),
        "seg": str(seg_path),
        "defeatured_step": str(out_step),

        "original_num_faces": int(len(original_faces)),
        "result_num_faces": int(len(result_faces)),

        "remove_face_indices": [int(i) for i in remove_indices],
        "num_removed_faces": int(len(remove_indices)),
        "num_mapped_result_faces": int(len(mapped_result_faces)),
        "num_recovered_patch_faces": num_recovered,
        "is_result_valid": bool(is_valid),

        "before_feature_count": before_count,
        "after_feature_count": after_count,

        "before_surface_type_count": before_surface_type_count,
        "after_surface_type_count": after_surface_type_count,

        "before_feature_surface_count": before_feature_surface_count,
        "after_feature_surface_count": after_feature_surface_count,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    item = {
        "stem": stem,
        "status": "success",

        "original_step": str(Path("data") / "step" / step_path.name),
        "seg": str(Path("data") / "seg" / seg_path.name),
        "defeatured_step": str(Path("data") / "defeatured_step" / out_step.name),

        "before_mesh": str(Path("compare_mesh_cache") / before_npz.name),
        "after_mesh": str(Path("compare_mesh_cache") / after_npz.name),
        "log": str(Path("compare_mesh_cache") / "logs" / log_path.name),

        "original_num_faces": int(len(original_faces)),
        "result_num_faces": int(len(result_faces)),

        "num_removed_faces": int(len(remove_indices)),
        "num_recovered_patch_faces": num_recovered,
        "is_result_valid": bool(is_valid),

        "before_feature_count": before_count,
        "after_feature_count": after_count,

        "before_surface_type_count": before_surface_type_count,
        "after_surface_type_count": after_surface_type_count,

        "before_feature_surface_count": before_feature_surface_count,
        "after_feature_surface_count": after_feature_surface_count,

        "before_mesh_info": before_info,
        "after_mesh_info": after_info,
    }

    return item


# ============================================================
# 10. main
# ============================================================

def main():
    cache_root = Path(CONFIG["CACHE_ROOT"]).expanduser().resolve()
    compare_cache_dir = cache_root / "compare_mesh_cache"
    compare_cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[CONFIG]")
    print(f"CACHE_ROOT: {cache_root}")
    print(f"LINEAR_DEFLECTION_RATIO: {CONFIG['LINEAR_DEFLECTION_RATIO']}")
    print(f"OVERWRITE: {CONFIG['OVERWRITE']}")
    print(f"LIMIT: {CONFIG['LIMIT']}")
    print("=" * 80)

    index_to_name, name_to_index = load_segment_names(cache_root)

    print("[INFO] Segment names:")
    for k, v in sorted(index_to_name.items()):
        print(f"  {k}: {v}")

    print("[INFO] Surface type names:")
    for k, v in sorted(SURFACE_TYPE_TO_NAME.items()):
        print(f"  {k}: {v}")

    pairs = scan_pairs(cache_root)
    print(f"[INFO] STEP+SEG pairs: {len(pairs)}")

    items = []
    failures = []

    for idx, (step_path, seg_path) in enumerate(pairs, start=1):
        print(f"\n[{idx}/{len(pairs)}] {step_path.stem}")

        try:
            item = process_one_model(
                step_path=step_path,
                seg_path=seg_path,
                cache_root=cache_root,
                index_to_name=index_to_name,
                name_to_index=name_to_index,
            )

            if item is not None:
                items.append(item)
                print(
                    f"[OK] removed={item['num_removed_faces']} | "
                    f"before_faces={item['original_num_faces']} | "
                    f"after_faces={item['result_num_faces']} | "
                    f"recovered={item['num_recovered_patch_faces']} | "
                    f"valid={item['is_result_valid']}"
                )
                print(f"     before surface types: {item['before_surface_type_count']}")
                print(f"     after  surface types: {item['after_surface_type_count']}")

        except Exception as e:
            print(f"[FAIL] {step_path.name}: {e}")
            failures.append(
                {
                    "stem": step_path.stem,
                    "step": str(step_path),
                    "seg": str(seg_path),
                    "error": str(e),
                }
            )

            if not bool(CONFIG["CONTINUE_ON_FAIL"]):
                raise

    manifest = {
        "segment_names": {str(k): v for k, v in index_to_name.items()},
        "feature_colors": FEATURE_COLORS,
        "surface_type_names": {str(k): v for k, v in SURFACE_TYPE_TO_NAME.items()},
        "recovered_patch_label": int(CONFIG["RECOVERED_PATCH_LABEL"]),
        "items": items,
        "failures": failures,
    }

    manifest_path = compare_cache_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("[DONE]")
    print(f"Success: {len(items)}")
    print(f"Failed : {len(failures)}")
    print(f"Manifest: {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()