import json
import random
import shutil
from pathlib import Path


# ============================================================
# 0. 你只需要改这里
# ============================================================

CONFIG = {
    # 原始 Fusion 360 Gallery Segmentation 数据集根目录
    # 这个目录下面应该能看到 breps/step, breps/seg, segment_names.json
    "SOURCE_ROOT": r"E:\fusion gallery 360 segmentation",

    # 输出目录
    # 脚本会创建：
    # OUT_ROOT/data/step
    # OUT_ROOT/data/seg
    "OUT_ROOT": r"E:\fusion_cache_50",

    # 复制多少个带 Fillet 或 Chamfer 的模型
    "LIMIT": 40,

    # 是否随机挑选
    # False = 按文件名顺序找，找到 LIMIT 个就停止
    # True  = 先找完所有候选，再随机抽 LIMIT 个
    "SHUFFLE": False,

    # 随机种子，仅当 SHUFFLE=True 时有用
    "SEED": 42,

    # 如果你非常确定 Fillet=4, Chamfer=5，可以设为 True
    # 如果设为 False，会从 segment_names.json 读取 Fillet/Chamfer 的 index
    "USE_HARDCODED_LABELS": False,

    # 仅当 USE_HARDCODED_LABELS=True 时使用
    "FILLET_LABEL": 4,
    "CHAMFER_LABEL": 5,

    # 是否覆盖已经存在的目标文件
    "OVERWRITE": True,
}


# ============================================================
# 1. 查找数据集目录
# ============================================================

def find_step_dir(source_root: Path) -> Path:
    candidates = [
        source_root / "breps" / "step",
        source_root / "breps" / "steps",
        source_root / "step",
        source_root / "steps",
    ]

    for p in candidates:
        if p.exists() and p.is_dir():
            return p

    raise FileNotFoundError(
        f"找不到 STEP 文件夹。请检查 SOURCE_ROOT。\n"
        f"当前 SOURCE_ROOT = {source_root}\n"
        f"期望路径类似：SOURCE_ROOT/breps/step"
    )


def find_seg_dir(source_root: Path) -> Path:
    candidates = [
        source_root / "breps" / "seg",
        source_root / "breps" / "segments",
        source_root / "seg",
        source_root / "segments",
    ]

    for p in candidates:
        if p.exists() and p.is_dir():
            return p

    raise FileNotFoundError(
        f"找不到 SEG 文件夹。请检查 SOURCE_ROOT。\n"
        f"当前 SOURCE_ROOT = {source_root}\n"
        f"期望路径类似：SOURCE_ROOT/breps/seg"
    )


# ============================================================
# 2. 读取 Fillet / Chamfer label index
# ============================================================

def load_label_indices(source_root: Path):
    if CONFIG["USE_HARDCODED_LABELS"]:
        fillet_label = int(CONFIG["FILLET_LABEL"])
        chamfer_label = int(CONFIG["CHAMFER_LABEL"])
        print(f"[INFO] 使用硬编码 label: Fillet={fillet_label}, Chamfer={chamfer_label}")
        return fillet_label, chamfer_label

    segment_names_path = source_root / "segment_names.json"

    if not segment_names_path.exists():
        raise FileNotFoundError(
            f"找不到 segment_names.json: {segment_names_path}\n"
            f"你可以把 CONFIG['USE_HARDCODED_LABELS'] 改成 True，直接使用 Fillet=4, Chamfer=5。"
        )

    with open(segment_names_path, "r", encoding="utf-8") as f:
        segment_names = json.load(f)

    if isinstance(segment_names, list):
        name_to_index = {name: i for i, name in enumerate(segment_names)}
    elif isinstance(segment_names, dict):
        # 兼容 {"0": "ExtrudeSide", ...} 这种格式
        name_to_index = {name: int(idx) for idx, name in segment_names.items()}
    else:
        raise ValueError("segment_names.json 格式不是 list 或 dict。")

    if "Fillet" not in name_to_index:
        raise RuntimeError(f"segment_names.json 中找不到 Fillet。当前内容：{name_to_index}")

    if "Chamfer" not in name_to_index:
        raise RuntimeError(f"segment_names.json 中找不到 Chamfer。当前内容：{name_to_index}")

    fillet_label = name_to_index["Fillet"]
    chamfer_label = name_to_index["Chamfer"]

    print(f"[INFO] 从 segment_names.json 读取 label: Fillet={fillet_label}, Chamfer={chamfer_label}")

    return fillet_label, chamfer_label


# ============================================================
# 3. 快速判断 .seg 是否包含 Fillet 或 Chamfer
# ============================================================

def seg_contains_any_label(seg_path: Path, target_labels: set[int]) -> bool:
    """
    .seg 是纯文本，每行一个整数。
    这里不使用 numpy.loadtxt，而是直接读文本 split，速度更快，开销更小。

    注意不能用简单的字符串包含：
        "4" in text
    因为如果未来出现 14、40 之类会误判。
    所以这里按 token 切开后比较整数。
    """
    try:
        text = seg_path.read_text(encoding="utf-8", errors="ignore")
        for token in text.split():
            try:
                if int(token) in target_labels:
                    return True
            except ValueError:
                continue
        return False
    except Exception as e:
        print(f"[WARN] 读取 SEG 失败: {seg_path.name} | {e}")
        return False


# ============================================================
# 4. 找同名 STEP
# ============================================================

def find_matching_step(step_dir: Path, stem: str) -> Path | None:
    """
    根据 seg 文件 stem 找同名 STEP。
    例如：
        xxx.seg -> xxx.step 或 xxx.stp
    """
    candidates = [
        step_dir / f"{stem}.step",
        step_dir / f"{stem}.stp",
        step_dir / f"{stem}.STEP",
        step_dir / f"{stem}.STP",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


# ============================================================
# 5. 复制文件
# ============================================================

def copy_pair(step_path: Path, seg_path: Path, out_step_dir: Path, out_seg_dir: Path, overwrite: bool):
    out_step_path = out_step_dir / step_path.name
    out_seg_path = out_seg_dir / seg_path.name

    if not overwrite:
        if out_step_path.exists() and out_seg_path.exists():
            return False

    shutil.copy2(step_path, out_step_path)
    shutil.copy2(seg_path, out_seg_path)

    return True


# ============================================================
# 6. 主流程
# ============================================================

def main():
    source_root = Path(CONFIG["SOURCE_ROOT"]).expanduser().resolve()
    out_root = Path(CONFIG["OUT_ROOT"]).expanduser().resolve()

    limit = int(CONFIG["LIMIT"])
    shuffle = bool(CONFIG["SHUFFLE"])
    seed = int(CONFIG["SEED"])
    overwrite = bool(CONFIG["OVERWRITE"])

    step_dir = find_step_dir(source_root)
    seg_dir = find_seg_dir(source_root)

    out_step_dir = out_root / "data" / "step"
    out_seg_dir = out_root / "data" / "seg"

    out_step_dir.mkdir(parents=True, exist_ok=True)
    out_seg_dir.mkdir(parents=True, exist_ok=True)

    fillet_label, chamfer_label = load_label_indices(source_root)
    target_labels = {fillet_label, chamfer_label}

    print("=" * 80)
    print("[CONFIG]")
    print(f"SOURCE_ROOT = {source_root}")
    print(f"STEP_DIR    = {step_dir}")
    print(f"SEG_DIR     = {seg_dir}")
    print(f"OUT_ROOT    = {out_root}")
    print(f"LIMIT       = {limit}")
    print(f"SHUFFLE     = {shuffle}")
    print(f"TARGETS     = {target_labels}")
    print("=" * 80)

    seg_files = sorted(seg_dir.glob("*.seg"))

    print(f"[INFO] SEG files found: {len(seg_files)}")

    selected = []

    if shuffle:
        # 随机模式：必须先找完所有候选，再随机抽样
        print("[INFO] SHUFFLE=True，所以会先扫描所有 .seg 候选。")

        candidates = []
        for idx, seg_path in enumerate(seg_files, start=1):
            if idx % 1000 == 0:
                print(f"[SCAN] {idx}/{len(seg_files)}")

            if not seg_contains_any_label(seg_path, target_labels):
                continue

            step_path = find_matching_step(step_dir, seg_path.stem)
            if step_path is None:
                print(f"[WARN] 找到 SEG 但没有同名 STEP: {seg_path.name}")
                continue

            candidates.append((step_path, seg_path))

        print(f"[INFO] candidates with Fillet/Chamfer: {len(candidates)}")

        random.seed(seed)
        random.shuffle(candidates)
        selected = candidates[:limit]

    else:
        # 非随机模式：找到 limit 个就停止，最快
        print("[INFO] SHUFFLE=False，找到 LIMIT 个后立即停止。")

        for idx, seg_path in enumerate(seg_files, start=1):
            if idx % 1000 == 0:
                print(f"[SCAN] {idx}/{len(seg_files)} | selected={len(selected)}")

            if not seg_contains_any_label(seg_path, target_labels):
                continue

            step_path = find_matching_step(step_dir, seg_path.stem)
            if step_path is None:
                print(f"[WARN] 找到 SEG 但没有同名 STEP: {seg_path.name}")
                continue

            selected.append((step_path, seg_path))

            print(f"[FOUND] {len(selected)}/{limit}: {seg_path.stem}")

            if len(selected) >= limit:
                break

    print(f"[INFO] selected pairs: {len(selected)}")

    manifest = []
    copied_count = 0
    skipped_count = 0

    for idx, (step_path, seg_path) in enumerate(selected, start=1):
        copied = copy_pair(
            step_path=step_path,
            seg_path=seg_path,
            out_step_dir=out_step_dir,
            out_seg_dir=out_seg_dir,
            overwrite=overwrite,
        )

        if copied:
            copied_count += 1
            status = "COPIED"
        else:
            skipped_count += 1
            status = "SKIPPED"

        print(f"[{status}] {idx}/{len(selected)} {step_path.name} + {seg_path.name}")

        manifest.append(
            {
                "stem": seg_path.stem,
                "step": str(Path("data") / "step" / step_path.name),
                "seg": str(Path("data") / "seg" / seg_path.name),
            }
        )

    manifest_path = out_root / "data" / "copy_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("[DONE]")
    print(f"Copied pairs : {copied_count}")
    print(f"Skipped pairs: {skipped_count}")
    print(f"Output STEP  : {out_step_dir}")
    print(f"Output SEG   : {out_seg_dir}")
    print(f"Manifest     : {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()