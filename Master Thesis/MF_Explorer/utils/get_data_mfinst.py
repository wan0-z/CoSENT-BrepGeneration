import shutil
from pathlib import Path


def clear_folder(folder_path):
    folder_path = Path(folder_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()


def extract_sample_index(sample_name):
    """
    Example:
        20240116_231044_123_result -> 123
    """
    try:
        return int(sample_name.split("_")[-2])
    except Exception:
        return 10**18


def copy_labels_and_steps_only(
    n,
    src_root,
    dst_root,
    clear_dst=True,
    require_rel_label=True,
):
    """
    Copy only labels and steps from original MFInstSeg.

    Source:
        src_root/
        ├── labels/
        └── steps/

    Target:
        dst_root/
        ├── labels/
        ├── steps/
        └── graphs/   # empty, generated later by AAGExtractor
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)

    src_labels_dir = src_root / "labels"
    src_steps_dir = src_root / "steps"

    dst_labels_dir = dst_root / "labels"
    dst_steps_dir = dst_root / "steps"
    dst_graphs_dir = dst_root / "graphs"

    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    dst_steps_dir.mkdir(parents=True, exist_ok=True)
    dst_graphs_dir.mkdir(parents=True, exist_ok=True)

    if clear_dst:
        clear_folder(dst_labels_dir)
        clear_folder(dst_steps_dir)
        clear_folder(dst_graphs_dir)

    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Source labels folder not found: {src_labels_dir}")

    if not src_steps_dir.exists():
        raise FileNotFoundError(f"Source steps folder not found: {src_steps_dir}")

    sample_names = []

    for label_path in src_labels_dir.glob("*.json"):
        name = label_path.stem

        if name.endswith("_rel"):
            continue

        sample_names.append(name)

    sample_names = sorted(sample_names, key=extract_sample_index)

    copied = 0
    skipped = 0

    for name in sample_names:
        if copied >= n:
            break

        src_label = src_labels_dir / f"{name}.json"
        src_rel_label = src_labels_dir / f"{name}_rel.json"
        src_step = src_steps_dir / f"{name}.step"

        missing = []

        if not src_label.exists():
            missing.append(src_label)

        if require_rel_label and not src_rel_label.exists():
            missing.append(src_rel_label)

        if not src_step.exists():
            missing.append(src_step)

        if missing:
            skipped += 1
            print(f"⚠️ Skipped incomplete sample: {name}")
            for p in missing:
                print(f"   missing: {p}")
            continue

        shutil.copy2(src_label, dst_labels_dir / f"{name}.json")

        if require_rel_label:
            shutil.copy2(src_rel_label, dst_labels_dir / f"{name}_rel.json")

        shutil.copy2(src_step, dst_steps_dir / f"{name}.step")

        copied += 1
        print(f"✅ Copied {copied}/{n}: {name}")

    if copied < n:
        raise RuntimeError(
            f"Only copied {copied}/{n} samples. Skipped {skipped}."
        )

    print("\n🎉 Finished copying labels and steps.")
    print(f"Copied samples: {copied}")
    print(f"Graphs folder is still empty: {dst_graphs_dir}")