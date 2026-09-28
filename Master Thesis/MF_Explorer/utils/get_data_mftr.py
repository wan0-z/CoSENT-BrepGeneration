import os
import shutil


def clear_dataset_folders(dst_root):
    """
    Clear all files in the 'graphs', 'labels', 'steps' subfolders under the target path.

    Parameters:
        dst_root (str): Target data root directory.
    """
    subfolders = ['graphs', 'labels', 'steps']

    for folder in subfolders:
        folder_path = os.path.join(dst_root, folder)
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"✅ Cleared: {folder_path}")
        else:
            print(f"⚠️ Subfolder does not exist: {folder_path}")


def load_valid_sample_names(valid_txt_path):
    """
    Load valid sample names from valid_label_samples.txt.

    Each line should be like:
        20240116_231044_0_result

    Also supports:
        20240116_231044_0_result.json

    Returns:
        list[str]
    """
    if not os.path.exists(valid_txt_path):
        raise FileNotFoundError(f"valid_label_samples.txt not found: {valid_txt_path}")

    valid_names = []

    with open(valid_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()

            if not name:
                continue

            if name.endswith(".json"):
                name = name[:-5]

            valid_names.append(name)

    # remove duplicates while keeping order
    seen = set()
    unique_valid_names = []
    for name in valid_names:
        if name not in seen:
            unique_valid_names.append(name)
            seen.add(name)

    return unique_valid_names


def extract_sample_index(sample_name):
    """
    Extract index from sample name.

    Example:
        20240116_231044_123_result -> 123

    If extraction fails, return a very large number so it goes to the end.
    """
    try:
        return int(sample_name.split("_")[-2])
    except Exception:
        return 10**18


def copy_dataset_from_valid_list(n, src_root, dst_root, valid_txt_path):
    """
    Copy n complete samples directly according to valid_label_samples.txt.

    For each sample, copy:
        graphs/{sample}.json
        labels/{sample}.json
        labels/{sample}_rel.json
        steps/{sample}.step

    Parameters:
        n (int): Number of complete valid samples to copy.
        src_root (str): Source data root directory.
        dst_root (str): Target data root directory.
        valid_txt_path (str): Path to valid_label_samples.txt.
    """
    subfolders = ['graphs', 'labels', 'steps']
    for folder in subfolders:
        os.makedirs(os.path.join(dst_root, folder), exist_ok=True)

    valid_names = load_valid_sample_names(valid_txt_path)
    valid_names = sorted(valid_names, key=extract_sample_index)

    print(f"✅ Loaded {len(valid_names)} valid sample names from: {valid_txt_path}")

    if len(valid_names) == 0:
        raise ValueError("valid_label_samples.txt is empty. No samples can be copied.")

    copied_count = 0
    skipped_count = 0

    for prefix in valid_names:
        if copied_count >= n:
            break

        graph_file = f"{prefix}.json"
        label_file_1 = f"{prefix}.json"
        label_file_2 = f"{prefix}_rel.json"
        step_file = f"{prefix}.step"

        src_graph = os.path.join(src_root, 'graphs', graph_file)
        src_label1 = os.path.join(src_root, 'labels', label_file_1)
        src_label2 = os.path.join(src_root, 'labels', label_file_2)
        src_step = os.path.join(src_root, 'steps', step_file)

        dst_graph = os.path.join(dst_root, 'graphs', graph_file)
        dst_label1 = os.path.join(dst_root, 'labels', label_file_1)
        dst_label2 = os.path.join(dst_root, 'labels', label_file_2)
        dst_step = os.path.join(dst_root, 'steps', step_file)

        source_files = [src_graph, src_label1, src_label2, src_step]
        target_files = [dst_graph, dst_label1, dst_label2, dst_step]

        source_complete = all(os.path.exists(p) for p in source_files)
        target_exists = any(os.path.exists(p) for p in target_files)

        if not source_complete:
            missing = [p for p in source_files if not os.path.exists(p)]
            skipped_count += 1
            print(f"⚠️ Source incomplete, skipped: {prefix}")
            for p in missing:
                print(f"   missing: {p}")
            continue

        if target_exists:
            skipped_count += 1
            print(f"⚠️ Target already has files, skipped: {prefix}")
            continue

        shutil.copy2(src_graph, dst_graph)
        shutil.copy2(src_label1, dst_label1)
        shutil.copy2(src_label2, dst_label2)
        shutil.copy2(src_step, dst_step)

        copied_count += 1
        print(f"✅ Copied valid sample {copied_count}/{n}: {prefix}")

    if copied_count < n:
        raise RuntimeError(
            f"Only copied {copied_count}/{n} valid complete samples. "
            f"Skipped {skipped_count} samples. "
            f"Please check whether valid_label_samples.txt contains enough complete samples."
        )

    print(f"\n🎉 Successfully copied {n} complete valid samples.")
    print(f"Skipped samples: {skipped_count}")


def check_dataset_integrity(dst_root, n):
    """
    Check if target directory contains expected number of files.
    """
    graphs_dir = os.path.join(dst_root, 'graphs')
    labels_dir = os.path.join(dst_root, 'labels')
    steps_dir = os.path.join(dst_root, 'steps')

    graph_count = len([f for f in os.listdir(graphs_dir) if f.endswith('.json')])
    label_count = len([f for f in os.listdir(labels_dir) if f.endswith('.json')])
    step_count = len([f for f in os.listdir(steps_dir) if f.endswith('.step')])

    print("📊 Check results:")
    print(f"graphs: {graph_count} / expected {n}")
    print(f"labels: {int(label_count / 2)} / expected {n}")
    print(f"steps:  {step_count} / expected {n}")

    if graph_count == n and label_count == 2 * n and step_count == n:
        print("✅ Files complete!")
    else:
        print("❌ Files incomplete, please check path or source files.")


def find_missing_indices(dst_root, n, prefix='20240116_231044'):
    """
    Check for missing index numbers in graphs, labels, steps.

    Parameters:
        dst_root (str): Target data root directory.
        n (int): Expected number of data items.
        prefix (str): Number prefix, e.g. '20240116_231044'.
    """
    missing_graphs = []
    missing_labels = []
    missing_steps = []

    for i in range(n):
        base = f"{prefix}_{i}_result"

        graph_file = f"{base}.json"
        graph_path = os.path.join(dst_root, 'graphs', graph_file)
        if not os.path.exists(graph_path):
            missing_graphs.append(i)

        label_file_1 = f"{base}.json"
        label_file_2 = f"{base}_rel.json"
        label_path_1 = os.path.join(dst_root, 'labels', label_file_1)
        label_path_2 = os.path.join(dst_root, 'labels', label_file_2)
        if not (os.path.exists(label_path_1) and os.path.exists(label_path_2)):
            missing_labels.append(i)

        step_file = f"{base}.step"
        step_path = os.path.join(dst_root, 'steps', step_file)
        if not os.path.exists(step_path):
            missing_steps.append(i)

    print("🧾 Missing file indices:")
    print(f"graphs missing indices: {missing_graphs}")
    print(f"labels missing indices: {missing_labels}")
    print(f"steps missing indices:  {missing_steps}")

    return missing_graphs, missing_labels, missing_steps