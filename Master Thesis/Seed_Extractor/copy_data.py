# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil

from pathlib import Path
from typing import Iterable


SCRIPT_DIRECTORY = Path(__file__).resolve().parent

# ======================== User configuration ========================
# This directory must contain the "steps" and "labels" subdirectories.
DATASET_ROOT = (
    SCRIPT_DIRECTORY.parent
    / "MF_Explorer"
    / "data"
    / "mfinstseg"
)

# Change this value when running the script without command-line arguments.
SAMPLE_NAME = "20221123_142528_0"
# ====================================================================

SINGLE_DATA_ROOT = SCRIPT_DIRECTORY / "data" / "single"
STEP_EXTENSIONS = (".step", ".stp")


def find_step_file(step_directory: Path, sample_name: str) -> Path:
    direct_candidates = [
        step_directory / f"{sample_name}{extension}"
        for extension in STEP_EXTENSIONS
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    matches: Iterable[Path] = step_directory.glob(f"{sample_name}.*")
    valid_matches = [
        path
        for path in matches
        if path.is_file() and path.suffix.lower() in STEP_EXTENSIONS
    ]
    if len(valid_matches) == 1:
        return valid_matches[0]
    if not valid_matches:
        raise FileNotFoundError(
            f"No STEP file found for sample {sample_name} in {step_directory}"
        )
    raise RuntimeError(
        f"Multiple STEP files found for sample {sample_name}: {valid_matches}"
    )


def copy_sample(
    dataset_root: Path,
    sample_name: str,
    destination_root: Path = SINGLE_DATA_ROOT,
) -> Path:
    dataset_root = dataset_root.resolve()
    step_path = find_step_file(dataset_root / "steps", sample_name)
    label_path = dataset_root / "labels" / f"{sample_name}.json"

    if not label_path.exists():
        raise FileNotFoundError(
            f"No label JSON found for sample {sample_name}: {label_path}"
        )

    destination_directory = destination_root.resolve() / sample_name
    destination_step = destination_directory / f"{sample_name}.step"
    destination_label = destination_directory / f"{sample_name}.json"

    if destination_directory.exists():
        raise FileExistsError(
            "The single-sample folder already exists and was not changed: "
            f"{destination_directory}"
        )

    destination_directory.mkdir(parents=True, exist_ok=False)

    try:
        shutil.copy2(step_path, destination_step)
        shutil.copy2(label_path, destination_label)
    except Exception:
        # Only remove the newly-created empty/partial folder from this attempt.
        shutil.rmtree(destination_directory)
        raise

    return destination_directory


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one STEP/label pair into data/single/<sample> without "
            "overwriting existing samples."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--sample", default=SAMPLE_NAME)
    parser.add_argument("--destination-root", type=Path, default=SINGLE_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    destination = copy_sample(
        dataset_root=arguments.data_root,
        sample_name=str(arguments.sample),
        destination_root=arguments.destination_root,
    )
    print(f"Sample copied to: {destination}")


if __name__ == "__main__":
    main()
