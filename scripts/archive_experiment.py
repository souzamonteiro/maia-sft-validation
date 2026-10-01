#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import transformers


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def sha256_file(path: Path) -> str:
    sha = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha.update(chunk)

    return sha.hexdigest()


def git_info() -> dict:
    result = {
        "commit": None,
        "branch": None,
        "dirty": None,
    }

    try:
        result["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()

        result["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()

        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            text=True,
        )

        result["dirty"] = bool(status.strip())

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return result


def environment_info() -> dict:
    mps_built = False
    mps_available = False

    if hasattr(torch.backends, "mps"):
        mps_built = torch.backends.mps.is_built()
        mps_available = torch.backends.mps.is_available()

    return {
        "timestamp": datetime.now().astimezone().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_built": mps_built,
        "mps_available": mps_available,
    }


def next_experiment_number() -> int:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    numbers = []

    for path in EXPERIMENTS_DIR.iterdir():
        if not path.is_dir():
            continue

        parts = path.name.split("-")

        if len(parts) < 2:
            continue

        try:
            number_part = parts[0].split("_")[-1]
            numbers.append(int(number_part))
        except ValueError:
            continue

    return max(numbers, default=0) + 1


def copy_file(
    source: Path,
    destination: Path,
    manifest_files: list,
):
    if not source.exists():
        print(f"WARNING: file not found: {source}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    manifest_files.append(
        {
            "path": str(destination.relative_to(destination.parents[2])),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }
    )


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def create_readme(
    path: Path,
    name: str,
    metrics: dict,
    environment: dict,
    git: dict,
):
    result = metrics.get("result", "UNKNOWN")

    lines = [
        f"# {name}",
        "",
        "## Result",
        "",
        f"**{result}**",
        "",
        "## Metrics",
        "",
    ]

    for key, value in metrics.items():
        lines.append(f"- **{key}:** {value}")

    lines.extend(
        [
            "",
            "## Environment",
            "",
            f"- Python: `{environment['python'].split()[0]}`",
            f"- PyTorch: `{environment['pytorch']}`",
            f"- Transformers: `{environment['transformers']}`",
            f"- Platform: `{environment['platform']}`",
            f"- Machine: `{environment['machine']}`",
            f"- MPS built: `{environment['mps_built']}`",
            f"- MPS available: `{environment['mps_available']}`",
            "",
            "## Git",
            "",
            f"- Commit: `{git['commit']}`",
            f"- Branch: `{git['branch']}`",
            f"- Dirty working tree: `{git['dirty']}`",
            "",
            "## Reproducibility",
            "",
            "All copied artifacts are listed in `manifest.json` with SHA-256 hashes.",
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Archive a completed Maia SFT validation experiment."
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name, e.g. stage0-positive-control",
    )

    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to experiment metrics JSON.",
    )

    parser.add_argument(
        "--config",
        help="Path to experiment configuration JSON.",
    )

    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset file to archive. May be specified multiple times.",
    )

    parser.add_argument(
        "--log",
        help="Training log to archive.",
    )

    args = parser.parse_args()

    metrics_path = Path(args.metrics)

    if not metrics_path.is_absolute():
        metrics_path = PROJECT_ROOT / metrics_path

    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}"
        )

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    experiment_number = next_experiment_number()

    date = datetime.now().strftime("%Y-%m-%d")

    directory_name = (
        f"{date}_{experiment_number:03d}-{args.name}"
    )

    experiment_dir = EXPERIMENTS_DIR / directory_name

    if experiment_dir.exists():
        raise RuntimeError(
            f"Experiment directory already exists: {experiment_dir}"
        )

    experiment_dir.mkdir(parents=True)

    print("=" * 80)
    print("MAIA SFT VALIDATION - EXPERIMENT ARCHIVER")
    print("=" * 80)
    print(f"Experiment: {args.name}")
    print(f"Archive:    {experiment_dir}")
    print()

    manifest_files = []

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    metrics_destination = experiment_dir / "metrics.json"

    copy_file(
        metrics_path,
        metrics_destination,
        manifest_files,
    )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    if args.config:
        config_path = Path(args.config)

        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path

        copy_file(
            config_path,
            experiment_dir / "config.json",
            manifest_files,
        )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    for dataset_string in args.dataset:
        dataset_path = Path(dataset_string)

        if not dataset_path.is_absolute():
            dataset_path = PROJECT_ROOT / dataset_path

        copy_file(
            dataset_path,
            experiment_dir / "data" / dataset_path.name,
            manifest_files,
        )

    # ------------------------------------------------------------------
    # Training log
    # ------------------------------------------------------------------

    if args.log:
        log_path = Path(args.log)

        if not log_path.is_absolute():
            log_path = PROJECT_ROOT / log_path

        copy_file(
            log_path,
            experiment_dir / "logs" / log_path.name,
            manifest_files,
        )

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    environment = environment_info()

    environment_path = experiment_dir / "environment.json"

    write_json(
        environment_path,
        environment,
    )

    manifest_files.append(
        {
            "path": "environment.json",
            "size_bytes": environment_path.stat().st_size,
            "sha256": sha256_file(environment_path),
        }
    )

    # ------------------------------------------------------------------
    # Git metadata
    # ------------------------------------------------------------------

    git = git_info()

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    manifest = {
        "archive_format": 1,
        "experiment": args.name,
        "experiment_number": experiment_number,
        "created_at": datetime.now().astimezone().isoformat(),
        "git": git,
        "files": manifest_files,
    }

    manifest_path = experiment_dir / "manifest.json"

    write_json(
        manifest_path,
        manifest,
    )

    # ------------------------------------------------------------------
    # README
    # ------------------------------------------------------------------

    create_readme(
        experiment_dir / "README.md",
        args.name,
        metrics,
        environment,
        git,
    )

    print()
    print("=" * 80)
    print("EXPERIMENT ARCHIVED")
    print("=" * 80)
    print(f"Directory: {experiment_dir}")
    print(f"Result:    {metrics.get('result', 'UNKNOWN')}")
    print(f"Files:     {len(manifest_files)}")
    print("=" * 80)


if __name__ == "__main__":
    main()