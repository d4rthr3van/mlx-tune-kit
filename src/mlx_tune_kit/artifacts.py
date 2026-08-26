"""Reproducible run metadata and checkpoint manifests."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from mlx_tune_kit.config import AppConfig
from mlx_tune_kit.data import DatasetSummary

__all__ = ["initialize_run", "load_run_config", "save_metrics", "write_manifest"]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _package_versions() -> dict[str, str]:
    result = {}
    for package in ("mlx-tune-kit", "datasets", "mlx", "mlx-vlm", "torch", "transformers"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            continue
    return result


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def initialize_run(config: AppConfig, summary: DatasetSummary) -> Path:
    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "resolved-config.json", config.as_dict())
    _write_json(run_dir / "dataset-summary.json", asdict(summary))
    _write_json(
        run_dir / "environment.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(),
            "git_commit": _git_commit(),
        },
    )
    write_manifest(run_dir, status="initialized", checkpoints=[])
    return run_dir


def write_manifest(run_dir: Path, *, status: str, checkpoints: list[str], **extra: Any) -> None:
    _write_json(
        run_dir / "manifest.json",
        {"status": status, "checkpoints": checkpoints, **extra},
    )


def save_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    _write_json(run_dir / "metrics.json", metrics)


def load_run_config(run_dir: str | Path) -> AppConfig:
    """Restore a run's resolved config without requiring its original TOML file."""
    from mlx_tune_kit.config import (
        DatasetConfig,
        EvaluationConfig,
        FormatConfig,
        ModelConfig,
        OutputConfig,
        TrainingConfig,
    )

    path = Path(run_dir) / "resolved-config.json"
    if not path.exists():
        raise FileNotFoundError(f"Run configuration not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return AppConfig(
        model=ModelConfig(**raw["model"]),
        dataset=DatasetConfig(**raw["dataset"]),
        format=FormatConfig(**raw["format"]),
        training=TrainingConfig(**raw["training"]),
        output=OutputConfig(**raw["output"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )
