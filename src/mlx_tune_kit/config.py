"""Typed, strict TOML configuration."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from mlx_tune_kit.errors import ConfigError

__all__ = [
    "AppConfig",
    "DatasetConfig",
    "EvaluationConfig",
    "FormatConfig",
    "ModelConfig",
    "OutputConfig",
    "TrainingConfig",
    "load_config",
]


@dataclass(frozen=True)
class ModelConfig:
    id: str
    revision: str | None = None
    trust_remote_code: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    source: Literal["hub", "file"]
    id: str | None = None
    path: str | None = None
    name: str | None = None
    revision: str | None = None
    cache_dir: str = ".cache/mlx-tune-kit/datasets"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    overlap_policy: Literal["error", "drop"] = "error"
    files: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FormatConfig:
    kind: Literal["messages", "instruction", "classification"]
    messages_column: str = "messages"
    instruction_column: str = "instruction"
    input_column: str | None = "input"
    response_column: str = "response"
    text_column: str = "text"
    label_column: str = "label"
    system_prompt: str | None = None
    label_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    max_seq_length: int = 1024
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    steps_per_eval: int = 100
    steps_per_save: int = 200
    val_batches: int = 25
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class OutputConfig:
    dir: str = "outputs"
    run_name: str = "fine-tune"
    keep_best: bool = True
    keep_last: bool = True


@dataclass(frozen=True)
class EvaluationConfig:
    split: Literal["validation", "test"] = "test"
    max_samples: int | None = None
    compare_base: bool = False
    max_tokens: int = 64


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    dataset: DatasetConfig
    format: FormatConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.output.dir) / self.output.run_name

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _section(cls: Any, raw: Any, name: str, *, required: bool) -> Any:
    if raw is None:
        if required:
            raise ConfigError(f"Missing required [{name}] section")
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] must be a TOML table")
    allowed = {item.name for item in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown keys in [{name}]: {', '.join(unknown)}")
    try:
        return cls(**raw)
    except TypeError as exc:
        raise ConfigError(f"Invalid [{name}] section: {exc}") from exc


def _validate(config: AppConfig) -> None:
    dataset = config.dataset
    if not config.model.id.strip():
        raise ConfigError("[model].id cannot be empty")
    if dataset.source not in {"hub", "file"}:
        raise ConfigError("[dataset].source must be 'hub' or 'file'")
    if dataset.source == "hub" and not dataset.id:
        raise ConfigError("[dataset].id is required when source = 'hub'")
    if dataset.source == "file" and not (dataset.path or dataset.files):
        raise ConfigError("[dataset].path or [dataset.files] is required for local data")
    if not 0 <= dataset.validation_ratio < 1 or not 0 <= dataset.test_ratio < 1:
        raise ConfigError("Dataset split ratios must be between 0 (inclusive) and 1")
    if dataset.validation_ratio + dataset.test_ratio >= 1:
        raise ConfigError("validation_ratio + test_ratio must be less than 1")
    if dataset.overlap_policy not in {"error", "drop"}:
        raise ConfigError("[dataset].overlap_policy must be 'error' or 'drop'")
    if config.format.kind == "classification" and not config.format.label_map:
        raise ConfigError("[format.label_map] is required for classification")
    if config.format.kind not in {"messages", "instruction", "classification"}:
        raise ConfigError("[format].kind must be 'messages', 'instruction', or 'classification'")
    if config.training.epochs < 1 or config.training.batch_size < 1:
        raise ConfigError("epochs and batch_size must be positive")
    if config.training.max_seq_length < 8:
        raise ConfigError("max_seq_length must be at least 8")
    if not config.output.run_name.strip():
        raise ConfigError("[output].run_name cannot be empty")
    if Path(config.output.run_name).name != config.output.run_name:
        raise ConfigError("[output].run_name must be a single directory name")
    if config.evaluation.split not in {"validation", "test"}:
        raise ConfigError("[evaluation].split must be 'validation' or 'test'")


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a strict TOML configuration file."""
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}: {exc}") from exc

    known = {"model", "dataset", "format", "training", "output", "evaluation"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError(f"Unknown top-level sections: {', '.join(unknown)}")
    config = AppConfig(
        model=_section(ModelConfig, raw.get("model"), "model", required=True),
        dataset=_section(DatasetConfig, raw.get("dataset"), "dataset", required=True),
        format=_section(FormatConfig, raw.get("format"), "format", required=True),
        training=_section(TrainingConfig, raw.get("training"), "training", required=False),
        output=_section(OutputConfig, raw.get("output"), "output", required=False),
        evaluation=_section(EvaluationConfig, raw.get("evaluation"), "evaluation", required=False),
    )
    _validate(config)
    return config
