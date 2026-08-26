"""Dataset loading, deterministic splitting and canonical normalization."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlx_tune_kit.config import AppConfig, DatasetConfig, FormatConfig
from mlx_tune_kit.errors import DatasetError

__all__ = [
    "DatasetSummary",
    "PreparedData",
    "load_and_prepare",
    "normalize_row",
    "prepare_dataset_dict",
]


@dataclass
class DatasetSummary:
    source: str
    columns: dict[str, list[str]] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    label_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    duplicates: dict[str, int] = field(default_factory=dict)
    overlaps: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PreparedData:
    splits: dict[str, Any]
    summary: DatasetSummary


def _required_columns(fmt: FormatConfig) -> set[str]:
    if fmt.kind == "messages":
        return {fmt.messages_column}
    if fmt.kind == "instruction":
        return {fmt.instruction_column, fmt.response_column}
    return {fmt.text_column, fmt.label_column}


def _text(value: Any, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise DatasetError(f"Column '{field_name}' contains an empty value")
    return str(value).strip()


def normalize_row(row: Mapping[str, Any], fmt: FormatConfig) -> dict[str, Any]:
    """Convert a supported source row into canonical chat messages."""
    missing = sorted(_required_columns(fmt) - set(row))
    if missing:
        raise DatasetError(f"Missing required columns: {', '.join(missing)}")

    if fmt.kind == "messages":
        source_messages = row[fmt.messages_column]
        if not isinstance(source_messages, list) or len(source_messages) < 2:
            raise DatasetError("A messages row must contain at least two messages")
        messages = []
        for message in source_messages:
            if not isinstance(message, Mapping):
                raise DatasetError("Each message must be an object")
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise DatasetError(f"Unsupported message role: {role!r}")
            content = message.get("content")
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, Mapping) and part.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            messages.append({"role": role, "content": _text(content, "content")})
        if messages[-1]["role"] != "assistant":
            raise DatasetError("The final message must have role 'assistant'")
        return {"messages": messages}

    canonical_messages: list[dict[str, str]] = []
    if fmt.system_prompt:
        canonical_messages.append({"role": "system", "content": fmt.system_prompt.strip()})
    if fmt.kind == "instruction":
        instruction = _text(row[fmt.instruction_column], fmt.instruction_column)
        if fmt.input_column and row.get(fmt.input_column) not in (None, ""):
            instruction = f"{instruction}\n\n{str(row[fmt.input_column]).strip()}"
        response = _text(row[fmt.response_column], fmt.response_column)
        canonical_messages.extend(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
        )
        return {"messages": canonical_messages}

    raw_label = str(row[fmt.label_column])
    if raw_label not in fmt.label_map:
        raise DatasetError(f"Unknown label {raw_label!r}; add it to [format.label_map]")
    canonical_messages.extend(
        [
            {"role": "user", "content": _text(row[fmt.text_column], fmt.text_column)},
            {"role": "assistant", "content": fmt.label_map[raw_label]},
        ]
    )
    return {"messages": canonical_messages, "label": raw_label}


def _extension_loader(path: str) -> str:
    suffix = Path(path).suffix.lower()
    loaders = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet"}
    if suffix not in loaders:
        raise DatasetError(f"Unsupported dataset extension {suffix!r}; use JSONL, CSV or Parquet")
    return loaders[suffix]


def _load_raw(config: DatasetConfig) -> Mapping[str, Any]:
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise DatasetError("Install the project dependencies with 'uv sync'") from exc

    try:
        if config.source == "hub":
            return load_dataset(
                config.id,
                config.name,
                revision=config.revision,
                cache_dir=config.cache_dir,
            )
        if config.files:
            files = dict(config.files)
            loader_types = {_extension_loader(path) for path in files.values()}
            if len(loader_types) != 1:
                raise DatasetError("All files in [dataset.files] must use the same format")
            return load_dataset(loader_types.pop(), data_files=files, cache_dir=config.cache_dir)
        assert config.path is not None
        return DatasetDict(
            {
                "train": load_dataset(
                    _extension_loader(config.path),
                    data_files=config.path,
                    split="train",
                    cache_dir=config.cache_dir,
                )
            }
        )
    except DatasetError:
        raise
    except Exception as exc:
        message = str(exc)
        if "auth" in message.lower() or "gated" in message.lower():
            raise DatasetError("Dataset access failed; authenticate with 'hf auth login'") from exc
        raise DatasetError(f"Could not load dataset: {message}") from exc


def _input_fingerprint(row: Mapping[str, Any], fmt: FormatConfig) -> str:
    messages = normalize_row(row, fmt)["messages"][:-1]
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()


def _deterministic_groups(dataset: Any, fmt: FormatConfig, seed: int) -> list[list[int]]:
    """Order input-equivalent row groups, stratified for classification."""
    equivalent: dict[str, list[int]] = {}
    group_labels: dict[str, str] = {}
    for index, row in enumerate(dataset):
        fingerprint = _input_fingerprint(row, fmt)
        equivalent.setdefault(fingerprint, []).append(index)
        if fmt.kind == "classification":
            label = str(row[fmt.label_column])
            previous = group_labels.setdefault(fingerprint, label)
            if previous != label:
                raise DatasetError("Identical classification inputs have conflicting labels")

    groups: dict[str, list[list[int]]] = {}
    for fingerprint, indices in equivalent.items():
        label = group_labels[fingerprint] if fmt.kind == "classification" else "all"
        groups.setdefault(label, []).append(indices)
    rng = random.Random(seed)
    for grouped_indices in groups.values():
        rng.shuffle(grouped_indices)
    ordered: list[list[int]] = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered


def _deterministic_indices(dataset: Any, fmt: FormatConfig, seed: int) -> list[int]:
    return [index for group in _deterministic_groups(dataset, fmt, seed) for index in group]


def _take_split(dataset: Any, ratio: float, fmt: FormatConfig, seed: int) -> tuple[Any, Any]:
    if ratio <= 0:
        return dataset, dataset.select([])
    if len(dataset) < 3:
        raise DatasetError("At least three training rows are required to create splits")
    count = max(1, round(len(dataset) * ratio))
    count = min(count, len(dataset) - 1)
    held_out: set[int] = set()
    for group in _deterministic_groups(dataset, fmt, seed):
        if len(held_out) >= count:
            break
        if len(held_out) + len(group) < len(dataset):
            held_out.update(group)
    if not held_out:
        raise DatasetError("The dataset has too few distinct inputs to create a split")
    return (
        dataset.select([index for index in range(len(dataset)) if index not in held_out]),
        dataset.select(sorted(held_out)),
    )


def _create_validation_and_test(
    dataset: Any, validation_ratio: float, test_ratio: float, fmt: FormatConfig, seed: int
) -> tuple[Any, Any, Any]:
    if len(dataset) < 3:
        raise DatasetError("At least three training rows are required to create splits")
    validation_count = max(1, round(len(dataset) * validation_ratio))
    test_count = max(1, round(len(dataset) * test_ratio))
    if validation_count + test_count >= len(dataset):
        raise DatasetError("The dataset is too small for the configured validation/test ratios")
    ordered = _deterministic_groups(dataset, fmt, seed)
    validation_indices: set[int] = set()
    test_indices: set[int] = set()
    for group in ordered:
        if len(validation_indices) < validation_count:
            validation_indices.update(group)
        elif len(test_indices) < test_count:
            test_indices.update(group)
        else:
            break
    if not validation_indices or not test_indices:
        raise DatasetError("The dataset has too few distinct inputs to create splits")
    if len(validation_indices) + len(test_indices) >= len(dataset):
        raise DatasetError("Duplicate groups leave no training rows for the configured splits")
    train_indices = [
        index
        for index in range(len(dataset))
        if index not in validation_indices and index not in test_indices
    ]
    return (
        dataset.select(train_indices),
        dataset.select(sorted(validation_indices)),
        dataset.select(sorted(test_indices)),
    )


def _rename_splits(raw: Mapping[str, Any], cfg: DatasetConfig) -> dict[str, Any]:
    aliases = {
        "train": cfg.train_split,
        "validation": cfg.validation_split,
        "test": cfg.test_split,
    }
    result = {canonical: raw[source] for canonical, source in aliases.items() if source in raw}
    if "train" not in result:
        raise DatasetError(f"Training split {cfg.train_split!r} was not found")
    return result


def _sanitize_source_splits(
    splits: dict[str, Any], fmt: FormatConfig
) -> tuple[dict[str, Any], list[str]]:
    """Remove ambiguous labels and keep overlapping inputs only in the safest split."""
    labels_by_input: dict[str, set[str]] = {}
    if fmt.kind == "classification":
        for dataset in splits.values():
            for row in dataset:
                fingerprint = _input_fingerprint(row, fmt)
                labels_by_input.setdefault(fingerprint, set()).add(str(row[fmt.label_column]))
    conflicting = {
        fingerprint for fingerprint, labels in labels_by_input.items() if len(labels) > 1
    }

    claimed: set[str] = set()
    cleaned: dict[str, Any] = {}
    removed_conflicts = 0
    removed_overlaps = 0
    for name in ("test", "validation", "train"):
        if name not in splits:
            continue
        dataset = splits[name]
        kept: list[int] = []
        present: set[str] = set()
        for index, row in enumerate(dataset):
            fingerprint = _input_fingerprint(row, fmt)
            if fingerprint in conflicting:
                removed_conflicts += 1
            elif fingerprint in claimed:
                removed_overlaps += 1
            else:
                kept.append(index)
                present.add(fingerprint)
        cleaned[name] = dataset.select(kept)
        claimed.update(present)

    warnings: list[str] = []
    if removed_conflicts:
        warnings.append(
            f"dropped {removed_conflicts} rows whose identical inputs had conflicting labels"
        )
    if removed_overlaps:
        warnings.append(
            f"dropped {removed_overlaps} rows from lower-priority splits to prevent overlap"
        )
    return cleaned, warnings


def _limit(dataset: Any, maximum: int | None, fmt: FormatConfig, seed: int) -> Any:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise DatasetError("Sample limits must be positive")
    return dataset.select(_deterministic_indices(dataset, fmt, seed)[:maximum])


def _fingerprints(dataset: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()
        for row in dataset
    ]


def prepare_dataset_dict(raw: Mapping[str, Any], config: AppConfig) -> PreparedData:
    """Split, validate and normalize an already loaded DatasetDict-like object."""
    cfg = config.dataset
    fmt = config.format
    splits = _rename_splits(raw, cfg)
    warnings: list[str] = []
    if cfg.overlap_policy == "drop":
        splits, sanitization_warnings = _sanitize_source_splits(splits, fmt)
        warnings.extend(sanitization_warnings)
    if "test" not in splits and "validation" not in splits:
        splits["train"], splits["validation"], splits["test"] = _create_validation_and_test(
            splits["train"],
            cfg.validation_ratio,
            cfg.test_ratio,
            fmt,
            config.training.seed,
        )
        warnings.extend(
            [
                "validation was created deterministically from train",
                "test was created deterministically from train",
            ]
        )
    elif "test" not in splits:
        splits["train"], splits["test"] = _take_split(
            splits["train"], cfg.test_ratio, fmt, config.training.seed + 1
        )
        warnings.append("test was created deterministically from train")
    if "validation" not in splits:
        splits["train"], splits["validation"] = _take_split(
            splits["train"], cfg.validation_ratio, fmt, config.training.seed
        )
        warnings.append("validation was created deterministically from train")

    splits["train"] = _limit(
        splits["train"], config.training.max_train_samples, fmt, config.training.seed
    )
    splits["validation"] = _limit(
        splits["validation"],
        config.training.max_validation_samples,
        fmt,
        config.training.seed,
    )
    normalized: dict[str, Any] = {}
    columns: dict[str, list[str]] = {}
    labels: dict[str, dict[str, int]] = {}
    for name, dataset in splits.items():
        if len(dataset) == 0:
            raise DatasetError(f"Split {name!r} is empty")
        columns[name] = list(dataset.column_names)
        missing = _required_columns(fmt) - set(dataset.column_names)
        if missing:
            raise DatasetError(f"Split {name!r} is missing columns: {', '.join(sorted(missing))}")
        try:
            normalized[name] = dataset.map(
                lambda row: normalize_row(row, fmt),
                remove_columns=dataset.column_names,
                desc=f"Normalizing {name}",
            )
        except DatasetError:
            raise
        except Exception as exc:
            cause = exc.__cause__ or exc
            raise DatasetError(f"Invalid row in split {name!r}: {cause}") from exc
        if fmt.kind == "classification":
            labels[name] = dict(Counter(str(value) for value in dataset[fmt.label_column]))

    fingerprints = {name: _fingerprints(dataset) for name, dataset in normalized.items()}
    input_fingerprints = {
        name: {
            hashlib.sha256(json.dumps(row["messages"][:-1], sort_keys=True).encode()).hexdigest()
            for row in dataset
        }
        for name, dataset in normalized.items()
    }
    duplicate_counts = {
        name: len(values) - len(set(values)) for name, values in fingerprints.items()
    }
    overlaps: dict[str, int] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = len(input_fingerprints[left] & input_fingerprints[right])
        overlaps[f"{left}:{right}"] = overlap
        if overlap:
            raise DatasetError(f"Found {overlap} identical examples across {left} and {right}")

    summary = DatasetSummary(
        source=config.dataset.id or config.dataset.path or "local files",
        columns=columns,
        rows={name: len(dataset) for name, dataset in normalized.items()},
        label_distribution=labels,
        duplicates=duplicate_counts,
        overlaps=overlaps,
        warnings=warnings,
        examples=[normalized["train"][index] for index in range(min(2, len(normalized["train"])))],
    )
    return PreparedData(splits=normalized, summary=summary)


def load_and_prepare(config: AppConfig) -> PreparedData:
    """Load a configured dataset and produce canonical, disjoint splits."""
    return prepare_dataset_dict(_load_raw(config.dataset), config)
