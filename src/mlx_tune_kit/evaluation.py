"""Evaluate a base model or adapter on canonical reference messages."""

from __future__ import annotations

import gc
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlx_tune_kit.artifacts import save_metrics
from mlx_tune_kit.config import AppConfig
from mlx_tune_kit.data import PreparedData, load_and_prepare
from mlx_tune_kit.errors import MLXTuneKitError
from mlx_tune_kit.inference import generate_response, load_model
from mlx_tune_kit.metrics import classification_metrics, exact_match_metrics

__all__ = ["evaluate", "evaluate_predictions"]


def evaluate_predictions(
    config: AppConfig, expected: list[str], predicted: list[str]
) -> dict[str, Any]:
    if config.format.kind == "classification":
        return classification_metrics(expected, predicted, config.format.label_map.values())
    return exact_match_metrics(expected, predicted)


def _run_model(
    config: AppConfig,
    dataset: Any,
    adapter_path: str | None,
    generator: Callable[..., str],
    model_loader: Callable[[AppConfig, str | None], tuple[Any, Any]],
) -> dict[str, Any]:
    model, processor = model_loader(config, adapter_path)
    expected: list[str] = []
    predicted: list[str] = []
    try:
        for row in dataset:
            messages = row["messages"]
            expected.append(messages[-1]["content"])
            predicted.append(
                generator(
                    model,
                    processor,
                    messages[:-1],
                    max_tokens=config.evaluation.max_tokens,
                )
            )
    finally:
        del model, processor
        gc.collect()
        mx = sys.modules.get("mlx.core")
        if mx is not None:
            mx.clear_cache()
    return {**evaluate_predictions(config, expected, predicted), "predictions": predicted}


def evaluate(
    config: AppConfig,
    *,
    prepared: PreparedData | None = None,
    run_dir: str | Path | None = None,
    generator: Callable[..., str] = generate_response,
    model_loader: Callable[[AppConfig, str | None], tuple[Any, Any]] = load_model,
) -> dict[str, Any]:
    prepared = prepared or load_and_prepare(config)
    split_name = config.evaluation.split
    dataset = prepared.splits[split_name]
    maximum = config.evaluation.max_samples
    if maximum is not None and maximum < len(dataset):
        if maximum < 1:
            raise MLXTuneKitError("evaluation.max_samples must be positive")
        indices = list(range(len(dataset)))
        random.Random(config.training.seed).shuffle(indices)
        dataset = dataset.select(indices[:maximum])
    target_dir = Path(run_dir) if run_dir else config.run_dir
    adapter = target_dir / "adapters.safetensors"
    if not adapter.exists():
        raise MLXTuneKitError(f"Adapter not found: {adapter}. Run 'mlx-tune train' first.")

    results: dict[str, Any] = {
        "provenance": {
            "model": config.model.id,
            "model_revision": config.model.revision,
            "dataset": config.dataset.id or config.dataset.path or "local files",
            "dataset_revision": config.dataset.revision,
            "split": split_name,
            "seed": config.training.seed,
        },
        "fine_tuned": _run_model(config, dataset, str(target_dir), generator, model_loader),
    }
    if config.evaluation.compare_base:
        results["base"] = _run_model(config, dataset, None, generator, model_loader)
    save_metrics(target_dir, results)
    return results
