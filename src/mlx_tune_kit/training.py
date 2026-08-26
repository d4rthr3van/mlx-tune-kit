"""MLX-VLM QLoRA training orchestration."""

from __future__ import annotations

import argparse
import math
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlx_tune_kit.artifacts import initialize_run, write_manifest
from mlx_tune_kit.config import AppConfig
from mlx_tune_kit.data import PreparedData, load_and_prepare
from mlx_tune_kit.errors import MLXTuneKitError, PlatformError

__all__ = ["train"]


def _check_platform() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise PlatformError("Training requires macOS on Apple Silicon (arm64)")


def _select_and_prune_checkpoints(run_dir: Path, config: AppConfig) -> list[str]:
    numbered = sorted(run_dir.glob("[0-9]*_adapters.safetensors"))
    for checkpoint in numbered:
        checkpoint.unlink()
    final_adapter = run_dir / "adapters.safetensors"
    best_adapter = run_dir / "best_adapters.safetensors"
    last_adapter = run_dir / "last_adapters.safetensors"
    if config.output.keep_best and best_adapter.exists():
        final_adapter.replace(last_adapter)
        best_adapter.replace(final_adapter)
        retained = [final_adapter.name]
        if config.output.keep_last:
            retained.append(last_adapter.name)
        else:
            last_adapter.unlink()
    else:
        best_adapter.unlink(missing_ok=True)
        retained = [final_adapter.name]
    return retained


def train(
    config: AppConfig,
    *,
    prepared: PreparedData | None = None,
    backend: Callable[[AppConfig, PreparedData, Path], Any] | None = None,
) -> Path:
    """Prepare data, initialize a reproducible run and train an adapter."""
    if (config.run_dir / "resolved-config.json").exists():
        raise MLXTuneKitError(
            f"Run directory already contains a run: {config.run_dir}. "
            "Choose a different output.run_name."
        )
    prepared = prepared or load_and_prepare(config)
    run_dir = initialize_run(config, prepared.summary)
    if backend is not None:
        backend(config, prepared, run_dir)
        checkpoints = (
            ["adapters.safetensors"] if (run_dir / "adapters.safetensors").exists() else []
        )
        write_manifest(run_dir, status="completed", checkpoints=checkpoints)
        return run_dir

    _check_platform()
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx_vlm import load
        from mlx_vlm.lora import setup_model_for_training
        from mlx_vlm.trainer import sft_trainer
        from mlx_vlm.trainer.datasets import VisionDataset
        from mlx_vlm.trainer.sft_trainer import TrainingArgs
        from mlx_vlm.trainer.utils import print_trainable_parameters, save_adapter
    except (ImportError, RuntimeError) as exc:
        write_manifest(run_dir, status="failed", checkpoints=[], error=str(exc))
        raise PlatformError(
            "MLX/Metal is unavailable; run training in a local Apple Silicon terminal"
        ) from exc

    mx.random.seed(config.training.seed)
    load_kwargs: dict[str, Any] = {"revision": config.model.revision}
    if config.model.trust_remote_code:
        load_kwargs["processor_config"] = {"trust_remote_code": True}
    try:
        model, processor = load(config.model.id, **load_kwargs)
        namespace = argparse.Namespace(
            full_finetune=False,
            train_vision=False,
            lora_rank=config.training.lora_rank,
            lora_alpha=config.training.lora_alpha,
            lora_dropout=config.training.lora_dropout,
        )
        model = setup_model_for_training(model, namespace)
        print_trainable_parameters(model)
        model_config = model.config.__dict__
        train_data = VisionDataset(
            prepared.splits["train"], model_config, processor, train_on_completions=True
        )
        validation_data = VisionDataset(
            prepared.splits["validation"], model_config, processor, train_on_completions=True
        )
        iterations = (
            math.ceil(len(train_data) / config.training.batch_size) * config.training.epochs
        )
        args = TrainingArgs(
            batch_size=config.training.batch_size,
            iters=iterations,
            steps_per_report=10,
            steps_per_eval=config.training.steps_per_eval,
            steps_per_save=config.training.steps_per_save,
            val_batches=config.training.val_batches,
            max_seq_length=config.training.max_seq_length,
            adapter_file=str(run_dir / "adapters.safetensors"),
            grad_checkpoint=True,
            grad_clip=1.0,
            learning_rate=config.training.learning_rate,
            warmup_steps=min(50, max(1, iterations // 20)),
            min_learning_rate=config.training.learning_rate / 10,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        )
        best_validation_loss = float("inf")
        original_evaluate = sft_trainer.evaluate

        def track_best(*evaluate_args: Any, **evaluate_kwargs: Any) -> float:
            nonlocal best_validation_loss
            loss = original_evaluate(*evaluate_args, **evaluate_kwargs)
            if config.output.keep_best and loss < best_validation_loss:
                best_validation_loss = loss
                candidate_model = evaluate_kwargs.get("model") or evaluate_args[0]
                save_adapter(candidate_model, run_dir / "best_adapters.safetensors")
            return loss

        sft_trainer.evaluate = track_best
        try:
            sft_trainer.train(
                model=model,
                optimizer=optim.AdamW(
                    learning_rate=config.training.learning_rate, weight_decay=0.01
                ),
                train_dataset=train_data,
                val_dataset=validation_data,
                args=args,
                train_on_completions=True,
            )
        finally:
            sft_trainer.evaluate = original_evaluate
    except Exception as exc:
        write_manifest(run_dir, status="failed", checkpoints=[], error=str(exc))
        if "memory" in str(exc).lower():
            raise MLXTuneKitError(
                "Training ran out of memory; lower max_seq_length or batch_size"
            ) from exc
        raise

    retained = _select_and_prune_checkpoints(run_dir, config)
    write_manifest(
        run_dir,
        status="completed",
        checkpoints=retained,
        best_checkpoint=("adapters.safetensors" if config.output.keep_best else None),
        best_validation_loss=(
            best_validation_loss if best_validation_loss != float("inf") else None
        ),
        last_checkpoint=(
            "last_adapters.safetensors"
            if "last_adapters.safetensors" in retained
            else "adapters.safetensors"
        ),
    )
    return run_dir
