"""Command-line interface for reusable fine-tuning runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from mlx_tune_kit.artifacts import load_run_config
from mlx_tune_kit.config import AppConfig, load_config
from mlx_tune_kit.data import load_and_prepare
from mlx_tune_kit.errors import MLXTuneKitError
from mlx_tune_kit.evaluation import evaluate
from mlx_tune_kit.inference import generate_response, load_model
from mlx_tune_kit.training import train

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-tune",
        description="Fine-tune text datasets with MLX-VLM on Apple Silicon",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate and preview a dataset without loading a model"
    )
    validate_parser.add_argument("--config", required=True, type=Path)
    validate_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    train_parser = subparsers.add_parser("train", help="Train a LoRA adapter")
    train_parser.add_argument("--config", required=True, type=Path)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate an existing run")
    evaluate_parser.add_argument("--run", required=True, type=Path)

    predict_parser = subparsers.add_parser("predict", help="Generate one response")
    source = predict_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--run", type=Path)
    predict_parser.add_argument("--text", help="User text; reads stdin when omitted")
    return parser


def _print_summary(config: AppConfig, as_json: bool) -> None:
    prepared = load_and_prepare(config)
    payload = asdict(prepared.summary)
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    print(f"Dataset: {payload['source']}")
    print(f"Format: {config.format.kind}")
    for split in ("train", "validation", "test"):
        labels = payload["label_distribution"].get(split)
        label_suffix = f" labels={labels}" if labels else ""
        print(
            f"- {split}: {payload['rows'][split]} rows; "
            f"columns={payload['columns'][split]}{label_suffix}; "
            f"duplicates={payload['duplicates'][split]}"
        )
    for pair, count in payload["overlaps"].items():
        print(f"- overlap {pair}: {count}")
    for warning in payload["warnings"]:
        print(f"WARNING: {warning}")
    print("Canonical examples:")
    print(json.dumps(payload["examples"], indent=2, ensure_ascii=False))


def _prediction_messages(config: AppConfig, text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if config.format.system_prompt:
        messages.append({"role": "system", "content": config.format.system_prompt})
    messages.append({"role": "user", "content": text})
    return messages


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            _print_summary(load_config(args.config), args.json)
            return 0
        if args.command == "train":
            run_dir = train(load_config(args.config))
            print(f"Run completed: {run_dir}")
            return 0
        if args.command == "evaluate":
            config = load_run_config(args.run)
            results = evaluate(config, run_dir=args.run)
            print(json.dumps(results, indent=2, ensure_ascii=False))
            return 0
        if args.command == "predict":
            config = load_config(args.config) if args.config else load_run_config(args.run)
            adapter_path = str(args.run) if args.run else None
            model, processor = load_model(config, adapter_path)
            text = args.text if args.text is not None else sys.stdin.read()
            if not text.strip():
                raise MLXTuneKitError("Prediction text cannot be empty")
            print(
                generate_response(
                    model,
                    processor,
                    _prediction_messages(config, text.strip()),
                    max_tokens=config.evaluation.max_tokens,
                )
            )
            return 0
    except (MLXTuneKitError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
