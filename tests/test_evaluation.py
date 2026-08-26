from __future__ import annotations

from dataclasses import replace

from mlx_tune_kit.data import prepare_dataset_dict
from mlx_tune_kit.evaluation import evaluate, evaluate_predictions


def test_evaluate_predictions_uses_task_metrics(instruction_config) -> None:
    generative = evaluate_predictions(instruction_config, ["Hello"], [" hello "])
    assert generative["exact_match"] == 1.0

    classification_config = replace(
        instruction_config,
        format=replace(
            instruction_config.format,
            kind="classification",
            label_map={"0": "NO", "1": "YES"},
        ),
    )
    classification = evaluate_predictions(classification_config, ["YES", "NO"], ["yes", "invalid"])
    assert classification["accuracy"] == 0.5
    assert classification["invalid_rate"] == 0.5


def test_evaluate_base_and_adapter_without_metal(
    instruction_config, instruction_dataset, tmp_path
) -> None:
    config = replace(
        instruction_config,
        evaluation=replace(instruction_config.evaluation, compare_base=True),
    )
    prepared = prepare_dataset_dict(instruction_dataset, config)
    run_dir = tmp_path / "evaluation-run"
    run_dir.mkdir()
    (run_dir / "adapters.safetensors").write_bytes(b"fake")
    loaded_adapters: list[str | None] = []

    def fake_loader(config, adapter_path):
        loaded_adapters.append(adapter_path)
        return object(), object()

    def fake_generator(model, processor, messages, *, max_tokens):
        question_number = messages[-1]["content"].split()[-1]
        return f"Answer {question_number}"

    results = evaluate(
        config,
        prepared=prepared,
        run_dir=run_dir,
        model_loader=fake_loader,
        generator=fake_generator,
    )
    assert results["fine_tuned"]["exact_match"] == 1.0
    assert results["base"]["exact_match"] == 1.0
    assert loaded_adapters == [str(run_dir), None]
    assert (run_dir / "metrics.json").exists()
