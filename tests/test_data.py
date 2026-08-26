from __future__ import annotations

from dataclasses import replace

import pytest
from datasets import Dataset, DatasetDict

from mlx_tune_kit.config import FormatConfig
from mlx_tune_kit.data import normalize_row, prepare_dataset_dict
from mlx_tune_kit.errors import DatasetError


def test_normalize_instruction_with_optional_input() -> None:
    row = {"instruction": "Translate", "input": "hello", "response": "hola"}
    result = normalize_row(row, FormatConfig(kind="instruction", system_prompt="Help"))
    assert result["messages"] == [
        {"role": "system", "content": "Help"},
        {"role": "user", "content": "Translate\n\nhello"},
        {"role": "assistant", "content": "hola"},
    ]


def test_normalize_classification() -> None:
    fmt = FormatConfig(kind="classification", label_map={"0": "NO", "1": "YES"})
    result = normalize_row({"text": "example", "label": 1}, fmt)
    assert result["messages"][-1]["content"] == "YES"
    assert result["label"] == "1"


def test_normalize_messages_rejects_missing_assistant() -> None:
    fmt = FormatConfig(kind="messages")
    with pytest.raises(DatasetError, match="final message"):
        normalize_row(
            {"messages": [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}]},
            fmt,
        )


def test_created_splits_are_deterministic_and_disjoint(
    instruction_config, instruction_dataset
) -> None:
    first = prepare_dataset_dict(instruction_dataset, instruction_config)
    second = prepare_dataset_dict(instruction_dataset, instruction_config)
    assert first.splits["train"]["messages"] == second.splits["train"]["messages"]
    assert first.summary.rows == {"train": 6, "test": 2, "validation": 2}
    assert all(value == 0 for value in first.summary.overlaps.values())


def test_created_splits_keep_duplicate_inputs_together(instruction_config) -> None:
    rows = [
        {"instruction": f"Question {index}", "input": "", "response": f"Answer {index}"}
        for index in range(6)
        for _ in range(2)
    ]
    prepared = prepare_dataset_dict(
        DatasetDict({"train": Dataset.from_list(rows)}), instruction_config
    )

    assert prepared.summary.rows == {"train": 8, "test": 2, "validation": 2}
    assert all(value == 0 for value in prepared.summary.overlaps.values())


def test_preserves_provided_test(instruction_config, instruction_rows) -> None:
    raw = DatasetDict(
        {
            "train": Dataset.from_list(instruction_rows),
            "test": Dataset.from_list(
                [{"instruction": "Held out", "input": "", "response": "Never trained"}]
            ),
        }
    )
    prepared = prepare_dataset_dict(raw, instruction_config)
    assert prepared.splits["test"][0]["messages"][-1]["content"] == "Never trained"
    assert prepared.summary.rows["test"] == 1


def test_rejects_cross_split_overlap(instruction_config, instruction_rows) -> None:
    validation = [instruction_rows[0]]
    raw = DatasetDict(
        {
            "train": Dataset.from_list(instruction_rows),
            "validation": Dataset.from_list(validation),
            "test": Dataset.from_list(
                [{"instruction": "Held out", "input": "", "response": "answer"}]
            ),
        }
    )
    with pytest.raises(DatasetError, match="identical examples"):
        prepare_dataset_dict(raw, instruction_config)


def test_drop_overlap_policy_preserves_held_out_split(
    instruction_config, instruction_rows
) -> None:
    config = replace(
        instruction_config,
        dataset=replace(instruction_config.dataset, overlap_policy="drop"),
    )
    raw = DatasetDict(
        {
            "train": Dataset.from_list(instruction_rows),
            "validation": Dataset.from_list(instruction_rows[1:3]),
            "test": Dataset.from_list([instruction_rows[0]]),
        }
    )

    prepared = prepare_dataset_dict(raw, config)

    assert prepared.summary.rows == {"train": 7, "validation": 2, "test": 1}
    assert all(value == 0 for value in prepared.summary.overlaps.values())
    assert any("lower-priority splits" in warning for warning in prepared.summary.warnings)


def test_sample_limit_is_not_source_prefix(instruction_config, instruction_dataset) -> None:
    limited = replace(
        instruction_config,
        training=replace(instruction_config.training, max_train_samples=2),
    )
    prepared = prepare_dataset_dict(instruction_dataset, limited)
    values = [row["messages"][1]["content"] for row in prepared.splits["train"]]
    assert values != ["Question 0", "Question 1"]
