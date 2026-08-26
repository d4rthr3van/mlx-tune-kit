from __future__ import annotations

import pytest
from datasets import Dataset, DatasetDict

from mlx_tune_kit.config import (
    AppConfig,
    DatasetConfig,
    EvaluationConfig,
    FormatConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)


@pytest.fixture
def instruction_config(tmp_path) -> AppConfig:
    return AppConfig(
        model=ModelConfig(id="test/model"),
        dataset=DatasetConfig(
            source="file",
            path="unused.jsonl",
            validation_ratio=0.2,
            test_ratio=0.2,
        ),
        format=FormatConfig(kind="instruction"),
        training=TrainingConfig(epochs=1, seed=7),
        output=OutputConfig(dir=str(tmp_path), run_name="smoke"),
        evaluation=EvaluationConfig(),
    )


@pytest.fixture
def instruction_rows() -> list[dict[str, str]]:
    return [
        {"instruction": f"Question {index}", "input": "", "response": f"Answer {index}"}
        for index in range(10)
    ]


@pytest.fixture
def instruction_dataset(instruction_rows) -> DatasetDict:
    return DatasetDict({"train": Dataset.from_list(instruction_rows)})
