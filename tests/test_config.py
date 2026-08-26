from __future__ import annotations

import pytest

from mlx_tune_kit.config import load_config
from mlx_tune_kit.errors import ConfigError


def test_load_example_configs() -> None:
    expected = {
        "conversation.toml": "messages",
        "instruction.toml": "instruction",
        "classification.toml": "classification",
        "prompt-injection.toml": "classification",
    }
    for path, kind in expected.items():
        assert load_config(f"configs/{path}").format.kind == kind


def test_rejects_unknown_keys(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[model]
id = "model"
typo = true
[dataset]
source = "file"
path = "data.jsonl"
[format]
kind = "messages"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_config(path)


def test_classification_requires_label_map(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[model]
id = "model"
[dataset]
source = "file"
path = "data.csv"
[format]
kind = "classification"
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="label_map"):
        load_config(path)
