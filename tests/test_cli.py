from __future__ import annotations

import pytest

from mlx_tune_kit.cli import main


def test_top_level_help_does_not_initialize_metal() -> None:
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0


def test_validate_local_example(capsys) -> None:
    assert main(["validate", "--config", "configs/instruction.toml"]) == 0
    output = capsys.readouterr().out
    assert "Canonical examples" in output
    assert "validation was created" in output


def test_invalid_config_returns_actionable_error(tmp_path, capsys) -> None:
    missing = tmp_path / "missing.toml"
    assert main(["validate", "--config", str(missing)]) == 2
    assert "Configuration file not found" in capsys.readouterr().err
