from pathlib import Path

import pytest

from mlx_tune_kit.data import prepare_dataset_dict
from mlx_tune_kit.errors import MLXTuneKitError
from mlx_tune_kit.training import _select_and_prune_checkpoints, train


def test_training_smoke_with_backend(instruction_config, instruction_dataset) -> None:
    prepared = prepare_dataset_dict(instruction_dataset, instruction_config)

    def fake_backend(config, data, run_dir: Path) -> None:
        assert len(data.splits["validation"]) == 2
        (run_dir / "adapters.safetensors").write_bytes(b"fake")

    run_dir = train(instruction_config, prepared=prepared, backend=fake_backend)
    assert (run_dir / "resolved-config.json").exists()
    assert (run_dir / "dataset-summary.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "adapters.safetensors").read_bytes() == b"fake"


def test_training_refuses_to_overwrite_existing_run(
    instruction_config, instruction_dataset
) -> None:
    prepared = prepare_dataset_dict(instruction_dataset, instruction_config)

    def fake_backend(config, data, run_dir: Path) -> None:
        (run_dir / "adapters.safetensors").write_bytes(b"fake")

    train(instruction_config, prepared=prepared, backend=fake_backend)
    with pytest.raises(MLXTuneKitError, match="already contains"):
        train(instruction_config, prepared=prepared, backend=fake_backend)


def test_checkpoint_selection_keeps_best_and_final(instruction_config, tmp_path) -> None:
    run_dir = tmp_path / "checkpoints"
    run_dir.mkdir()
    (run_dir / "adapters.safetensors").write_bytes(b"final")
    (run_dir / "best_adapters.safetensors").write_bytes(b"best")
    (run_dir / "0000200_adapters.safetensors").write_bytes(b"periodic")

    retained = _select_and_prune_checkpoints(run_dir, instruction_config)

    assert retained == ["adapters.safetensors", "last_adapters.safetensors"]
    assert (run_dir / "adapters.safetensors").read_bytes() == b"best"
    assert (run_dir / "last_adapters.safetensors").read_bytes() == b"final"
    assert not (run_dir / "0000200_adapters.safetensors").exists()
