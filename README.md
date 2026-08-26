# MLX Tune Kit

A configuration-driven template for fine-tuning text models supported by
[MLX-VLM](https://github.com/Blaizzy/mlx-vlm) with LoRA on Apple Silicon Macs.
Users describe the model, dataset, and column mappings in TOML without modifying
Python code.

The repository includes examples for conversations, instruction/response data,
classification, and prompt-injection detection. 
Prompt-injection detection is only an example and is not built into the core package.

## Requirements

- macOS on Apple Silicon for training or inference.
- Python 3.12–3.14 and [`uv`](https://docs.astral.sh/uv/).
- An authenticated Hugging Face account (`hf auth login`) for private or gated
  models and datasets.

Dataset validation, CLI help, and the test suite do not initialize Metal.

## Installation

```bash
uv sync
uv run mlx-tune --help
```

## Tutorial: from JSONL to an adapter

Create `data/my-dataset.jsonl`:

```json
{"instruction":"Translate to Spanish","input":"Good morning","response":"Buenos días"}
{"instruction":"Translate to Spanish","input":"Thank you","response":"Gracias"}
```

Copy [`configs/instruction.toml`](configs/instruction.toml) to
`configs/my-training.toml` and change at least the following values:

```toml
[dataset]
source = "file"
path = "data/my-dataset.jsonl"

[output]
dir = "outputs"
run_name = "my-training"
```

Validate the conversion and generated splits before loading the model:

```bash
uv run mlx-tune validate --config configs/my-training.toml
```

The output shows columns, split sizes, label distributions, duplicates,
cross-split overlaps, and two normalized examples. When a dataset only contains
`train`, the tool creates deterministic `validation` and `test` splits. A
provided `test` split is never used for training or hyperparameter selection.

Train and evaluate the adapter:

```bash
uv run mlx-tune train --config configs/my-training.toml
uv run mlx-tune evaluate --run outputs/my-training
```

Run a single prediction:

```bash
uv run mlx-tune predict --run outputs/my-training --text "Translate: good night"
```

## Supported formats

### Conversations

Set `kind = "messages"` and select a column containing a list of `role` and
`content` objects. The final message must have the `assistant` role. See
[`configs/conversation.toml`](configs/conversation.toml).

### Instruction and response

Set `kind = "instruction"` and configure `instruction_column`,
`response_column`, and the optional `input_column`. See
[`configs/instruction.toml`](configs/instruction.toml).

### Classification

Set `kind = "classification"`, select the text and label columns, and declare
every accepted label in `[format.label_map]`. The model learns to generate the
mapped output exactly. See [`configs/classification.toml`](configs/classification.toml).

Local datasets may use JSON, JSONL, CSV, or Parquet. To provide explicit split
files, use:

```toml
[dataset]
source = "file"

[dataset.files]
train = "data/train.jsonl"
validation = "data/validation.jsonl"
test = "data/test.jsonl"
```

The dataset cache defaults to `.cache/mlx-tune-kit/datasets`. Change it with
`dataset.cache_dir`; the tool does not require write access to the user's global
Hugging Face cache.

For Hugging Face datasets, set `source = "hub"`, `id`, and preferably an exact
`revision`. Map non-standard split names with `train_split`, `validation_split`,
and `test_split`. Cross-split overlap raises an error by default. Set
`overlap_policy = "drop"` to keep each input only in the highest-priority split
(`test`, then `validation`, then `train`) and discard classification inputs with
contradictory labels.

## Reproducible artifacts

Each run creates the following files under `outputs/<run_name>/`:

- `adapters.safetensors`, selected using the lowest validation loss;
- `last_adapters.safetensors`, containing the final training state;
- `resolved-config.json`;
- `dataset-summary.json`;
- `environment.json`, including dependency versions and the Git commit when
  available;
- `manifest.json`;
- `metrics.json` after evaluation.

Outputs are excluded from Git. To share an adapter, create a Hugging Face model
repository, upload the run artifacts, and document the base model, exact
revision, dataset license, and known limitations.

## Evaluation

Generative tasks report normalized exact match. Classification tasks report
accuracy, precision, recall, macro-F1, a confusion matrix, and the invalid-output
rate. Setting `compare_base = true` evaluates the base model on the exact same
examples.

The [`configs/prompt-injection.toml`](configs/prompt-injection.toml) example uses
`xTRam1/safe-guard-prompt-injection`. A classifier like this
must not be the only security boundary. Meaningful validation also requires
out-of-distribution cases, indirect attacks, obfuscation, multilingual data, and
legitimate content discussing prompt injection.

## Project structure

```text
src/mlx_tune_kit/  configuration, data, training, inference, and metrics
configs/           ready-to-copy configuration examples
examples/data/     small versioned datasets
tests/             unit and smoke tests that do not require Metal
```

The package intentionally uses a shallow module hierarchy with explicit public
interfaces. Dataset formats normalize into canonical chat messages before the
MLX backend is loaded.

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run mypy
uv run pytest
uv build
```
