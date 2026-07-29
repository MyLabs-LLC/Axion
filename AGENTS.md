# Repository Guidelines

## Project Structure

- `src/axion/` contains the Python package, model stages, data loading, distributed pipeline, metrics, and configuration code.
- `scripts/` contains operational tools for data preparation, training launches, evaluation, checkpoint merging, and node synchronization.
- `configs/` contains YAML configurations for model sizes and training runs.
- `tests/` contains the pytest suite. Generated datasets, checkpoints, experiment logs, and MLflow state belong in local `data/`, `relics/`, `experiments/`, `runs/`, or `mlruns/` directories.

## Build, Test, and Development Commands

Use the repository-managed environment through `uv`:

```bash
uv sync                 # install or update declared dependencies
uv run pytest -q        # run the test suite
uv run python -m py_compile scripts/*.py src/axion/*.py
```

For a local smoke run, use the smallest config and a short step count. Three-node training is launched with `scripts/launch_m1.sh`; inspect the selected config before starting GPU work.

## Coding Style & Naming

Use Python 3.12-compatible code, four-space indentation, type annotations, and concise comments only where behavior is non-obvious. Follow `snake_case` for functions and variables, `PascalCase` for classes, and descriptive lowercase names for configs and run directories. Prefer existing helpers and PyTorch/distributed patterns over new abstractions.

## Testing Guidelines

Tests use pytest and are organized by module, for example `tests/test_data.py` and `tests/test_model.py`. Name tests `test_<behavior>`. Add focused regression coverage for data contracts, checkpoint recovery, distributed coordination, and metric changes. Run `uv run pytest -q` before committing.

## Commits and Pull Requests

Use short, imperative commit subjects such as `Fix corpus resume validation` or `Add 250M training config`. Keep commits focused. Pull requests should explain the change, list verification commands, identify configuration or dataset assumptions, and include relevant MLflow run names or metric artifacts for training changes. Never commit datasets, checkpoints, secrets, local databases, or generated logs.

## Distributed Training and Configuration

Treat `oblivion`, `oblivion1`, and `oblivion2` as separate slow-link nodes. Validate corpus metadata and tokenizer compatibility before synchronizing data. Record architecture, token counts, validation scores, and communication metrics for every substantial run. Keep model and data paths configurable; do not hard-code machine-local paths in library code.
