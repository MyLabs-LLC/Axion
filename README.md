# Axion

Axion is a small language-model training scaffold for proving out distributed pretraining on consumer GPUs. The first milestone is `axion-1m`: a decoder-only transformer that can run as a single-GPU baseline or as a three-stage pipeline across the Subhalo-0 machines.

The project uses a dark-sector naming scheme in logs and docs, but the core ML modules use plain names:

- `src/axion/model.py`: transformer model
- `src/axion/data.py`: FineWeb-Edu token memmap reader
- `src/axion/train_single.py`: single-process baseline
- `src/axion/train_pipeline.py`: three-rank pipeline smoke path
- `scripts/haloscope.py`: sample from a relic checkpoint
- `scripts/rotation_curve.py`: summarize run metrics

See [CLAUDE.md](CLAUDE.md) for the full architecture, milestone definitions, and Subhalo-0 runbook.

## Status

Current scaffold:

- FineWeb-Edu tokenizer/data preparation is implemented.
- Normal training requires `data/fineweb_edu_100M.bin`; synthetic data is only available when explicitly enabled for tests.
- `axion-1m` single-GPU smoke runs on FineWeb-Edu tokens.
- `axion-1m` local three-rank pipeline smoke runs with CPU/gloo.
- `axion-25m` single-GPU smoke config is available for the next larger rung.
- Full M1 acceptance still needs the three-host CUDA pipeline run and parity gate.

## Requirements

- Python 3.11+
- `uv`
- CUDA-capable PyTorch for GPU runs
- Optional but expected for M1: three RTX 3090-class hosts with passwordless SSH from macho0

Install the project environment:

```bash
uv sync --extra dev
```

Run tests:

```bash
uv run pytest -q
```

## Data

Training uses FineWeb-Edu `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, tokenized into a uint16 memmap.

Train the 4k SentencePiece tokenizer:

```bash
uv run scripts/train_tokenizer.py
```

Build the full local token file:

```bash
uv run scripts/prepare_data.py --tokens 102000000
```

For a quick validation corpus, use a smaller token count:

```bash
uv run scripts/prepare_data.py --tokens 2000000
```

Generated data is intentionally ignored by git:

- `data/tokenizer/axion4k.model`
- `data/fineweb_edu_100M.bin`
- `data/meta.json`

## Single-GPU Runs

Run the 1M baseline:

```bash
uv run -m axion.train_single --config configs/axion_1m.yaml
```

Run a bounded smoke:

```bash
uv run -m axion.train_single --config configs/axion_1m.yaml --steps 2
```

Run the next larger model smoke:

```bash
uv run -m axion.train_single --config configs/axion_25m.yaml --steps 1
```

Each run writes:

- `runs/<timestamp>/metrics.jsonl`
- `relics/<timestamp>/relic-0-PQ.pt`
- `relics/<timestamp>/final.pt`

Summarize a run:

```bash
uv run scripts/rotation_curve.py runs/<timestamp>
```

Sample from a final relic:

```bash
uv run scripts/haloscope.py relics/<timestamp>/final.pt --prompt "The universe"
```

## Pipeline Smoke

Local three-rank CPU/gloo smoke:

```bash
CUDA_VISIBLE_DEVICES='' uv run torchrun --standalone --nproc-per-node=3 \
  -m axion.train_pipeline --config configs/axion_1m.yaml --steps 1
```

Subhalo-0 launch from macho0:

```bash
./scripts/launch_m1.sh configs/axion_1m.yaml
```

Pipeline runs write per-stage shards under `relics/<timestamp>/`. Merge them into a single checkpoint:

```bash
uv run scripts/merge_ckpt.py relics/<timestamp>
```

Then sample the merged relic:

```bash
uv run scripts/haloscope.py relics/<timestamp>/merged.pt --prompt "The universe"
```

## Repository Hygiene

The following are local/generated artifacts and are gitignored:

- `data/`
- `runs/`
- `relics/`
- `.venv/`
- Python caches and test caches

Keep committed changes focused on source, configs, scripts, tests, and documentation.
