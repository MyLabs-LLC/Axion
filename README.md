# Axion

Axion is a language-model training scaffold for proving distributed pretraining on consumer GPUs. The current scale test is a 234.9M-parameter decoder-only transformer split across three RTX 3090 hosts: `oblivion`, `oblivion1`, and `oblivion2`.

The project uses a dark-sector naming scheme in logs and docs, but the core ML modules use plain names:

- `src/axion/model.py`: transformer model
- `src/axion/data.py`: validated FineWeb-Edu token memmap reader
- `src/axion/train_pipeline.py`: resumable three-rank GPipe trainer
- `scripts/run_week_training.py`: MLflow-backed long-run supervisor
- `scripts/haloscope.py`: checkpoint sampler
- `scripts/rotation_curve.py`: run metrics summary

See [CLAUDE.md](CLAUDE.md) for the architecture, milestone definitions, and Subhalo-0 runbook.

## Status

The 250M run uses:

- 234,914,816 parameters (`d_model=1024`, 16 layers, 16 heads, 16K tokenizer)
- 5.0B FineWeb-Edu training tokens, approximately 21.3 tokens per parameter
- a separate 10M-token held-out validation tail
- three-stage GPipe with FP32 master weights, BF16 fused CUDA compute, and global gradient clipping
- periodic optimizer-aware checkpoints and validation CE/perplexity logged to MLflow

The FineWeb-Edu corpus has an atomic completion manifest. Interrupted preallocated files are rejected unless their saved prefix is deterministically reproduced and validated during resume.

## Requirements

- Python 3.11+
- `uv`
- CUDA-capable PyTorch
- Three RTX 3090-class hosts with passwordless SSH from `oblivion`

Install and test:

```bash
uv sync --extra dev
uv run pytest -q
```

## Data

Train the 16K SentencePiece tokenizer:

```bash
uv run scripts/train_tokenizer.py --vocab-size 16384 \
  --model-prefix data/tokenizer/axion16k
```

Build 5.0B training tokens plus the held-out validation tail:

```bash
uv run scripts/prepare_data.py \
  --tokens 5010000000 \
  --train-tokens 5000000000 \
  --val-tokens 10000000 \
  --out data/fineweb_edu_5B_axion16k.bin \
  --tokenizer data/tokenizer/axion16k.model \
  --meta data/meta_250m_5b_axion16k.json
```

Resume an interrupted corpus only from a previously flushed progress manifest. A manually supplied boundary must first be verified against the file and preparation log:

```bash
uv run scripts/prepare_data.py \
  --tokens 5010000000 --train-tokens 5000000000 --val-tokens 10000000 \
  --out data/fineweb_edu_5B_axion16k.bin \
  --tokenizer data/tokenizer/axion16k.model \
  --meta data/meta_250m_5b_axion16k.json \
  --resume --resume-tokens 1114000000
```

## 250M Distributed Run

The supervisor validates the corpus, synchronizes code/data, launches one stage per node, captures the architecture, streams metrics to MLflow, retains the latest resumable checkpoints, and scores the final merged model:

```bash
uv run scripts/run_week_training.py \
  --config configs/axion_250m.yaml \
  --experiment-name axion-250m-subhalo \
  --run-kind macho-250m-finewebedu-5b \
  --milestone 250m_full_pretrain \
  --token-target 5000000000 \
  --checkpoint-every 5000 \
  --eval-every 1000 \
  --val-batches 4 \
  --prepare-data --sync-data
```

Resume model training under the same run name after all three stage checkpoints are present:

```bash
uv run scripts/run_week_training.py \
  --config configs/axion_250m.yaml \
  --run-name <existing-run-name> \
  --token-target 5000000000 \
  --resume-training --sync-data
```

Each long run writes:

- `experiments/<run>/architecture.json` and `architecture.md`
- `runs/<run>/metrics.jsonl`
- `relics/<run>/stage*-step*.pt` with model and optimizer state
- `experiments/<run>/final_score.json`

## MLflow

Start the local UI from the repository root:

```bash
uv run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000
```

Open `http://oblivion:5000` from a machine that can reach the host. The 250M run appears in the `axion-250m-subhalo` experiment and reports training loss, held-out validation loss/perplexity, throughput, communication time, gradient norm, and GPU memory.

## Smaller Smokes

Single-GPU 1M baseline:

```bash
uv run -m axion.train_single --config configs/axion_1m.yaml --steps 2
```

Three-rank local CPU smoke:

```bash
CUDA_VISIBLE_DEVICES='' uv run torchrun --standalone --nproc-per-node=3 \
  -m axion.train_pipeline --config configs/axion_1m.yaml --steps 1
```

Three-node launch:

```bash
./scripts/launch_m1.sh configs/axion_1m.yaml
```

Merge and test a completed distributed checkpoint:

```bash
uv run scripts/merge_ckpt.py relics/<run> --config configs/axion_250m.yaml
uv run scripts/evaluate_relic.py relics/<run>/merged.pt \
  --config configs/axion_250m.yaml --batches 4
uv run scripts/haloscope.py relics/<run>/merged.pt --prompt "The universe"
```

Run the full zero-shot external benchmark suite with the pinned harness version:

```bash
uv run --with lm-eval==0.4.12 scripts/evaluate_benchmarks.py \
  --relic relics/<run>/merged.pt \
  --config configs/axion_250m.yaml \
  --tokenizer data/tokenizer/axion16k.model \
  --tasks hellaswag arc_easy piqa winogrande lambada_openai wikitext \
  --batch-size 32 \
  --out reports/axion_250m_benchmarks.json
```

Use `--limit 2` for a six-task smoke before a full evaluation.

Export a retained distributed checkpoint as a Hugging Face model:

```bash
uv run --extra export scripts/export_hf.py \
  --run-name macho-1b-finewebedu-20b \
  --step 36000 \
  --config configs/axion_1b.yaml \
  --tokenizer data/tokenizer/axion16k.model
```

The exporter collects the three stage shards when needed, verifies the
checkpoint contract, and writes BF16 SafeTensors plus custom Transformers and
tokenizer files under `hf_export/`. The active training process is not changed.
See the generated model card for Transformers, vLLM, and Hugging Face upload
commands.

## Repository Hygiene

Data, checkpoints, run logs, MLflow state, reports, and experiment captures are generated artifacts and are ignored by git. Committed changes should remain limited to source, configs, scripts, tests, and documentation.
