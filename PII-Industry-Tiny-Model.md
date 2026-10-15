# Axion PII and Industry Tiny Model

## Status

This document is an implementation-ready design and reproducibility runbook for
a new model. The PII/industry encoder, data builders, training commands, and
export commands described below are proposed deliverables; they are not part of
the current Axion codebase yet.

Development must remain separate from the active
`macho-1b-finewebedu-20b` decoder training run. Do not stop, modify, reuse, or
overwrite that run's processes, data, logs, or checkpoints.

The intended result is a small, non-generative model that:

- locates and classifies personally identifiable information (PII) spans;
- assigns one or more hierarchical industry labels to text;
- returns labels, character offsets, and calibrated confidence scores;
- runs quickly on a single CPU with no GPU required for inference;
- can be trained from randomly initialized weights;
- is reproducible on a machine other than the current Axion training cluster.

## Core Decision

Use a bidirectional Transformer encoder with two task heads. Do not use a
decoder-only language model for this problem. Autoregressive generation adds
latency, creates output-format failure modes, and is unnecessary for token and
document classification.

The production detector should be a hybrid system:

1. The neural model finds contextual and ambiguous PII and predicts industry
   labels.
2. Deterministic recognizers find structured identifiers such as email
   addresses, IP addresses, payment cards, and government IDs.
3. Validators apply checksums, length rules, geographic rules, and contextual
   constraints.
4. A merger resolves overlaps and applies per-label confidence thresholds.

Rules are not a substitute for the model. They are a high-recall safety layer
for identifiers with reliable structure.

## Target Model

Working name: `Axion-PII-Industry-Tiny`

| Field | Initial value |
|---|---:|
| Architecture | bidirectional Transformer encoder |
| Transformer layers | 6 |
| Hidden width | 256 |
| Attention heads | 4 |
| Head dimension | 64 |
| Feed-forward width | 1,024 |
| Vocabulary | 16,384 |
| Maximum training context | 256 tokens |
| Maximum supported context | 512 tokens |
| Position embeddings | learned absolute, length 512 |
| Activation | GELU |
| Normalization | LayerNorm |
| Embedding/MLM output | tied during pretraining |
| Dropout | 0.1 during training, disabled in inference |
| Estimated parameters | approximately 9-11 million |
| FP32 weight size | approximately 40-50 MB |
| INT8 deployment size | approximately 12-20 MB |

Standard BERT-compatible operations are intentional. Exotic attention or
custom CUDA operations can make CPU export slower, less portable, or impossible
to fuse. Optimize the measured ONNX graph before changing the architecture.

### Shared encoder and task heads

The encoder produces one vector per token plus a pooled document vector.

The PII head is a linear token classifier using BIOES labels:

- `B-TYPE`: beginning of a multi-token entity;
- `I-TYPE`: internal token;
- `E-TYPE`: final token;
- `S-TYPE`: single-token entity;
- `O`: not PII.

Start without a conditional random field (CRF). Add one only if controlled
evaluation shows enough boundary improvement to justify its CPU and export
cost.

The industry head is a hierarchical multilabel classifier. It pools each
chunk, predicts NAICS levels, and aggregates chunk logits with masked max or
log-sum-exp pooling. Multilabel output is required because a document or
organization may describe more than one line of business.

## Label Taxonomy

### Initial PII labels

Version the label file and define every label operationally. A practical first
version is:

- `PERSON_NAME`
- `EMAIL_ADDRESS`
- `PHONE_NUMBER`
- `STREET_ADDRESS`
- `LOCATION_DETAIL`
- `DATE_OF_BIRTH`
- `AGE`
- `SOCIAL_SECURITY_NUMBER`
- `GOVERNMENT_ID`
- `PASSPORT_NUMBER`
- `DRIVER_LICENSE_NUMBER`
- `TAX_ID`
- `BANK_ACCOUNT_NUMBER`
- `ROUTING_NUMBER`
- `PAYMENT_CARD_NUMBER`
- `USERNAME`
- `PASSWORD_OR_SECRET`
- `IP_ADDRESS`
- `DEVICE_IDENTIFIER`
- `BIOMETRIC_IDENTIFIER`
- `MEDICAL_RECORD_NUMBER`
- `INSURANCE_IDENTIFIER`

Do not silently change label meaning between dataset versions. Specify whether
organization names, generic geographic names, dates, and job titles count as
PII for the intended product. They are often context-dependent.

For the first version, reject overlapping gold entities during validation.
Nested PII requires a span-classification head and should be a separately
measured later feature.

### Industry labels

Use the 2022 North American Industry Classification System (NAICS) as the
canonical hierarchy unless the application requires a different jurisdiction.
Store each code with its description and parent code.

Implement industry prediction in stages:

1. Two-digit sector predictions establish a robust broad classifier.
2. Three- and four-digit predictions refine the sector.
3. Six-digit predictions are enabled only where there is enough reviewed data.
4. `UNKNOWN` or abstention is always permitted.

Never force a child code whose parent probability is below its threshold.
Evaluate both exact-code correctness and hierarchical distance from the gold
code.

## Repository Layout to Implement

Keep the feature isolated so it does not complicate Axion's distributed decoder
trainer:

```text
configs/pii_industry/
  tiny.yaml
  labels_pii.json
  labels_naics_2022.json
src/axion_pii/
  __init__.py
  config.py
  tokenizer.py
  model.py
  data.py
  rules.py
  losses.py
  metrics.py
  inference.py
scripts/pii_industry/
  prepare_corpus.py
  train_tokenizer.py
  pretrain_mlm.py
  build_synthetic_pii.py
  prepare_supervised.py
  train_multitask.py
  evaluate.py
  calibrate.py
  export_onnx.py
  benchmark_cpu.py
tests/pii_industry/
  test_offsets.py
  test_labels.py
  test_data_contract.py
  test_model_shapes.py
  test_rules.py
  test_onnx_parity.py
```

Generated inputs and outputs belong under ignored directories:

```text
data/pii_industry/
relics/pii_industry/
runs/pii_industry/
reports/pii_industry/
exports/pii_industry/
```

Never commit corpora, labeled production records, checkpoints, exported models,
logs, secrets, or local databases.

## Environment on a Separate Machine

### Minimum inference system

- x86-64 or ARM64 CPU;
- 2 GB free RAM, although the deployed model should use much less;
- Python 3.11 or 3.12;
- ONNX Runtime CPU execution provider.

### Practical training system

- Linux;
- Python 3.11 or 3.12 and `uv`;
- 32 GB system RAM;
- one NVIDIA GPU with 12 GB or more VRAM;
- 100 GB or more working storage, adjusted for corpus size;
- CUDA-compatible PyTorch for GPU training.

The model can technically train on CPU, but masked-language pretraining on
billions of tokens would be unnecessarily slow. Inference must still be tested
without CUDA installed.

### Proposed dependencies

Add a dedicated optional dependency group rather than increasing the base
Axion installation:

```toml
[project.optional-dependencies]
pii = [
  "evaluate",
  "onnx",
  "onnxruntime",
  "scikit-learn",
  "tokenizers",
  "transformers>=4.57,<6",
]
```

After the implementation is committed, a clean machine should be able to run:

```bash
git clone <AXION_REPOSITORY_URL> Axion
cd Axion
uv sync --extra dev --extra pii
uv run pytest -q tests/pii_industry
```

Pin the resolved environment with the repository's `uv.lock`. Record the Git
commit and dependency-lock hash in every training manifest.

## Data Contracts

All text must be UTF-8. PII offsets use zero-based Python character indices,
with an inclusive start and exclusive end. Preserve original text for label
alignment. Tokenizer normalization must not lowercase text, remove punctuation,
collapse whitespace, or normalize digits.

### PII JSONL

One record per line:

```json
{
  "id": "stable-example-id",
  "text": "Contact Maria at maria@example.com.",
  "entities": [
    {"start": 8, "end": 13, "label": "PERSON_NAME"},
    {"start": 17, "end": 34, "label": "EMAIL_ADDRESS"}
  ],
  "source_group": "synthetic-template-0042",
  "split": "train"
}
```

The loader must verify:

- `text[start:end]` is non-empty;
- offsets land on valid character boundaries;
- labels exist in the versioned taxonomy;
- entities do not overlap in version 1;
- duplicate IDs and duplicate text do not cross splits;
- source groups do not cross train, validation, and test splits.

### Industry JSONL

```json
{
  "id": "stable-document-id",
  "text": "We manufacture industrial pumps for municipal water systems.",
  "naics": ["333914"],
  "source_group": "organization-identifier",
  "split": "train"
}
```

Every six-digit label must resolve to its four-, three-, and two-digit parents.
The data builder should materialize those parent targets rather than deriving
them differently in each training worker.

### MLM corpus format

Use one deduplicated UTF-8 document per JSONL record with a stable ID and source
group. Keep document boundaries. Record licenses, provenance categories,
filters, hashes, and token counts in a manifest without exposing sensitive
source text.

## Data Safety

Do not collect real secrets simply because the intended model detects secrets.
Prefer synthetic identifiers and authorized, minimized, access-controlled
examples. Redact unrelated PII before a reviewer sees a record.

At minimum:

- document the lawful basis and permitted use of every source;
- encrypt sensitive labeled data at rest and in transit;
- restrict access to the minimum set of operators;
- prevent raw examples from entering logs, exception messages, MLflow, or Git;
- set retention and deletion policies;
- run memorization and canary-extraction tests before release;
- publish only synthetic examples in model cards and tests.

## Tokenizer

Train a case-sensitive 16K Unigram or byte-level tokenizer with lossless byte
fallback. The important properties are full Unicode coverage and reliable
character-offset mapping.

Tokenizer training data should contain:

- a representative sample of the unlabeled domain corpus;
- synthetic PII formats;
- URLs, email addresses, account-like strings, JSON, tables, and source code;
- multilingual text if multilingual inference is required.

Acceptance tests must round-trip difficult strings and prove token-to-character
alignment for Unicode names, combining marks, emoji, punctuation, repeated
whitespace, and malformed identifiers.

Proposed command after implementation:

```bash
uv run scripts/pii_industry/train_tokenizer.py \
  --input data/pii_industry/mlm/train.jsonl \
  --output data/pii_industry/tokenizer \
  --vocab-size 16384 \
  --byte-fallback \
  --preserve-case
```

Write a tokenizer manifest containing input hashes, sample counts, vocabulary
size, special-token IDs, normalization settings, and training seed.

## Building the Training Data

### Unlabeled pretraining corpus

Target 1-3 billion high-quality, domain-relevant tokens for the first serious
run. Deduplicate by exact hash and near-duplicate similarity. Split by source
before tokenization. Reserve a stable held-out validation set.

Quality is more important than simply reaching the token target. Include the
kinds of text the detector will actually see: correspondence, documents,
forms, support tickets, OCR, structured records, websites, business
descriptions, and mixed-format text, subject to authorization.

### Synthetic PII

Generate millions of examples, but avoid a small set of repetitive templates.
Vary:

- position, grammar, and surrounding context;
- names and address conventions across intended locales;
- whitespace, punctuation, casing, and Unicode;
- OCR substitutions and missing separators;
- HTML, Markdown, JSON, CSV, logs, and form layouts;
- valid and invalid checksums;
- strings that resemble PII but are not PII;
- adjacent entities and multiple entities in one passage.

Keep generator dictionaries and template families disjoint across splits. A
random row split allows the model to memorize templates and gives misleading
scores.

### Reviewed PII

Synthetic data alone is not sufficient. Target 50,000-200,000 carefully
reviewed real or realistically transformed examples covering the intended
deployment distribution. Double-annotate a subset, adjudicate disagreements,
and report agreement by label.

Oversample rare critical types, but report evaluation on their natural
frequency. Add hard negatives found by running preliminary models over
unlabeled data and reviewing high-confidence errors.

### Industry data

Construct labels from authorized company descriptions, filings, product text,
websites, and existing NAICS-labelled records. Split by organization so text
from one organization cannot occur in multiple splits.

Include:

- short and long descriptions;
- multi-industry organizations;
- descriptions without enough information;
- confusing sibling industries;
- non-company text and irrelevant text;
- historical labels mapped explicitly to the selected NAICS version.

Begin with high-quality two-digit supervision. Only train finer codes with
enough examples to support meaningful held-out evaluation.

## Training Stages

### Stage 0: deterministic baseline

Implement the structured PII rules and validators first. Measure them on the
same locked test set as the model. This establishes which labels actually need
contextual learning and provides hard negatives for training.

### Stage 1: masked-language pretraining

Initialize all encoder weights randomly. Train masked-language modeling with
tied input/output embeddings.

Initial hyperparameters:

| Field | Value |
|---|---:|
| Sequence length | 256 |
| Masked tokens | 15% |
| Effective batch | 32K-64K tokens |
| Optimizer | AdamW |
| Peak learning rate | 5e-4 |
| Weight decay | 0.01 |
| Warmup | 2% of steps |
| Schedule | cosine decay |
| Gradient clipping | 1.0 |
| Training precision | BF16 when supported |
| Validation cadence | every 1,000 steps |
| Checkpoint cadence | every 2,000-5,000 steps |

Train for at least 1 billion tokens before judging the architecture. Prefer
2-3 billion if validation loss and downstream probe scores continue to improve.
Save optimizer-aware resumable checkpoints and retain the best validation
checkpoint in addition to recent recovery checkpoints.

Proposed command:

```bash
uv run scripts/pii_industry/pretrain_mlm.py \
  --config configs/pii_industry/tiny.yaml \
  --train data/pii_industry/mlm/train.jsonl \
  --validation data/pii_industry/mlm/validation.jsonl \
  --tokenizer data/pii_industry/tokenizer \
  --token-target 2000000000 \
  --run-name axion-pii-industry-tiny-mlm-v1
```

### Stage 2: supervised multitask training

Attach both task heads and mix PII and industry batches. Do not require every
example to have both annotation types. Mask the loss for missing tasks.

Use this starting objective:

```text
total_loss =
    1.0 * pii_token_cross_entropy
  + 0.5 * industry_binary_cross_entropy
  + 0.1 * hierarchy_consistency_penalty
  + optional_distillation_loss
```

Tune weights from validation gradients and metrics, not dataset size alone.
Use class weighting or focal loss only after confirming that ordinary weighted
cross-entropy cannot handle rare labels.

Starting values:

| Field | Value |
|---|---:|
| Encoder learning rate | 1e-4 |
| Head learning rate | 3e-4 |
| Effective batch | 128-256 sequences |
| Epochs | 3-10, selected by validation |
| Warmup | 5% |
| Early stopping | macro metrics and critical PII recall |
| Gradient clipping | 1.0 |

Proposed command:

```bash
uv run scripts/pii_industry/train_multitask.py \
  --config configs/pii_industry/tiny.yaml \
  --encoder relics/pii_industry/mlm/best \
  --pii-train data/pii_industry/pii/train.jsonl \
  --pii-validation data/pii_industry/pii/validation.jsonl \
  --industry-train data/pii_industry/industry/train.jsonl \
  --industry-validation data/pii_industry/industry/validation.jsonl \
  --run-name axion-pii-industry-tiny-v1
```

### Stage 3: teacher distillation

Random initialization does not prohibit teacher supervision. Distillation is
the most practical way to transfer boundary behavior and difficult semantic
distinctions into a tiny model.

Use an ensemble of strong PII/NER and industry teachers to produce token and
class logits on authorized training text. Never treat teacher labels as gold.
Mix reviewed gold loss with temperature-scaled KL divergence, retain hard
negative examples, and test the student against the locked human-labelled set.

Keep teacher identity, versions, prompts, thresholds, and output hashes in the
dataset manifest. Do not require teachers at inference time.

### Stage 4: threshold calibration

Fit per-label decision thresholds on a dedicated calibration split. Critical
PII normally needs a recall-oriented threshold, while ambiguous labels may need
higher precision. Calibrate an `UNKNOWN` or abstention outcome for industry
classification.

Do not tune thresholds on the final test set.

## Chunking and Inference

PII must retain global character offsets across chunks:

1. Tokenize with offset mappings.
2. Create 256-token windows with a 32-token overlap.
3. Run the PII and industry heads for each window.
4. Convert local token spans back to original character offsets.
5. Merge duplicate spans from overlapping windows.
6. Aggregate industry logits across chunks.
7. Run deterministic PII recognizers and validators.
8. Resolve overlaps using validation-derived label priority and confidence.

Never return the original sensitive text in logs. A normal API result may look
like:

```json
{
  "entities": [
    {"start": 17, "end": 34, "label": "EMAIL_ADDRESS", "score": 0.998}
  ],
  "industries": [
    {"code": "333914", "score": 0.91}
  ],
  "model_version": "axion-pii-industry-tiny-v1"
}
```

## Evaluation

Freeze train, validation, calibration, and test manifests before serious model
comparison. Group splits by organization, source, template family, and entity
dictionary as applicable.

### PII metrics

- exact-span micro precision, recall, and F1;
- exact-span macro F1 across entity types;
- per-label precision and recall;
- boundary-only and type-only error counts;
- false negatives per million tokens for critical PII;
- performance by source type, language, corruption, and document format;
- rules-only, model-only, and combined-system results.

Token accuracy is not an acceptable primary PII metric because `O` tokens
dominate most corpora.

### Industry metrics

- micro and macro F1 at each NAICS depth;
- exact match and top-k recall;
- hierarchical precision, recall, and distance;
- calibration error and abstention coverage;
- performance for rare classes and confusing siblings;
- performance on organizations absent from training.

### Initial release gates

These are targets, not promised outcomes:

| Gate | Initial target |
|---|---:|
| Overall exact-span PII F1 | at least 0.95 |
| Critical structured PII recall, combined system | at least 0.99 |
| PII macro F1 | at least 0.90 |
| Two-digit industry macro F1 | at least 0.90 |
| INT8 PII F1 regression from FP32 | no more than 0.005 absolute |
| INT8 industry macro-F1 regression | no more than 0.005 absolute |
| ONNX versus PyTorch logit parity before quantization | within documented tolerance |
| Batch-1 CPU p95 at 128 tokens | target at most 25 ms on named hardware |

Report the exact CPU model, core count, thread settings, operating system,
ONNX Runtime version, batch size, sequence length, warmup runs, and sample count.
A latency claim without those fields is not reproducible.

Proposed evaluation command:

```bash
uv run scripts/pii_industry/evaluate.py \
  --model relics/pii_industry/multitask/best \
  --pii-test data/pii_industry/pii/test.jsonl \
  --industry-test data/pii_industry/industry/test.jsonl \
  --out reports/pii_industry/fp32-test.json
```

## ONNX and INT8 CPU Deployment

Export a batch-dynamic, sequence-dynamic FP32 ONNX model first. Check its logits
and decoded spans against PyTorch on a fixed parity set. Then create static
64-, 128-, 256-, and 512-token buckets if benchmarking shows that static shapes
improve latency enough to justify multiple artifacts.

Use ONNX Runtime graph preprocessing and Transformer optimization before
quantization. Start with dynamic INT8 quantization for Transformer weights.
Test per-channel quantization and CPU-appropriate signedness on the actual
deployment processor. If post-training INT8 misses the accuracy gate, use
quantization-aware training rather than accepting the regression silently.

Proposed export command:

```bash
uv run scripts/pii_industry/export_onnx.py \
  --model relics/pii_industry/multitask/best \
  --output exports/pii_industry/axion-pii-industry-tiny-v1 \
  --opset 17 \
  --optimize \
  --quantize-int8 \
  --validate-parity
```

The export directory should contain:

```text
model.fp32.onnx
model.int8.onnx
tokenizer.json
tokenizer_config.json
labels_pii.json
labels_naics_2022.json
thresholds.json
rules.json
config.json
manifest.json
MODEL_CARD.md
```

The manifest records source checkpoint hash, export command, Git commit,
dependency-lock hash, ONNX opset, quantization settings, artifact hashes, model
parameter count, taxonomy versions, and evaluation report hashes.

### CPU benchmark

Benchmark with batch size 1 and realistic text lengths. Include tokenizer,
chunking, neural inference, rules, and postprocessing in end-to-end latency.
Also report model-only latency so regressions can be localized.

```bash
CUDA_VISIBLE_DEVICES='' uv run scripts/pii_industry/benchmark_cpu.py \
  --model exports/pii_industry/axion-pii-industry-tiny-v1/model.int8.onnx \
  --lengths 64 128 256 512 \
  --batch-sizes 1 8 \
  --warmup 100 \
  --iterations 1000 \
  --threads 1 2 4 \
  --out reports/pii_industry/cpu-benchmark.json
```

Tune ONNX Runtime intra-op and inter-op thread counts instead of assuming all
available cores are faster. Re-run accuracy evaluation through the exported
INT8 inference path.

## Monitoring and Recovery

Every training stage should write append-only JSONL metrics and a concise
console log suitable for `tail -F`:

```bash
tail -F runs/pii_industry/<run-name>/metrics.jsonl
```

Record at least:

- optimizer step and tokens seen;
- train and validation loss;
- PII exact-span metrics by label;
- industry micro/macro and hierarchical metrics;
- learning rate, gradient norm, and throughput;
- GPU memory during training;
- checkpoint path and hash;
- best-checkpoint decision and early-stopping state.

Checkpoints must include model, optimizer, scheduler, gradient scaler where
used, random-number generator states, dataset cursor, tokenizer hash, label
taxonomy hashes, and configuration. Test interruption and deterministic resume
in an automated smoke run.

## Reproducible End-to-End Sequence

The following is the intended sequence once the proposed implementation exists:

```bash
uv sync --extra dev --extra pii

uv run pytest -q tests/pii_industry

uv run scripts/pii_industry/prepare_corpus.py \
  --config configs/pii_industry/tiny.yaml \
  --output data/pii_industry/mlm

uv run scripts/pii_industry/train_tokenizer.py \
  --input data/pii_industry/mlm/train.jsonl \
  --output data/pii_industry/tokenizer \
  --vocab-size 16384 --byte-fallback --preserve-case

uv run scripts/pii_industry/pretrain_mlm.py \
  --config configs/pii_industry/tiny.yaml \
  --train data/pii_industry/mlm/train.jsonl \
  --validation data/pii_industry/mlm/validation.jsonl \
  --tokenizer data/pii_industry/tokenizer \
  --token-target 2000000000 \
  --run-name axion-pii-industry-tiny-mlm-v1

uv run scripts/pii_industry/build_synthetic_pii.py \
  --config configs/pii_industry/tiny.yaml \
  --output data/pii_industry/pii-synthetic

uv run scripts/pii_industry/prepare_supervised.py \
  --config configs/pii_industry/tiny.yaml \
  --output data/pii_industry/supervised

uv run scripts/pii_industry/train_multitask.py \
  --config configs/pii_industry/tiny.yaml \
  --encoder relics/pii_industry/mlm/best \
  --pii-train data/pii_industry/pii/train.jsonl \
  --pii-validation data/pii_industry/pii/validation.jsonl \
  --industry-train data/pii_industry/industry/train.jsonl \
  --industry-validation data/pii_industry/industry/validation.jsonl \
  --run-name axion-pii-industry-tiny-v1

uv run scripts/pii_industry/calibrate.py \
  --model relics/pii_industry/multitask/best \
  --output relics/pii_industry/multitask/thresholds.json

uv run scripts/pii_industry/evaluate.py \
  --model relics/pii_industry/multitask/best \
  --pii-test data/pii_industry/pii/test.jsonl \
  --industry-test data/pii_industry/industry/test.jsonl \
  --out reports/pii_industry/fp32-test.json

uv run scripts/pii_industry/export_onnx.py \
  --model relics/pii_industry/multitask/best \
  --output exports/pii_industry/axion-pii-industry-tiny-v1 \
  --opset 17 --optimize --quantize-int8 --validate-parity

CUDA_VISIBLE_DEVICES='' uv run scripts/pii_industry/evaluate.py \
  --onnx exports/pii_industry/axion-pii-industry-tiny-v1/model.int8.onnx \
  --pii-test data/pii_industry/pii/test.jsonl \
  --industry-test data/pii_industry/industry/test.jsonl \
  --out reports/pii_industry/int8-test.json

CUDA_VISIBLE_DEVICES='' uv run scripts/pii_industry/benchmark_cpu.py \
  --model exports/pii_industry/axion-pii-industry-tiny-v1/model.int8.onnx \
  --lengths 64 128 256 512 --batch-sizes 1 8 \
  --warmup 100 --iterations 1000 --threads 1 2 4 \
  --out reports/pii_industry/cpu-benchmark.json
```

## Implementation Order for Another Coding System

If this document is handed to another coding agent or development system, use
this order:

1. Add the optional dependency group and package/config skeleton.
2. Implement and test label schemas, data validation, splitting, and offset
   alignment before model training.
3. Implement the tokenizer workflow and alignment tests.
4. Implement the encoder, MLM head, PII head, and hierarchical industry head.
5. Add tiny synthetic fixtures and CPU-only unit/smoke tests.
6. Implement resumable MLM training and verify a 20-step smoke run.
7. Implement rules and deterministic validators with regression tests.
8. Implement multitask training, metrics, calibration, and grouped evaluation.
9. Implement ONNX export, graph optimization, INT8 conversion, and numerical
   parity tests.
10. Implement end-to-end CPU benchmarking and artifact manifests.
11. Run the complete test suite with `uv run pytest -q`.
12. Run a small end-to-end proving experiment before preparing billion-token
    corpora or long GPU jobs.

No implementation step may alter the active Axion decoder run or its artifacts.

## Acceptance Criteria

The implementation is complete only when:

- a clean clone installs from the lockfile;
- all data contracts reject malformed labels and split leakage;
- Unicode and token-offset tests pass;
- a CPU-only smoke can train, save, resume, evaluate, and export;
- the full model can resume from an optimizer-aware checkpoint;
- PyTorch and FP32 ONNX predictions pass parity tests;
- INT8 regression stays inside the declared metric gates;
- the hybrid rules-plus-model path is evaluated separately and together;
- the benchmark reports reproducible p50, p95, p99, and throughput;
- no raw PII appears in logs or committed fixtures;
- artifacts include manifests, hashes, taxonomies, thresholds, and a model card;
- generated datasets, checkpoints, exports, and logs remain ignored by Git.

## Expected Development Strategy

First prove the entire pipeline using a four-layer, 128-wide smoke model and a
small synthetic dataset. That model is disposable and exists to expose data,
resume, metric, export, and inference bugs quickly.

Then train the proposed six-layer, 256-wide model. If it misses accuracy goals:

1. improve reviewed data, hard negatives, and taxonomy definitions;
2. improve teacher distillation and threshold calibration;
3. increase pretraining quality or tokens;
4. test an eight-layer, 320- or 384-wide model only after the earlier steps.

Increasing parameters should not be the first response to a weak label policy
or leaky evaluation set. Maintain at least two deployment candidates on the
accuracy/latency Pareto frontier rather than declaring the largest candidate
the winner automatically.

## References

- ModernBERT, a modern bidirectional encoder design:
  <https://arxiv.org/abs/2412.13663>
- GLiNER, compact bidirectional entity recognition:
  <https://arxiv.org/abs/2311.08526>
- Official 2022 NAICS manual:
  <https://www.census.gov/naics/reference_files_tools/2022_NAICS_Manual.pdf>
- ONNX Runtime quantization guidance:
  <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- ONNX Runtime Transformer optimization guidance:
  <https://onnxruntime.ai/docs/performance/transformers-optimization.html>
