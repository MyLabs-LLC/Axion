# Axion 3B Model Concept

## Status

This document records a future scaling concept. It is not the active training target and must not interrupt or alter the current `macho-1b-finewebedu-20b` run.

## Hardware Assumption

The proposed eight-system subhalo contains one GPU per host:

- 3 x RTX 3090 with 24 GB VRAM
- 1 x RTX 3080 Ti with 12 GB VRAM
- 4 x RTX 3060 with 12 GB VRAM
- 132 GB aggregate VRAM

Hosts, network interfaces, and physical link speeds must be inventoried before implementation. Fast and reliable hosts should occupy the endpoint stages because those stages also own the token embedding or language-model head.

## Proposed Model

Use a narrower and deeper decoder so whole transformer layers can be assigned to the heterogeneous GPUs without splitting an individual layer.

| Field | Proposed value |
|---|---:|
| Parameters | approximately 3.0B |
| Vocabulary | 16,384 |
| Model width | 2,304 |
| Transformer layers | 46 |
| Attention heads | 18 |
| Feed-forward width | 9,216 |
| Context length | 32,768 |
| Embeddings | untied |
| Compute precision | BF16 |
| Optimizer | AdamW |
| Activation checkpointing | enabled |
| Micro batch | 1 sequence |

The exact parameter estimate under the current Axion counting convention is about 3.0B. Recalculate it with `src/axion/params.py` when a concrete configuration is added.

## Heterogeneous Pipeline Plan

| Stage | GPU class | Layers | Additional responsibility |
|---:|---|---:|---|
| 0 | RTX 3090 24 GB | 9 | token embedding |
| 1 | RTX 3060 12 GB | 4 | none |
| 2 | RTX 3060 12 GB | 4 | none |
| 3 | RTX 3090 24 GB | 7 | none |
| 4 | RTX 3060 12 GB | 4 | none |
| 5 | RTX 3080 Ti 12 GB | 5 | none |
| 6 | RTX 3060 12 GB | 4 | none |
| 7 | RTX 3090 24 GB | 9 | final norm, LM head, and loss |

This assigns all 46 layers. The two 3090 endpoint stages absorb the embedding tables and the third 3090 carries a larger middle shard. The 3080 Ti receives one more layer than a 3060, subject to measured memory headroom.

## Capacity Estimates

These are planning estimates, not validated measurements:

| Resource | Estimate |
|---|---:|
| RTX 3060 stage peak | 8-10 GB |
| RTX 3080 Ti stage peak | 9-11.5 GB |
| Middle RTX 3090 peak | 14-18 GB |
| Endpoint RTX 3090 peak | 18-21 GB |
| One distributed resumable checkpoint | about 36 GB |
| Two retained checkpoints | about 72 GB |
| Merged BF16 model | about 6 GB |
| Merged FP32 model | about 12 GB |
| Quantized 4-bit serving artifact | roughly 1.5-2 GB |

The five-layer 3080 Ti shard is the first memory-risk checkpoint. A distributed smoke run must record peak allocated and reserved memory on every stage. If that stage lacks safe headroom, move one layer to the middle 3090.

## Pipeline Requirements

The current Axion training path is fixed to three stages and cannot run this plan unchanged. Before a 3B smoke run:

1. Generalize stage construction, rank routing, process groups, evaluation, and checkpoint recovery to an arbitrary world size.
2. Add configurable host and network-interface mappings for all eight systems.
3. Replace all-forward/all-backward GPipe with a memory-bounded 1F1B schedule.
4. Support heterogeneous layer maps rather than equal layer division.
5. Add eight-stage CPU/Gloo contract tests and distributed checkpoint recovery tests.
6. Record per-stage compute time, transfer time, payload bytes, memory peaks, and idle time.

With eight stages and only two microbatches, the theoretical pipeline bubble is 77.8%. A practical implementation should target 16-32 microbatches with bounded live activations:

| Microbatches | Theoretical bubble |
|---:|---:|
| 2 | 77.8% |
| 4 | 63.6% |
| 16 | 30.4% |
| 32 | 17.9% |

## Training Targets

Expected performance after heterogeneous scheduling is approximately 2,200-3,000 tokens per second. The current GPipe schedule would likely achieve only 700-1,000 tokens per second.

| Run type | Tokens | Approximate duration after pipeline work |
|---|---:|---:|
| Systems/proving run | 20B | 77-105 days |
| Full pretraining target | 60B | 8-12 months |

The 20B run is suitable for proving the worldwide training system but would leave a 3B model undertrained. Use approximately 60B high-quality training tokens as the full pretraining target, with a separate fixed validation split.

## Serving

Merge the eight stage relics after training and serve the resulting model on a single RTX 3090. A 3B BF16 model fits comfortably on one 24 GB card, while distributed inference across slow links would add latency without solving a memory constraint. Retain distributed serving as a later fault-tolerance or world-model experiment rather than the default deployment path.

## Gate Before Work Starts

Do not begin 3B implementation until the current 1B run has completed or has reached an explicitly chosen stopping milestone. Before approving the 3B build, capture the final 1B validation curve, throughput, communication measurements, checkpoint recovery result, merged-model score, and lessons that should change this design.
