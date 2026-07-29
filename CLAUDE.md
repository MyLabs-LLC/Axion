# AXION — a World Language Model (WLM)

> *Weakly interacting, massively parallel.*
>
> Training and serving a language model across geographically distributed consumer GPUs
> (RTX 3090-class, 24 GB) over ordinary internet links — no datacenter, no NVLink, no
> InfiniBand. Dark matter is five-sixths of the matter in the universe: it never collapses
> into a center, and its gravity is what holds every galaxy together. Datacenters are the
> stars. The world's idle GPUs are the dark matter. The axion was proposed to solve the
> strong CP problem; this one is built to solve the strong centralization problem.

## Glossary — the dark sector (use these terms consistently in code, logs, and docs)

| Term | The physics | Here |
|------|-------------|------|
| **Axion** | Ultralight particle, leading dark-matter candidate; invented to fix the strong CP problem | The model itself. Releases: axion-1m, axion-25m, ... (params are our µeV) |
| **the Halo** | The invisible mass structure that holds every galaxy together | The global network of all participating nodes |
| **Subhalo** | Gravitationally bound substructure inside a host halo | One LAN cluster holding ONE full replica, sharded by layers. Our 3 boxes = **Subhalo-0** |
| **macho** | MAssive Compact Halo Object | The node daemon — one GPU box in a closet. Hostnames: macho0, macho1, macho2 |
| **Lensing** | Invisible mass revealing itself by bending light | The WAN training protocol (DiLoCo-family). One sync round = a **lensing event** |
| **the Lens** | The mass doing the bending | The aggregator/coordinator role for a lensing event (rotates; any macho can serve) |
| **dark photon** | Hypothesized force carrier of the dark sector | The transport layer (`dark_photon.py`). Every byte between machos flows through it |
| **the Portal** | The coupling between the dark sector and the Standard Model (e.g. the *axion portal*) | The serving gateway — where the model couples to the visible world (OpenAI-compatible API) |
| **haloscope** | Resonant cavity converting axions → photons (ADMX, Univ. of Washington) | The sampling tool: point it at a relic, photons (tokens) come out (`haloscope.py`) |
| **relic** | Relic abundance — the dark matter left over from the early universe | A checkpoint (`relics/`). relic-0 = **"PQ"**, the initialized-but-untrained state |
| **rotation curve** | Rubin's flat galaxy rotation curves — the evidence for invisible mass | The run report proving invisible work happened (`rotation_curve.py`) |
| **standard candles** | Type Ia supernovae — fixed-brightness references that revealed dark energy | The fixed eval/benchmark suite (never changes mid-experiment) |
| **Hubble tension** | Disagreement between measurements of the cosmic expansion rate | Drift between subhalo replicas, measured at every lensing event (logged metric) |
| **Big Rip** | Runaway dark-energy expansion tearing everything apart | The divergence guard: loss/grad blowup → halt + alert |
| **Bullet Cluster** | Colliding clusters whose dark matter sailed straight through — proof it survives collisions | Milestone M4: chaos testing. Kill machos mid-run; the halo must survive |
| **misalignment** | The mechanism that set the axion field's initial value in the early universe | Weight initialization (`misalignment_init` in model.py) |
| **the CMB** | The cosmic microwave background — the primordial signal everything formed from | The tokenized training corpus (FineWeb-Edu memmap in `data/`) |
| **Quintessence** | Dynamical dark-energy field — the fifth element | The endgame: a distributed MoE larger than any single subhalo can hold |

**Naming policy:** the physics lives in services, milestones, and prose. Core ML files keep
boring, greppable names (`model.py`, `data.py`, `schedule.py`). Flavor never beats clarity.

---

## STATUS  <!-- Claude Code: keep this block updated after every work session -->

```
Current milestone : 1.275B/32K corpus preparation active (macho-1b-finewebedu-20b)
Naming            : v2 — dark-sector theme locked (AXION)
Next action       : validate and sync the 20.02B-token corpus, then automatically launch the 305,176-step pretrain
Blockers          : no active 1B blocker; original M1 parity and 100M-token acceptance remain validation debt
```

## How Claude Code should work in this repo

1. Read this whole file before writing code. Work strictly milestone by milestone (M0 → M1 → ...).
2. Before implementing a milestone, post a short plan (files to create, interfaces) and confirm.
3. A milestone is DONE only when its acceptance criteria pass AND `pytest -q` is green AND a
   `runs/<timestamp>/metrics.jsonl` exists proving the numbers. Update STATUS above.
4. Never touch two milestones in one change set. Never add a dependency without asking.
5. If a design decision here conflicts with reality on the machines, stop and surface it —
   do not silently work around it.

---

## 1. Mission and non-goals

**Mission:** prove that a competent LM can be pretrained and served across independent GPU
owners over the public internet, where no single node can hold or see the whole system state,
and machos may join, leave, or fail at any time.

**Hard constraint:** everything must run on a node with 1× RTX 3090 (24 GB, Ampere sm_86),
8 CPU cores, 32 GB RAM. If a design needs more than that per node, the design is wrong.

**Non-goals (for now):** SOTA benchmark quality, multimodality, RLHF, Kubernetes,
token/blockchain incentives. Those come after the physics works.

## 2. The physics (why the architecture looks like this)

| Link                | Usable bandwidth      | Latency          |
|---------------------|-----------------------|------------------|
| NVLink 4            | ~600 GB/s             | microseconds     |
| PCIe 4.0 x16        | ~32 GB/s              | microseconds     |
| 1 GbE LAN (ours)    | ~0.11 GB/s            | ~0.2 ms          |
| Home internet (WAN) | 0.003–0.06 GB/s       | 20–300 ms, jitter, loss |

Consequences — memorize these, they justify every design choice below:

1. **Tensor parallelism is impossible here at any tier** (needs NVLink-class links, all-to-all per layer).
2. **Pipeline parallelism works on LAN, dies on WAN.** Activations are small and the schedule
   tolerates ~0.2 ms hops, but 100 ms hops × micro-batches destroy it.
3. **Per-step data-parallel all-reduce dies on WAN** (payload = full model size, every step).
4. The only pattern that survives the open internet is **rare, compressed synchronization**:
   many local optimizer steps, then exchange a quantized weight-delta. This is the
   local-SGD / DiLoCo family, proven at 10B scale across continents (INTELLECT-1).
   We call it **Lensing** — the machos are weakly interacting by design.

## 3. Architecture: two tiers

```
                    THE HALO (WAN: high latency, unreliable, untrusted)
   ┌───────────────────────────── lensing events ─────────────────────────────┐
   │              int8 pseudo-gradients, every H≈500 local steps              │
   │                                                                          │
┌──┴─────────── Subhalo-0 (our LAN) ────────────┐   ┌── Subhalo-1 ──┐   ┌── Subhalo-N ──┐
│  macho0 ─1GbE─> macho1 ─1GbE─> macho2         │   │  (someone     │   │  (someone     │
│  stage 0        stage 1        stage 2        │   │  else's LAN)  │   │  else's LAN)  │
│  emb+L0-1       L2-3           L4-5+head+loss │   └───────────────┘   └───────────────┘
│  <──────── activations / grads ────────>      │
│  pipeline parallel (GPipe→1F1B), dark photon  │
└───────────────────────────────────────────────┘
```

- **Tier 1 — inside a Subhalo (LAN): pipeline parallelism.** The model is split by layers
  across the subhalo's GPUs. Activations flow forward, activation-gradients flow backward,
  micro-batching hides bubbles. A subhalo holds exactly one full replica. Our 3 machos = Subhalo-0.
- **Tier 2 — between Subhalos (WAN): Lensing.** Each subhalo runs H inner AdamW steps on its
  own data shard, then a **lensing event**: transmit Δ = θ_before − θ_after (int8 + fp32
  scale per tensor), average, apply an **outer Nesterov** step (lr 0.7, momentum 0.9 — DiLoCo
  defaults), broadcast θ. Communication drops by ~H×, and latency stops mattering. The Lens
  (aggregator role) rotates — any macho can serve.
- **Serving tier (later): the Portal.** Petals-style — layers hosted by volunteer machos,
  sessions routed through chains with redundant candidates per stage. The haloscope samples
  from relics locally; the Portal serves the halo to the visible world.
- **Endgame — Quintessence:** distributed MoE, experts pinned to their owners, learned routing
  under latency (Hivemind/SWARM direction). Research backlog, not now.

**Prior art we build on — read before implementing (do not reinvent, verify against):**
DiLoCo (arXiv 2311.08105) · OpenDiLoCo (2407.07852) · INTELLECT-1 technical report
(Prime Intellect, 2024 — 10B across 3 continents) · GPipe (1811.06965) ·
PipeDream/1F1B (1806.03377) · Petals (2209.01188) · SWARM parallelism (2301.11913) ·
Hivemind / learning@home (2002.04013) · DeMo decoupled momentum (2411.19870).

## 4. Hardware inventory — Subhalo-0 (the testbed)

| Node   | Hostname | IP (static)        | GPU           | Role in M1                |
|--------|----------|--------------------|---------------|---------------------------|
| macho0 | oblivion  | 192.168.77.232 | RTX 3090 24GB | stage 0 + torchrun master |
| macho1 | oblivion1 | 192.168.77.217 | RTX 3090 24GB | stage 1                   |
| macho2 | oblivion2 | 192.168.77.156 | RTX 3090 24GB | stage 2                   |

Each macho: 8 cores, 32 GB RAM, 1 GbE. Fill in TODOs before M1.

**Macho prerequisites (scripts/setup_node.sh must verify all of these):**
- Same NVIDIA driver major version on all machos (≥ 550), CUDA visible, `nvidia-smi` clean.
- Static IPs, passwordless SSH macho0→macho1,macho2 (for launch scripts), `chrony` time sync.
- TCP open between machos on 29500–29600 (torchrun rendezvous + NCCL sockets). On a home LAN
  with no inter-host firewall this is usually already true; verify anyway.
- `iperf3` between every pair sustains ≥ 900 Mbit/s. Record the numbers in the run log.

## 5. Repository layout

```
axion/
  CLAUDE.md                  # this file
  pyproject.toml             # uv-managed; pinned deps
  configs/
    axion_1m.yaml            # Milestone model (spec below)
    axion_25m.yaml           # next rung of the expansion (added at M2+)
  src/axion/
    model.py                 # GPT: RMSNorm, RoPE, causal attn, GELU MLP; misalignment_init; stage-splittable
    stages.py                # layer→stage assignment, per-stage module construction
    schedule.py              # GPipe schedule (M1), 1F1B (M1.5)
    dark_photon.py           # transport: p2p isend/irecv wrappers; byte + time accounting on EVERY transfer
    data.py                  # uint16 memmap over the CMB; deterministic (step, microbatch)→offset
    train_single.py          # M0 baseline
    train_pipeline.py        # M1 entry point (torchrun)
    lensing.py               # M2: inner/outer loop, int8 pseudo-gradients, lensing events, Hubble tension
    metrics.py               # jsonl logger + rich console; Big Rip divergence guard
  scripts/
    setup_node.sh            # env + macho prerequisite checks
    train_tokenizer.py       # sentencepiece BPE 4096 on FineWeb-Edu sample
    prepare_data.py          # stream FineWeb-Edu → tokenize → data/fineweb_edu_100M.bin (the CMB)
    launch_m1.sh             # ssh-driven torchrun launch across macho0..macho2
    merge_ckpt.py            # reassemble stage shards → single-GPU relic
    haloscope.py             # sample from a merged relic (axions → photons)
    netem.sh                 # WAN simulator on/off (profiles: home, bad)
    rotation_curve.py        # render a run's metrics.jsonl into a summary
  tests/
    test_model.py            # shapes, param count, first-step loss ≈ ln(vocab)
    test_schedule.py         # GPipe ordering + gradient correctness on a toy 3-stage model (CPU/gloo)
    test_parity.py           # THE DIRECT-DETECTION GATE: pipeline == single-process, fp32, ≤1e-4 rel
    test_quant.py            # int8 pseudo-gradient roundtrip error bounds
  data/                      # gitignored; the CMB + tokenizer, rsynced to all machos
  relics/                    # gitignored; checkpoints (relic-0 "PQ", per-stage shards, merged)
  runs/                      # gitignored; one dir per run with metrics.jsonl
```

## 6. Environment

- Python 3.11, managed with **uv**. `torch >= 2.5` (cu124 wheel; Ampere sm_86 fully supported),
  `numpy`, `sentencepiece`, `datasets`, `pytest`, `rich`, `pyyaml`. Nothing else without asking.
- **Forbidden in M0–M3:** DeepSpeed, Megatron, FairScale, Accelerate, torch.distributed.rpc,
  Ray. We are building the understanding, not importing it. (A PiPPy comparison spike is
  allowed later, in a branch.)
- Precision: bf16 autocast for real training; **fp32 + `torch.use_deterministic_algorithms(True)`
  for all parity tests**. Determinism switches centralized in one helper used by trainers and tests.
- 8 cores / 32 GB RAM ⇒ tokenization happens offline (memmap), dataloader is trivial reads,
  ≤ 2 workers.

## 7. The model — `axion-1m`

Decoder-only pre-norm transformer. RMSNorm, RoPE (theta 10000), GELU MLP, no biases,
no dropout, **untied** input embedding and LM head (tying across pipeline stages requires
cross-stage gradient sync for the shared weight — not worth it at this scale).
Init = `misalignment_init`: normal std 0.02, residual-out projections scaled by 1/√(2·n_layers).

```yaml
# configs/axion_1m.yaml
model:
  vocab_size: 4096        # sentencepiece BPE, byte_fallback=true (no OOV)
  d_model: 128
  n_layers: 6
  n_heads: 4              # head_dim 32
  d_ff: 512
  seq_len: 512
  rope_theta: 10000
  tie_embeddings: false
train:
  micro_batch: 8
  n_microbatches: 16      # global batch = 128 seqs = 65,536 tokens/step
  optimizer: adamw
  betas: [0.9, 0.95]
  lr: 3.0e-3
  min_lr: 3.0e-4
  warmup_steps: 100
  lr_schedule: cosine
  weight_decay: 0.1
  grad_clip: 1.0
  precision: bf16
  total_tokens: 100_000_000     # ≈ 1,526 steps
  seed: 1337
```

**Parameter math (assert this in test_model.py):**
- Per layer: attn 4·d² = 65,536 + MLP 2·d·d_ff = 131,072 + norms ≈ 512 → ≈ 196.9k
- 6 layers → **≈ 1.18 M non-embedding params — this is "the 1M model"**
- Embeddings (untied): 2 × 4096×128 = 1.05 M → total ≈ 2.23 M
- Sanity: loss at step 0 must be ≈ ln(4096) = **8.32 ± 0.05**.

**Stage split for Subhalo-0 (3 stages):**

| Stage | Macho  | Modules                                   | Params  |
|-------|--------|--------------------------------------------|---------|
| 0     | macho0 | token embedding + layers 0–1              | ~0.92 M |
| 1     | macho1 | layers 2–3                                | ~0.39 M |
| 2     | macho2 | layers 4–5 + final norm + LM head + loss  | ~0.92 M |

Stage boundaries carry a `(micro_batch=8, seq=512, d=128)` bf16 tensor = **1.0 MiB** per hop
per direction. Per optimizer step per link: 16 micro × (fwd + bwd) × 1 MiB = **32 MiB** →
~0.3 s serialized on 1 GbE, partially overlapped by the schedule. Expected M1 throughput:
**~100–200k tokens/s (comms-bound — this is expected and fine)**. A single 3090 will beat it
at this size; see "Expansion history" for where the pipeline starts winning.

## 8. Data — FineWeb-Edu (the CMB)

The primordial signal everything downstream forms from.
- Source: HuggingFace `HuggingFaceFW/fineweb-edu`, config `sample-10BT`, `streaming=True`.
- `scripts/train_tokenizer.py`: sentencepiece BPE, vocab 4096, `byte_fallback=true`,
  trained on a ~500 MB streamed sample → `data/tokenizer/axion4k.model`.
- `scripts/prepare_data.py`: stream documents → tokenize → append EOS between docs →
  write `data/fineweb_edu_100M.bin` as **uint16 memmap** (100 M train tokens + 2 M held-out
  val tokens at the tail; ~204 MB total) + `data/meta.json` (token counts, tokenizer hash).
- **rsync the CMB + tokenizer to all three machos.** During training there is NO data-plane
  traffic: stage 0 memmaps inputs `x = tok[o : o+512]`, stage 2 memmaps labels
  `y = tok[o+1 : o+513]` from the same file, using the same deterministic
  `(seed, step, microbatch) → offset` function in `data.py`. Labels are never sent over the wire.

## 9. Milestones

### M0 — "Misalignment": single-GPU baseline (1 macho)
Purpose: correctness reference + the denominator for every later speed claim. The
misalignment mechanism set the axion field's initial value; M0 sets ours.
- Implement `model.py`, `data.py`, `metrics.py`, `train_single.py`.
- Save **relic-0 ("PQ")** — the initialized, untrained model — before step 0.
- Train `axion-1m` for the full 100 M tokens on one 3090 (expect minutes, not hours;
  tiny models are kernel-launch-bound — record whatever tokens/s you get).
- **Done when:** step-0 loss ≈ 8.32; smooth decreasing loss; val CE evaluated every 100 steps
  on the fixed holdout; relic save + resume works bit-exact; `runs/*/metrics.jsonl` written;
  `haloscope.py` produces recognizably English babble from the final relic.

### M1 — "Subhalo-0": 3-stage pipeline across the LAN  ← the current goal
Purpose: the model physically split across all three machos, trained on FineWeb-Edu.
- Implement `stages.py`, `dark_photon.py`, `schedule.py` (GPipe first: all forwards, then all
  backwards, per-stage gradient accumulation, single optimizer step per stage per step).
- Transport: `torch.distributed` NCCL backend over TCP for GPU p2p (`dist.isend/irecv`
  via `batch_isend_irecv`), wrapped by dark_photon; a second **gloo** group for control
  scalars (loss → rank 0 for logging, abort signals). No RPC.
- Launch (from macho0, via `scripts/launch_m1.sh` over SSH):
  ```bash
  # per macho i ∈ {0,1,2}
  NCCL_SOCKET_IFNAME=<iface> GLOO_SOCKET_IFNAME=<iface> NCCL_IB_DISABLE=1 \
  torchrun --nnodes=3 --nproc-per-node=1 --node-rank=$i \
           --master-addr=192.168.1.101 --master-port=29500 \
           -m axion.train_pipeline --config configs/axion_1m.yaml
  ```
- Checkpointing: each stage saves its own shard + manifest under `relics/<run>/`;
  `scripts/merge_ckpt.py` reassembles a single-GPU relic for the haloscope and evals.
- **Done when:**
  1. **The direct-detection gate (the important one):** fp32 + deterministic algs, same
     seed — pipeline loss and per-parameter gradients match `train_single.py` to ≤ 1e-4
     relative for 20 steps.
  2. Full 100 M-token run whose loss curve overlays M0's (same global batch/hyperparams) within noise.
  3. Rotation curve report: tokens/s, MB/step per link, bubble fraction
     (GPipe theoretical: (S−1)/(M+S−1) = 2/18 ≈ 11%), per-stage compute ms, GPU mem.
  4. English babble from the merged relic via the haloscope.
- Expectation to hold, not fight: M1 is SLOWER than M0 at this scale. We are buying verified
  plumbing and instrumentation, not speed. Do not "optimize" by putting stages on one GPU.
- **M1.5 (optional):** swap GPipe → 1F1B in `schedule.py` behind a config flag; show identical
  loss and lower peak memory.

### M2 — "First Lensing": the WAN algorithm, first on LAN, then under simulated WAN
Purpose: the algorithm that makes "world" possible, validated without leaving the room.
- Reconfigure: each macho runs a FULL `axion-1m` replica (no pipeline), disjoint data shards.
- Inner loop: H = 500 AdamW steps. **Lensing event:** each macho computes
  Δ = θ_event_start − θ_now, int8-quantizes per tensor (fp32 scale), `all_gather` payloads
  over gloo, dequantize, average, apply **outer Nesterov** (lr 0.7, momentum 0.9), broadcast θ.
  (all_gather + local average rather than all_reduce so int8 works on any backend.)
- Log **hubble_tension** at every event: dispersion across the machos' pseudo-gradients
  (mean pairwise cosine distance + relative norm spread). This is the number that tells you
  whether H is too large. Watch it before touching anything else.
- Then apply `scripts/netem.sh home` (80 ms ± 20 ms delay, 0.5% loss, 100 Mbit) and rerun.
- **Done when:** final val CE within ~5% of an M0 run at equal TOTAL tokens; lensing payload
  ≈ 2.3 MB/macho (int8 of 2.23 M params); wall-clock overhead of lensing < 5%; the netem
  run's throughput within 10% of the clean run — demonstrating latency-immunity.

### M3 — "Structure Formation": pipelines inside subhalos × lensing between them
Hierarchical structure formation — small halos merging into large — is literally how the
cosmic web assembled, and it is the full AXION architecture.
- Subhalo = pipeline group; a lensing event exchanges per-stage pseudo-gradients between
  corresponding stages of different subhalos. With only one physical subhalo, simulate two:
  1-GPU "degenerate subhalos" or a netem-partitioned LAN.

### M4 — "Bullet Cluster": elasticity and failure
The Bullet Cluster's dark matter sailed straight through a cluster collision. So must training.
- Heartbeats; a macho dying mid-event must not corrupt training (drop its contribution,
  renormalize); relic every lensing event; a new macho joins at an event boundary by
  receiving θ. Kill -9 a macho during a run and show recovery.

### M5 — "The Portal": serving
- Petals-style: stage hosting registry, session routing with per-stage fallback candidates,
  KV-cache pinned at hosts, OpenAI-compatible endpoint in front — the coupling between the
  dark sector and the visible world. The haloscope goes interactive.

### Endgame — "Quintessence"
- Distributed MoE: experts pinned to their owners, latency-aware learned routing
  (Hivemind/SWARM direction). A model no single subhalo can hold.

### Expansion history (scaling ladder — where each mechanism starts paying rent)

| Rung       | Params (non-emb) | Config sketch              | Why this rung exists                         |
|------------|------------------|----------------------------|----------------------------------------------|
| axion-1m   | 1.2 M            | d128 L6 v4096              | plumbing + parity (M0–M3)                    |
| axion-25m  | 25 M             | d512 L8 v16384             | real learning curves; lensing at scale       |
| axion-124m | 124 M            | GPT-2-small shape, v32768  | 1 replica per 3090; pure lensing across subhalos (int8 event ≈ 124 MB — ~10 s on 100 Mbit, every ~30 min) |
| axion-1b+  | 1.3 B+           | d2048 L24                  | training no longer fits one 3090 comfortably → the 3-stage pipeline finally WINS; M3 hybrid becomes mandatory |

## 10. Comms conventions (the dark photon)

- One `dist.init_process_group("nccl")` world for tensors + one gloo group for control.
- Every transfer goes through `dark_photon.py`, which records bytes and wall time — no raw
  `dist.send/recv` calls anywhere else in the codebase.
- All activation/grad payloads bf16. Fixed tags per (link, direction, microbatch).
- Env per macho (checked at startup, hard-fail if missing): `NCCL_SOCKET_IFNAME`,
  `GLOO_SOCKET_IFNAME`, `NCCL_IB_DISABLE=1`. `NCCL_DEBUG=WARN` default, `INFO` when debugging.

## 11. Instrumentation (non-negotiable)

`runs/<ts>/metrics.jsonl`, one line per step: step, loss, lr, tokens_per_s, bytes_tx/rx per
link, bubble_frac, per-stage fwd/bwd ms, gpu_mem_mb; from M2: lensing_ms, lensing_bytes,
inner_step, **hubble_tension**. The **Big Rip guard** in `metrics.py` halts the run and
alerts when loss exceeds 3× its trailing-window median or a grad norm goes NaN (thresholds
in config). `scripts/rotation_curve.py` turns a run dir into a human-readable summary.
**If it isn't on the rotation curve, it didn't happen.**

## 12. WAN simulator — `scripts/netem.sh`

```bash
IF=<iface>
# profile: home
sudo tc qdisc add dev $IF root netem delay 80ms 20ms distribution normal loss 0.5% rate 100mbit
# profile: bad
sudo tc qdisc add dev $IF root netem delay 200ms 50ms distribution normal loss 2% rate 25mbit
# off
sudo tc qdisc del dev $IF root
```
Warning: applies to ALL traffic on the interface, including your SSH session — expect lag
while it's on. Always `netem.sh off` after tests (script must be idempotent).

## 13. Testing

- `pytest -q` green is part of every milestone's definition of done.
- `test_schedule.py` and `test_parity.py` run multi-process on ONE machine (gloo, CPU,
  fp32, deterministic) so CI needs no cluster: `torchrun --nproc-per-node=3 -m pytest ...`
  or spawn via `torch.multiprocessing`.
- `test_parity.py` is **the direct-detection gate** — the crown jewel of this repo.
  Nothing merges if it fails, ever.
- `standard_candles/` (added at M2): a fixed val-CE + babble-quality eval set that never
  changes mid-experiment, so every rung of the expansion is measured against the same ruler.

## 14. Coding conventions & guardrails

- Pure PyTorch + torch.distributed. Files ≤ ~400 lines. Type hints. No silent excepts.
- Config = YAML → dataclass; no argparse forests. Seeds set everywhere, logged in metrics.
- Determinism switches centralized in one helper used by both trainers and tests.
- Physics names for services and concepts; boring names for core ML files. Flavor never
  beats greppability.
- Commit style: `M1: <what>`. One milestone per change set.

## 15. Runbook

```bash
# once per macho
./scripts/setup_node.sh

# once, on macho0 (then rsync the CMB to macho1, macho2)
uv run scripts/train_tokenizer.py
uv run scripts/prepare_data.py --tokens 102_000_000
rsync -av data/ macho1:axion/data/ && rsync -av data/ macho2:axion/data/

# M0 (any macho)
uv run -m axion.train_single --config configs/axion_1m.yaml

# M1 (from macho0)
./scripts/launch_m1.sh configs/axion_1m.yaml

# results
uv run scripts/rotation_curve.py runs/<ts>
uv run scripts/merge_ckpt.py relics/<run>
uv run scripts/haloscope.py relics/<run>/merged.pt
```

## 16. Research backlog (post-M5 / open questions)

- Compression beyond int8: DeMo-style DCT momentum, top-k + error feedback.
- Async lensing (stale events, straggler tolerance) vs synchronous rounds.
- Heterogeneous subhalos: profile-based layer balancing when GPUs differ.
- **Trustless aggregation** — outlier/Byzantine rejection of contributed pseudo-gradients.
  Mandatory before opening the halo to strangers.
- Quintessence: experts pinned to owners, latency-aware routing (Hivemind/SWARM direction).
- Data governance: global dedup and shard assignment across untrusted contributors.

## 17. References

- DiLoCo: Distributed Low-Communication Training — arXiv:2311.08105
- OpenDiLoCo — arXiv:2407.07852 · INTELLECT-1 technical report — primeintellect.ai (10B, 3 continents)
- GPipe — arXiv:1811.06965 · PipeDream (1F1B) — arXiv:1806.03377
- Petals — arXiv:2209.01188 · SWARM parallelism — arXiv:2301.11913
- Hivemind / learning@home — arXiv:2002.04013 · DeMo — arXiv:2411.19870
- FineWeb-Edu — huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- ADMX — the Axion Dark Matter eXperiment, University of Washington. The namesake
  haloscope, about an hour up the road from Subhalo-0.
