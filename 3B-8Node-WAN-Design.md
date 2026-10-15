# Axion 3B: eight 12 GB nodes across the internet

Design date: 2026-10-15. Status: proposed architecture and implementation plan.

Build this after the current 1B run finishes. Nothing in this document changes
the running trainer, launches GPU jobs, or provisions infrastructure. Configuration
examples describe proposed interfaces, not capabilities already implemented.

## 1. Recommended design

Use one dense 3.006B model partitioned into eight contiguous pipeline stages,
one per GPU. Preserve Axion's local ownership of weights, gradients, and optimizer
state. Send hidden activations forward and activation gradients backward between
neighbors. Use synchronous optimizer updates and a memory-bounded 1F1B schedule.

The model fits this memory arrangement on paper. Internet bandwidth, sustained
upload capacity, and availability determine whether it is practical. Do not
carry forward the throughput predictions from the older heterogeneous concept.

The exact GPU models, CPU/RAM, network rates, RTTs, data caps, and uptime are
not yet known. Equal VRAM does not imply equal compute speed. All performance
numbers below are calculations or explicit targets, not measurements of the
future cluster.

```text
Local corpus                                           Local corpus/labels
     |                                                        |
     v                                                        v
 [node 0] -> [node 1] -> [node 2] -> ... -> [node 6] -> [node 7]
 embedding                                              norm + LM head
 layers 0-4                                             layers 41-45
       forward: BF16 activations; backward: BF16 activation gradients

 CPU coordinator: membership, sampler position, health, checkpoint manifests
 Worker services: exactly one live worker for each (run, attempt, stage)
```

The coordinator can run on one of the eight hosts; it does not need a ninth GPU.
Bulk training traffic goes directly between stage neighbors, rather than through
the coordinator.

## 2. Model specification

Keep the current Axion model family for the first 3B version. This minimizes
architectural changes while the distributed runtime is being replaced.

| Property | Proposed value |
| --- | ---: |
| Exact parameter count | 3,005,957,376 |
| Decoder layers | 46 |
| Hidden width | 2,304 |
| Attention heads / head dimension | 18 / 128 |
| Attention | Full multi-head causal attention, RoPE |
| Feed-forward width / activation | 9,216 / GELU |
| Normalization | RMSNorm |
| Vocabulary | 16,384, existing Axion tokenizer |
| Embeddings | Untied input embedding and output head |
| Initial training context | 2,048 tokens |
| Initial microbatch | One sequence per microbatch |
| Compute / communication precision | BF16, when supported by all GPUs |
| Parameters, gradients, Adam moments | FP32 for the initial implementation |
| Activation checkpointing | Per transformer block |
| Attention implementation | PyTorch SDPA with a measured memory-efficient kernel |

The parameter count was verified with both `src/axion/params.py` and construction
of `AxionModel` on PyTorch's `meta` device, without allocating real 3B weights.

The current layer formula is:

```text
parameters_per_layer = 4*d_model^2 + 2*d_model*d_ff + 2*d_model
                     = 63,705,600

total = 46*63,705,600 + 2*(16,384*2,304) + 2,304
      = 3,005,957,376
```

The narrower width reduces boundary bytes per token by 25% relative to a
3,072-wide candidate. The extra depth is a deliberate bandwidth tradeoff,
not a claim that this shape outperforms every other 3B architecture.

Reuse of the 16K tokenizer is a starting assumption. Measure token efficiency
on the intended English/code/domain mixture before locking it. Changing the
vocabulary changes parameter counts, endpoint balance, and checkpoint contracts.

## 3. Eight-stage allocation

Use explicit layer ownership, not the current equal-remainder partition helper.
Ranges below are inclusive and use zero-based global layer IDs.

| Node/stage | Global layers | Layer count | Additional parameters | Total parameters | Persistent training state, GiB |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | 0-4 | 5 | Token embedding | 356,276,736 | 5.31 |
| 1 | 5-10 | 6 | None | 382,233,600 | 5.70 |
| 2 | 11-16 | 6 | None | 382,233,600 | 5.70 |
| 3 | 17-22 | 6 | None | 382,233,600 | 5.70 |
| 4 | 23-28 | 6 | None | 382,233,600 | 5.70 |
| 5 | 29-34 | 6 | None | 382,233,600 | 5.70 |
| 6 | 35-40 | 6 | None | 382,233,600 | 5.70 |
| 7 | 41-45 | 5 | Final norm and output head | 356,279,040 | 5.31 |

Persistent state assumes 16 bytes/parameter: FP32 parameter + FP32 gradient +
two FP32 Adam moments. It excludes activations and temporary allocations.

For the largest stage, allow approximately another 0.71 GiB for cached BF16
parameter casts, plus checkpointed activations, transient attention/MLP buffers,
logits/loss workspace, CUDA context, and communication buffers. A planning range
is roughly 8-10.5 GiB peak per GPU at 2K context, but the smoke run must establish
the actual peak. Do not treat this range as a guarantee.

Measure both PyTorch allocated/reserved peaks and total device memory usage.
Target at least 1-1.5 GiB of device headroom after warmup and first optimizer
step. Measure the optimizer-state allocation and checkpoint capture too.

If memory is tight: reduce live communication buffers, verify efficient SDPA,
and reduce context before trying local CPU optimizer offload. Moving optimizer
state to another internet node is not an acceptable memory workaround.

The proposed layer map is the initial balance. Profile endpoints, including the
loss, and each GPU's forward/backward time. Adjust whole-layer ownership if GPU
speeds differ, subject to each stage's memory limit. Record global layer IDs in
checkpoints so a controlled offline repartition is possible later.

## 4. Per-node resources

- One dedicated 12 GB CUDA GPU; test BF16 support and real distributed collectives.
- 32 GB system RAM minimum; 64 GB preferred for staging and checkpoint recovery.
- SSD storage, preferably 500 GB at corpus endpoints and at least 250 GB elsewhere.
- Maintain at least 100 GB free before a long run; actual dataset placement may
  require more. Middle nodes do not need the full training corpus.
- One pinned Python/PyTorch/CUDA/NCCL environment tested on all GPU models.
- GPU kernel driver and installed user-space libraries must agree. Preflight
  must run NCCL, not just a single-GPU tensor operation.
- Schedule driver/kernel updates and reboots around the run. No reboot or driver
  update procedure should be allowed to silently restart only part of the pipeline.

Do not infer that eight GPUs provide a single 96 GB address space. Every assigned
stage must fit independently, including its optimizer and transient allocations.

## 5. Pipeline scheduling

Replace the current all-forward/all-backward loop with non-interleaved 1F1B:
warm up the pipeline, alternate forward and backward work, then drain it before
the optimizer update. Limit retained microbatch graphs and communication buffers.

Start with 32 microbatches, then test 64. At 2K context these represent 65,536 and
131,072 tokens per optimizer update. Microbatch count is not the node count.

For a balanced ideal pipeline, a useful bubble approximation is
`(stages - 1)/(microbatches + stages - 1)`:

| Microbatches | Tokens/update at 2K | Idealized bubble |
| ---: | ---: | ---: |
| 2 | 4,096 | 77.8% |
| 16 | 32,768 | 30.4% |
| 32 | 65,536 | 17.9% |
| 64 | 131,072 | 9.9% |

These percentages ignore bandwidth, latency, load imbalance, and failures. 1F1B
bounds activation lifetime relative to fill-drain scheduling; it does not remove
pipeline bubbles or make internet links faster. Large accumulation counts do
not reduce bytes per training token.

Use manually constructed stages with PyTorch's `PipelineStage` and
`Schedule1F1B` as the first implementation candidate, pinned to a tested release.
The official runtime provides stage scheduling and communication machinery;
verify its buffer allocation and scheduling behavior on the chosen backend.
See [PyTorch pipeline parallelism](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)
and the original [GPipe paper](https://arxiv.org/abs/1811.06965).

Avoid virtual/interleaved stages initially: extra stage transitions across these
internet links complicate the traffic and recovery budget.

### Gradient correctness is a release gate

Normalize the accumulated loss exactly once. For equal-length microbatches,
the gradient must equal the gradient of the mean loss over the complete batch.
Never divide the received gradient by the microbatch count again at each stage.

The current manual Axion loop divides in the final-stage loss backward and
also divides received gradients in earlier stages. That behavior must not be
copied into the new eight-stage implementation. Keep the current live run
unchanged; add gradient-parity tests when building the new trainer.

All stages must use one consistent accumulation/scaling contract. After the
pipeline drains, reduce only scalar norm/finite-status information across
stages, clip using the global norm, and perform one local AdamW update per stage.
If any stage reports non-finite gradients, every stage skips the update together.

## 6. Internet topology and transport

Use an authenticated WireGuard mesh, or an equivalent private overlay with
verified direct peer connections. Bind rendezvous, Gloo, and NCCL to the private
interfaces. Keep training ports private and restrict peer membership.

Measure all 28 host pairs in both directions before assigning stage order.
Record sustained transfer rates, RTT percentiles, jitter, loss, and simultaneous
traffic to both neighbors. Arrange the chain to avoid a weak link in an interior
position when possible. Slow upload connections are often better endpoints,
provided endpoint compute and availability remain acceptable.

Do not put a single VPN relay or coordinator in the bulk data path. An overlay
that falls back to a shared relay must be measured as that slower topology.
NAT/CGNAT and firewall reachability must be resolved before a run is admitted;
keepalive settings alone do not guarantee peer connectivity. See
[WireGuard's NAT guidance](https://www.wireguard.com/quickstart/).

Baseline transport: NCCL socket P2P for CUDA tensors over the private overlay,
plus a separate Gloo control group. Interface selection and disabling IB/RoCE
are supported by [NCCL's configuration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
The future launch configuration should explicitly set the selected interface
and use `NCCL_IB_DISABLE=1` on hosts without a usable RDMA network.

Do not assume CUDA operations succeeding means NCCL can initialize. Require a
small real eight-rank collective and neighbor P2P test before model allocation.

If the measured socket implementation stalls GPU work, evaluate a CPU-staged
Gloo transport with bounded pinned buffers as a separate implementation path.
That requires explicit scheduling, copies, and parity testing; it is not a
configuration-only replacement for GPU `PipelineStage` buffers.

## 7. Communication budget: the principal constraint

All following rates are decimal Mbps/GB/TB; memory columns use binary MiB/GiB.
The formulas describe the BF16 payload before framing, encryption, retransmits,
checkpoint transfers, or other users of the link.

```text
one activation = micro_batch * sequence_length * hidden_width * 2 bytes
               = 1 * 2,048 * 2,304 * 2 = 9 MiB

one boundary per token:
    forward activation = 4,608 bytes
    backward gradient  = 4,608 bytes in the opposite direction

interior node per token:
    upload   = activation to next + gradient to previous = 9,216 bytes
    download = activation from previous + gradient from next = 9,216 bytes

shared-uplink token/s ceiling = effective_upload_bits_per_second / (8 * 9,216)
```

An interior node's forward and backward sends share its uplink. Budgeting only
one neighbor's traffic overestimates throughput by up to twofold. Download and
per-path limits also apply; use the minimum feasible rate across the actual chain.

| Interior-node uplink | Raw token/s ceiling | 60B-token duration at that ceiling |
| ---: | ---: | ---: |
| 20 Mbps | 271 | 2,560 days |
| 50 Mbps | 678 | 1,024 days |
| 100 Mbps | 1,356 | 512 days |
| 200 Mbps | 2,713 | 256 days |
| 500 Mbps | 6,782 | 102 days |
| 1,000 Mbps | 13,563 | 51 days |

These are optimistic network-only lower bounds on duration, not expected
end-to-end speeds. Compute, bubbles, protocol overhead, backup traffic, and
downtime lengthen them. Advertised ISP download speed is not sustained upload.

At 1,000 training tokens/s, each interior node uploads approximately 796 GB/day
and downloads another 796 GB/day. At 60B tokens, each interior node transmits
about 553 TB. Across all seven boundaries, the cluster sends about **3.87 PB**
of activation/gradient payload, counting each transmission once. Summing both
transmit and receive counters would count the same bytes twice.

Data caps and transfer charges therefore need checking as early as VRAM.
Shorter context shrinks each message (2K: 9 MiB; 4K: 18 MiB; 32K: 144 MiB), but
does not reduce communication bytes per token at a fixed width and wire dtype.

### Compression path

Establish a correct BF16 reference first. Then evaluate 8-bit blockwise boundary
quantization of both activations and activation gradients. Include scale metadata,
finite checks, and any residual/error-feedback state in the design and checkpoint.

Halving payload can approximately double the network-only ceiling, but does not
guarantee twice the useful training throughput or equal convergence. Require
gradient/loss tests and matched-token validation comparisons against BF16.
Treat 4-bit boundary transport as a later experiment, not a requirement for the
first run. Inference quantization is a separate decision from training transport.

[SWARM parallelism](https://arxiv.org/abs/2301.11913) demonstrates relevant
techniques for slow, unreliable networks, including compression and rebalancing.
It does not establish a speed guarantee for this dense 3B model on eight hosts.

## 8. Training data and optimizer plan

Prepare and verify data before training rather than streaming it across the
internet during every step. A 60B-token uint16 corpus is about 120 GB before
metadata and validation material; shard it for resumable preparation/downloads.

Store matching corpus shards at the first and last stages. Middle stages need
configuration and sample descriptors, not copies of the entire corpus. A sample
descriptor must identify dataset/version/hash, global token position, sequence
length, microbatch, and deterministic sampling seed. Endpoints verify token and
label alignment and tokenizer/EOS/packing contracts at startup.

Start with a 100M-token system pilot, then a 1B-token validation milestone.
Use 60B high-quality tokens as an initial full-run planning budget, subject to
the pilot's throughput and validation results. The historical compute/data
scaling work is useful context, not a promise that 60B tokens produce a highly
competitive 3B model: [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

Initial optimizer proposal: AdamW, betas `(0.9, 0.95)`, weight decay `0.1`,
global grad clip `1.0`, candidate peak LR `2e-4`, cosine decay, and a candidate
floor of `2e-5`. Tune LR and warmup in the pilot; the current 1B run is not a
numerical guarantee for the corrected eight-stage implementation.

At 64 microbatches of 2K tokens, an update contains 131,072 tokens and a 60B-token
run takes 457,764 updates when rounded upward. Track scheduler progress by
consumed tokens, so changes in batch size or context do not silently alter it.

Use fixed, deduplicated validation material excluded from training. Evaluate
approximately 0.5-1M fixed validation tokens at major milestones, alongside
task-oriented benchmark and generation checks. Record dataset mixture, source
provenance, tokenizer hash, and evaluation token count with every release.
Set evaluation cadence using measured wall time rather than copying the 1B
run's four-batch evaluation setting.

Begin at 2K context. Move to 4K/8K only after profiling and a deliberate curriculum
decision. Keeping tokens/update fixed means reducing microbatch count as sequence
length rises, which increases bubbles. Longer-context training is additional
work, and the export must report the context actually trained and validated.

## 9. Failure behavior and coordinated recovery

Exactly eight GPUs provide no idle replacement for a missing stage. A lost stage
stops useful training for the whole chain. Loose geography is feasible; frequent
uncoordinated offline periods are a different and much harder problem.

For illustration, eight independent nodes each available 95% of the time give
only `0.95^8 = 66.3%` joint availability, before recovery overhead. At 99% each,
joint availability is 92.3%. Real correlated failures can be worse.

The initial policy is fail-stop and restart all eight workers from the latest
globally committed checkpoint. Do not continue with seven workers or silently
use stale stage weights. Automatic repartition and SWARM-style redundancy are
later projects requiring memory tests, state migration, and usually more capacity.

Give each attempt a unique ID and generation/lease. Each worker reports stage,
attempt, optimizer step, sampler position, heartbeat, GPU health, and disk status.
One elected coordinator launches or restarts workers; local services do not
independently start competing global jobs. Use local locks and explicit service
ownership to prevent orphan workers and port collisions.

Loss of connectivity invalidates the attempt, including any partial optimizer
update. Restart all stages from one committed step and reconstruct process
groups. Include tokens, RNG, LR, and quantizer state in recovery. Do not try to
reuse an NCCL communicator after a failed worker has departed.

Use bounded timeouts based on measured p99 step/link behavior, then exponential
restart backoff with a cap and jitter. Disk exhaustion, driver mismatch, corrupt
data, and missing checkpoint sets should enter a visible blocked state instead
of causing thousands of identical retries. A short WAN stall may wait within the
operation deadline; a failed attempt is not partially replayed while peers run.

## 10. Checkpoints: two complete generations, not two unrelated files

The full FP32-parameter/FP32-Adam training checkpoint contains roughly 12 bytes
per parameter, excluding small metadata/RNG overhead:

| Artifact | Approximate size |
| --- | ---: |
| Endpoint stage checkpoint | 3.98 GiB each |
| Middle stage checkpoint | 4.27 GiB each |
| One complete eight-stage generation | 33.59 GiB / 36.07 GB |
| Two complete generations | 67.19 GiB / 72.14 GB cluster-wide |
| Merged BF16 inference weights | 6.01 GB |

Checkpoint only at a globally agreed optimizer-step boundary. Initial cadence:
roughly every 30 minutes locally, and a complete off-node backup roughly every
two hours. Measure overhead and adjust; copying the old 2,000-step interval may
lose many hours on the new batch and network configuration.

Commit protocol:

1. Drain all stages and agree on step, sampler position, and attempt.
2. Each stage writes an immutable temporary snapshot, checks size/hash, fsyncs,
   and renames it locally. Capture a consistent CPU copy before letting an async
   writer overlap later parameter updates.
3. Each worker reports READY with its hash, architecture/layer map, token count,
   tokenizer/dataset hashes, and optimizer/RNG/scheduler state identifiers.
4. The coordinator publishes an atomic manifest only after all eight are READY.
   A local file by itself is not a resumable distributed checkpoint.
5. Keep the two newest committed complete generations. During capture, provision
   room for an in-progress third generation and pinned backup transfers.
6. Prune an old generation only after the replacement manifest is durable and
   every required stage acknowledges the retention decision.

If stage 0 runs out of disk, stages 1-7 must not each prune away the only shared
recovery step. This is the specific failure that the new protocol must prevent.
The current 1B recovery experience shows why per-node filename counts are
insufficient.

For off-node resilience, replicate each stage's full resumable snapshot to one
other host or an object store, not to a single overloaded coordinator. Preserve
two complete committed remote generations. A neighbor's disk can hold a backup
without hosting its GPU stage. Restore the shard before retrying a lost host.

Budget replica traffic explicitly: a 4.6 GB shard every two hours averages about
5 Mbps of additional upload. Keep the backup queue bounded, pin in-flight source
files, and throttle transfers. Do not promise a second simultaneous remote copy
without accounting for its bandwidth and storage.

The Hub inference export is separate from the optimizer backup. Uploading BF16
weights to Hugging Face does not preserve the full optimizer/RNG state.

## 11. Disk and operations safeguards

Before admitting a long run, check free bytes and projected requirements on every
host: local retained generations, an in-progress save, remote backup replicas,
corpus placement, export staging, and package/runtime overhead. Use the measured
stage checkpoint size rather than only a fixed percentage of disk capacity.

Set alerts around 100 GB free and stop admitting new optimizer work at the safe
checkpoint/disk reserve threshold. Keep the most recent committed generations
protected during every cleanup. A failed save must not advance the committed
manifest or trigger independent retention on other stages.

Bound caches and export staging. After verifying a Hub commit and its hashes,
delete obsolete local inference exports under a configured policy. Do not
duplicate every old optimizer checkpoint into permanent export directories.

Persist metrics in newline-complete records and tolerate an incomplete trailing
record when a monitor reads a file being appended. A monitoring parse failure
must not restart an otherwise healthy trainer.

Record per-stage forward/backward/optimizer time, communication wait and overlap,
queue depth, bytes in each direction, total memory peaks, throughput, validation,
learning rate, global grad norm, step progress, checkpoint/backup time, and free
disk. Keep one small cluster status view with the current attempt and last
committed recoverable step.

## 12. Required Axion changes after the 1B run

| Area | Current limitation observed in this checkout | Proposed change |
| --- | --- | --- |
| `src/axion/stages.py` | Rejects non-three-stage construction; head and loss fixed at rank 2 | Arbitrary world size, explicit global layer map, first/last predicates |
| `src/axion/train_pipeline.py` | Three-rank groups/routing, fill-drain schedule, repeated gradient scaling | Eight-stage 1F1B, one scaling contract, token-based schedule, globally coordinated updates/evaluation |
| `src/axion/dark_photon.py` | Individual sends/receives wait immediately | Bounded asynchronous buffers, explicit ordering and per-direction telemetry |
| `src/axion/config.py` | No cluster, partition, or recovery schema | Validated host/stage map, transport, scheduler, checkpoint, and disk policies |
| `src/axion/data.py` | Training currently constructs a token source on every rank | Endpoint data roles, verified deterministic descriptors |
| `scripts/launch_m1.sh` | Exactly three hosts/interfaces | Separate eight-node launcher using manifests and supervised worker leases |
| `scripts/run_week_training.py` | Three-node defaults and existing supervision assumptions | General coordinator or dedicated new cluster controller |
| `scripts/merge_ckpt.py`, `scripts/export_hf.py` | Three-shard assumptions and three-stage mapping | Read eight-shard committed manifest; preserve global layer names and exact tensor set |
| `src/axion/params.py` | Default partition estimate does not include explicit endpoint balancing | Estimate from the declared layer map and precision state budget |
| Tests | Existing small three-stage tests | Eight-stage numerical, ordering, recovery, failure, export, and resource contracts |

Preserve the working three-node entry point while the new path is developed.
Pin architecture, code commit, environment, partition, and tokenizer hashes in
each run manifest. Do not silently resume a checkpoint under a changed mapping.

Illustrative future design schema, **not accepted by today's config loader**:

```yaml
model:
  vocab_size: 16384
  d_model: 2304
  n_layers: 46
  n_heads: 18
  d_ff: 9216
  seq_len: 2048
  rope_theta: 500000
  tie_embeddings: false
  activation_checkpointing: true
pipeline:
  stages: 8
  layer_counts: [5, 6, 6, 6, 6, 6, 6, 5]
  schedule: 1f1b
  wire_dtype: bfloat16
  transport: nccl_socket_over_private_mesh
  compression: none
train:
  micro_batch: 1
  n_microbatches: 64
  parameter_dtype: float32
  gradient_dtype: float32
  optimizer: adamw
  optimizer_state_dtype: float32
  total_tokens: 60000000000
recovery:
  committed_generations_to_keep: 2
  local_checkpoint_target_seconds: 1800
  offnode_backup_target_seconds: 7200
  restart_policy: whole_pipeline_from_committed_manifest
```

Hostnames, addresses, credentials, interface names, and storage paths belong in
deployment configuration, not hard-coded library logic or committed secrets.

## 13. Acceptance gates and implementation sequence

1. **Inventory and network gate.** Capture GPU models and supported precision,
   RAM/disk, all-pair links, two-neighbor contention, data caps, and uptime. Apply
   the traffic formulas before committing to a token target or calendar date.
2. **CPU correctness gate.** Eight Gloo stages on a tiny Axion model must match
   an unpartitioned model's loss, gradients, global clipping, and optimizer update.
   Test microbatch counts 1/2/8/16/32/64, ordering, and fixed-shape/dtype contracts.
   Schedules with unsupported counts must reject them explicitly.
3. **GPU fit gate.** One stage at a time on the eventual idle 12 GB devices:
   profile memory and service time through optimizer allocation and checkpoint
   capture. Verify endpoint logits/loss workspace and the selected SDPA backend.
4. **Eight-node systems gate.** Run a small distributed model before loading 3B.
   Exercise NCCL/Gloo setup, 1F1B backpressure, WAN latency, and bounded queues.
5. **Recovery gate.** Kill a worker mid-microbatch and mid-update; disconnect a
   link; fill a test checkpoint volume; interrupt a save and backup. Resume all
   stages from the same committed generation. Preserve the last two complete
   generations through failed saves, and verify no duplicate processes remain.
6. **3B pilot gate.** Run 100M tokens at 2K context and record useful token/s,
   validation, memory, queue behavior, disk use, and total WAN transfer. Test the
   optional 8-bit transport against the BF16 reference before adopting it.
7. **Economics and quality gate.** Extrapolate from measured throughput and joint
   availability, including validation/checkpoint overhead. Approve the 1B-token
   milestone, then the full budget if the result is worthwhile.
8. **Release gate.** Export from a committed eight-stage manifest, compare merged
   and pipeline logits on fixed inputs, test the intended Transformers/vLLM
   environment, and publish an accurately labeled base-model checkpoint.

## 14. Release and timing expectations

Keep every existing 1B Hub release. Create separate 3B model repositories or
explicit immutable release revisions; record trained tokens and validated
context, not just a step number. Preserve optimizer backups outside inference
repositories. Follow pretraining with a separately evaluated instruction-tuning
phase if a conversational assistant is the actual product goal.

Do not quote the current 1B throughput as the future 3B speed. Calendar planning
must use measured useful throughput and availability. For reference only:

| Measured effective throughput | Time for 60B tokens |
| ---: | ---: |
| 500 tokens/s | 1,389 days |
| 1,000 tokens/s | 694 days |
| 2,000 tokens/s | 347 days |
| 5,000 tokens/s | 139 days |

If these rates exclude downtime/evaluation/checkpoints, add that overhead; if they
already measure useful tokens divided by total wall time, do not add it twice.

Recommendation: proceed with the eight-stage system design, but make the WAN
pilot the decision point. Stable direct links with several hundred Mbps upload
are a much stronger starting point than residential links with tens of Mbps.
For 50-100 Mbps interior uploads, even this memory-feasible 3B design is likely
a long experiment unless validated compression or a different deployment
arrangement changes the communication budget substantially.
