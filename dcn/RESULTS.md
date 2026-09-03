# Falcon reproduction results

Run date: 2026-08-24 (UTC)

## Run identity

| Physical slice | Falcon experiment | Job | Artifact |
|---|---|---|---|
| 0 | `exp-6fehifhu0w` | `job-fq9v9c19mt` | `art-6o763b2h8w` |
| 1 | `exp-u60vg4gji7` | `job-zssx138wqn` | `art-hwbhusu0dw` |

Both Falcon experiments and artifacts reached `SUCCEEDED`.

The optional `operator-analysis` record is `an-bgni8h0323`. Falcon accepted it
and resolved plugin version `pv-0a9sdwijm6`, but the analyzer controller did not
claim it: the record remained `PENDING` with an empty `updatedAt`, and
`falcon workflow analysis wait` reached its explicit 30-minute timeout. This is
a Falcon analyzer-queue follow-up; it does not affect the benchmark JSONL or
the two successful experiment artifacts.

## Preflight

```text
process_count: 2
local_device_count: 8
global_device_count: 16
slice_counts: {0: 8, 1: 8}
bidi block means: [2.0, 1.0]
uni block means:  [0.0, 1.0]
all-reduce means: [3.0, 3.0]
```

The run used JAX/JAXLIB 0.11.0, libtpu 0.0.44, two physical TPU7x
`2x2x1` slices, two 200 Gbps interfaces per host, BF16, five warmups, five
timed repetitions, and ten collective calls per timed batch.

## Selected results

Selection follows the source benchmark: choose the payload with the highest
best-observed host bandwidth, then report that payload's median.

| Variant | Dim | Shard/device | Reproduced median | Reference median | Ratio |
|---|---:|---:|---:|---:|---:|
| `ppermute_uni` | 32,768 | 1,024 MiB | **367.051 Gbps** | 364.726 Gbps | 1.006x |
| `ppermute_bidi` | 32,768 | 1,024 MiB | **294.618 Gbps** | 274.932 Gbps | 1.072x |
| `all_gather` | 32,768 | 1,024 MiB | **257.128 Gbps** | 254.010 Gbps | 1.012x |
| `all_reduce` | 16,384 | 256 MiB | **168.818 Gbps** | 172.084 Gbps | 0.981x |

## Payload sweep

| Matrix dim | Uni host TX | Bidi host TX | AllGather host TX | AllReduce host ring-equivalent |
|---:|---:|---:|---:|---:|
| 8,192 | 295.953 Gbps | 172.108 Gbps | 173.250 Gbps | 155.835 Gbps |
| 16,384 | 354.930 Gbps | 218.606 Gbps | 219.872 Gbps | 168.818 Gbps |
| 24,576 | 362.314 Gbps | 268.952 Gbps | 251.037 Gbps | 168.965 Gbps |
| 32,768 | 367.051 Gbps | 294.618 Gbps | 257.128 Gbps | 161.216 Gbps |

All selected results are within 8% of the recorded baseline. Three transport
cases are slightly faster; all-reduce is 1.9% lower. The movement checks and
network topology evidence passed, so this is a valid reproduction rather than
a single-slice ICI measurement.

## Instrumented host-network XLA flag experiment

Run date: 2026-09-02 (UTC)

These experiments addressed two measurement concerns: the first compiled call
was isolated from all reported samples, and the final confirmation used 10,000
compiled warmups rather than five. Each timed repetition dispatched ten BF16
dim-16384 all-reduces. Results below use the median of 20 paired repetitions,
where each repetition takes the slower of the two ranks and excludes the
trailing host barrier.

The environment was JAX/JAXLIB 0.11.0, libtpu 0.0.44, two TPU7x `2x2x1`
slices, and two 200 Gbps NICs per host. The Pods used `hostNetwork: true`
without a DRA ResourceClaim. MegaScale gRPC used `eth1,eth2,lo`; NIC counters
confirmed that both data interfaces carried the payload.

The flags under test were:

```text
--xla_tpu_dcn_all_reduce_combiner_threshold_bytes=1073741824
--xla_tpu_enable_sparse_core_collective_aggregator=true
```

Both were accepted by libtpu. The first JobSet screened both flags in separate
fresh processes with 1,000 warmups, bracketed by fresh baseline processes:

| 1k screen block | Critical median | p05-p95 | Bandwidth | 400 Gbps efficiency |
|---|---:|---:|---:|---:|
| Baseline/start | 113.197 ms | 109.119-117.739 ms | 151.770 Gbps | 37.94% |
| DCN all-reduce combiner, 1 GiB | 113.564 ms | 111.855-120.929 ms | 151.280 Gbps | 37.82% |
| Sparse-core collective aggregator | 110.670 ms | 107.900-112.685 ms | 155.235 Gbps | 38.81% |
| Baseline/end | 116.885 ms | 114.525-119.452 ms | 146.981 Gbps | 36.75% |

The ending baseline was 3.156% slower than the starting baseline. Relative to
a linearly interpolated baseline at each block's position, the combiner was
0.737% faster and the sparse-core aggregator was 4.481% faster. The combiner
also had a 166.709 ms outlier, so its small screen difference was unresolved.

The sparse-core flag advanced to a full 10,000-warmup A/B/A confirmation:

| 10k A/B/A block | Critical median | p05-p95 | Bandwidth | 400 Gbps efficiency |
|---|---:|---:|---:|---:|
| Baseline/start | 115.572 ms | 111.132-121.210 ms | 148.651 Gbps | 37.16% |
| Sparse-core collective aggregator | 117.252 ms | 111.115-123.216 ms | 146.521 Gbps | 36.63% |
| Baseline/end | 117.042 ms | 112.654-122.044 ms | 146.783 Gbps | 36.70% |

The candidate was 0.810% below the 147.717 Gbps mean of the bracketing
baselines. It was also below each control independently: -1.433% versus the
starting baseline and -0.179% versus the ending baseline. The positive 1k
screen signal therefore did not reproduce with 10,000 warmups.

Text scans of optimized HLO dumps were identical across the screen blocks and
both ranks: 127 HLO text files, 56 selected optimized files, and aggregate token
counts of 34 `all-reduce`, 22 `collective`, 416 `megascale`, zero `sparse`, and
zero `aggregator`. This does not rule out a backend-only change, but it provides
no compiler-artifact evidence that either flag transformed this single
all-reduce workload.

Neither flag is recommended for this microbenchmark based on these results.
Graphs containing multiple combinable DCN all-reduces or actual sparse-core
collectives may exercise different compiler paths and should be tested
separately.

## TPU7x dynamic slicing with Google's TCP rmem setting

Run date: 2026-09-03 (Asia/Shanghai)

The TPU7x-adapted Google `v6e-increase-rmem.yaml` DaemonSet was deployed on the
production Kueue TAS dynamic-slicing cluster. It reached 32/32 Ready Pods, but
all init logs showed that the node value was already the target both before and
after the write:

```text
4096 41943040 314572800
```

The benchmark then requested two dynamic `2x2x1` partitions using only the
`cloud.google.com/gke-tpu-slice-topology: 2x2x1` Pod annotation and the TPU7x
accelerator selector. It did not set a fixed topology or node-pool selector and
did not use JobSet exclusive topology. Kueue admitted the two replicas with
distinct partition IDs. Both Pods used the host network without DRA, and saw
`eth1` and `eth2` at 200 Gbps each.

The software and workload matched the earlier instrumented runs: JAX/JAXLIB
0.11.0, libtpu 0.0.44, BF16 dim-16384 all-reduce, an isolated first call,
10,000 warmups, and 20 paired batch-10 samples. Results use the slower rank for
each repetition and exclude the trailing barrier.

| Metric | Result |
|---|---:|
| First call, rank 0 / rank 1 | 254.198 / 253.005 ms |
| 10k warmup critical average | 114.737 ms/iteration |
| Paired collective-only median | 116.565 ms |
| p05-p95 | 111.641-124.156 ms |
| CV | 3.58% |
| Host algorithm bandwidth | **147.384 Gbps** |
| 400 Gbps efficiency | **36.85%** |

The previous static-node-pool host-network result was 145.812 Gbps (36.45%),
so this run was 1.078% faster. The previous DRA bracketing mean was 146.065
Gbps, making this run 0.903% faster. These differences are within earlier
run-to-run drift. Because the DaemonSet's before and after values were
identical, this is not an rmem A/B test and provides **no evidence of an
rmem-induced improvement**. It instead confirms comparable baseline throughput
on the production dynamic-slicing cluster.
