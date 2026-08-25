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
