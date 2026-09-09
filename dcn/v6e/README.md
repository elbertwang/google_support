# `dcn/` on TPU v6e — measurement notes

We ran the `dcn/` package from this repo on **TPU v6e** (2 × `ct6e-standard-4t`,
DP=2, GKE + DRANET) instead of TPU7x. `dcn/benchmark.py` is **untouched** — its
SHA256 still matches the value pinned in `dcn/README.md`.

**Read [`METHODOLOGY.md`](METHODOLOGY.md) for the full write-up**, and
[`ALGORITHM.md`](ALGORITHM.md) for what the fused all-reduce actually emits.
This file is the short version.

> **If you only read one thing: [`HOST-REDUCTION.md`](HOST-REDUCTION.md)** (written in Chinese).
> ```bash
> export LIBTPU_INIT_ARGS="--xla_tpu_use_megascale_host_reduction=false"
> ```
> **+54% on `jax.lax.psum` over DCN**, stock libtpu, no custom binary, no graph
> change. By default the sum is folded into the host transfer and executed by CPU
> threads in host DRAM — which is why a DCN all-reduce emits *zero* `add`
> instructions in HLO. The flag makes XLA lower it to reduce-scatter +
> all-gather instead, pure DMA, with the add back on TPU HBM.
>
> This supersedes the hand-written ring in [`MANUAL-AR.md`](MANUAL-AR.md). That
> ring was never winning on algorithm — it is built from `ppermute`, so it was
> bypassing host reduction. The flag gets the same thing from the compiler, so
> the usual objection ("the DP all-reduce is GSPMD-generated, you cannot
> hand-write it") no longer costs you anything. `MANUAL-AR.md` is still worth
> reading for the measurements and the two traps it documents.

## The number

`all_reduce`, dim 32768 (1 GiB/device), two 200 Gbps NICs:

**198.0 ± 6.0 Gbps** — n=3 interleaved with controls, σ 7–8% within a run,
reproduced across ~15 runs over two days.

Against the matching raw-TCP ceiling (347.8 Gbps bidirectional per direction,
measured with neper on the same NICs) that is **57%**. For comparison
`ppermute_bidi` reaches 68% and `all_gather` 64%.

xprof on a clean run puts **82% of the time in `recv-done`** — waiting for the
reduced result, not sending. 43% of the total is not wire time.

## The same benchmark will give you 33–200 Gbps

Seven things move it, in order of size. If you are seeing ~140, start at #1;
#0 is the largest single lever but was found last.

| # | thing | effect | detail |
|---|---|---|---|
| 0 | `--xla_tpu_use_megascale_host_reduction=false` | **+54%** | Only affects `all_reduce`. Default lowering runs the sum on the host CPU. [`HOST-REDUCTION.md`](HOST-REDUCTION.md). |
| 1 | Both NICs actually inside the Pod | up to **2x** | The k8s manifests in this repo set `DCN_INTERFACES=eth1,eth2,lo` but declare neither `hostNetwork` nor a `resourceClaim`, so the Pod has only `eth0`. Our 1-NIC number was 132.8 vs 198.0. |
| 2 | `ppermute_uni` must not run before `all_reduce` | **5.4x** | Poisons every subsequent DCN collective, never recovers. `benchmark.py` defaults to running it first. |
| 3 | `--warmup-runs` 5 → 200 | +11%, σ 33%→8% | Saturates at 200; 1000 buys nothing. |
| 4 | Discard the first run of a session | 15% | Cold start lands consistently low. |
| 5 | Report `best`, not `median` | ~9% | Long right tail the median does not reject. |
| 6 | Match directionality vs the ceiling | — | `ppermute_uni` only sends; everything else is full duplex. |

#2 also means the reference table in `dcn/RESULTS.md` needs re-measuring: it was
produced with the default variant order, so `ppermute_bidi` 274.932,
`all_gather` 254.010 and `all_reduce` 172.084 were all taken after
`ppermute_uni` ran. Only `ppermute_uni` 364.726 is clean.

## Quick check before trusting any number

From inside a running Pod:

```bash
for i in $(ls /sys/class/net | grep -v lo); do
  printf "%s speed=%s mtu=%s\n" "$i" \
    "$(cat /sys/class/net/$i/speed 2>/dev/null)" "$(cat /sys/class/net/$i/mtu)"
done
```

You want `eth1` and `eth2` both at `200000`. If they are missing you are
measuring half the fabric.

## Running it the clean way

```bash
kubectl apply -f ../k8s/dranet-claim.yaml
DIM=32768 WARMUPS=200 ONLY=baseline scripts/optsweep.sh    # all_reduce alone
scripts/order-matrix.sh                                    # reproduce finding #2
DIR=bidi scripts/nicbw.sh                                  # raw TCP ceiling
```

Discard the first run of each session.

## Two ways to get the NICs in — pick one, never both

- **DRA claim** — `../k8s/dranet-claim.yaml` + `../k8s/jobset-v6e-dranet.yaml`.
  Needs a DRANET-enabled cluster. Everything else in this directory was
  measured on this path.
- **`hostNetwork: true` with no claim** — a hostNetwork Pod sees the node's
  `eth1`/`eth2` directly, no DRANET needed. `../k8s/jobset-v6e-hostnet.yaml`.

Interleaved A/B says the two are **equivalent**: DRANET 191.4 ± 14.9 vs
hostNetwork 191.6 ± 6.7 Gbps, +0.1% apart, both splitting ~50/50 across the
NICs. Pick whichever fits your cluster (`results/netpath/`).

Combining the two fails hard and non-obviously:

```
NRI RunPodSandbox failed: using host network can not claim host devices
```

The Pod then retries forever while holding the ResourceClaim, so the node stays
occupied. We lost a production node pool to this for 21 hours.

## What actually works

> **Superseded.** This section concluded that nothing in the configuration space
> helps and that you must rewrite the graph. Both halves turned out to be wrong,
> for the same reason: `--xla_tpu_use_megascale_host_reduction=false` is a
> configuration lever, and it delivers what the rewritten graph delivered,
> because the rewritten graph was only ever bypassing host reduction.
> [`HOST-REDUCTION.md`](HOST-REDUCTION.md). The measurements below stand; the
> conclusion drawn from them does not.

What works at the graph level: at DP=2, `x + ppermute(x, [(0,1),(1,0)])`
instead of `psum`.

| | `psum` | hand-written | gain |
|---|---:|---:|---:|
| v6e, DP=2 | 193.9 ± 3.3 | 223.3 ± 8.5 (`exchange_add`) | **+15.2%** |
| tpu7x, DP=2 | 163.1 ± 2.7 | 271.3 ± 15.9 (`exchange_add`) | **+66.4%** |
| tpu7x, DP=4 | 118.3 ± 0.7 | 188.9 ± 11.4 (`ring_ar`) | **+59.7%** |

Scaling out (tpu7x, dim 32000, n dynamic 2x2x1 slices):

| DP | devices | `psum` | `ring_ar` | gain |
|---:|---:|---:|---:|---:|
| 2 | 16 | 169.1 | 285.6 | +68.8% |
| 4 | 32 | 113.8 | 183.8 | +61.5% |
| 8 | 64 | 95.1 ± 1.8 | 131.2 ± 10.0 | +37.9% |
| 10 | 80 | 82.0 ± 1.4 | 124.3 ± 6.1 | +51.6% |

The ring wins at every scale tested, 38–69%. The gain is not monotonic and with
1–3 runs per point the DP=8 vs DP=10 ordering is not separated.
`results/tpu7x/DP-SCALING.md`.

Three rounds each with the variant order rotated. At DP=4 `psum` reproduces to
sd 0.7, so that gap is roughly 100 sigma. Manually chunking `psum` buys almost
nothing (+1–3%), so the deficit is the fused `ALL_REDUCE` host transfer itself,
not message size.

Past DP=2 you need a real ring: the naive form moves 2x the bytes at n=4 and
loses 18%, and `psum_scatter` + `all_gather` is worse than `psum` at both scales.
See [`MANUAL-AR.md`](MANUAL-AR.md) for the full table and caveats.

### gRPC-over-TCP tuning specifically

Proposed as a short-term mitigation: `grpc_num_channels=16`,
`grpc_enable_numa_aware_transmit`, `grpc_use_event_engine_allocator`,
`grpc_enable_memcpy_eliding`. Measured individually and together on tpu7x
against both `psum` and `exchange_add`. Individually, `num_channels=16` and
`use_event_engine_allocator` *hurt* `exchange_add`. All four together, 3x
interleaved: +3.6% on psum (sd 7.3) and +2.9% on exchange_add (sd 16.2) — noise.

The premise does not hold anyway: the same gRPC/TCP transport, same hosts, same
process, same bytes, carries 167.6 Gbps as `psum` and 268.9 as `exchange_add`.
The transport is not the limit. `results/tpu7x/GRPC-TUNING.md`.

## Configuration levers: 56 screened, and we missed the one that mattered

> **Read with [`HOST-REDUCTION.md`](HOST-REDUCTION.md).** Everything below was
> measured with host reduction on, i.e. against a system bottlenecked elsewhere,
> so these null results say less than they appear to. Worse, one of the flags
> screened here — `--megascale_use_top_level_all_gather_and_local_reduction_for_ar`,
> whose name describes the mechanism that eventually worked — measured 164.3 vs
> 163.1 and led us to discard the correct hypothesis. Black-box screening cannot
> tell "no effect" from "dead switch".

52 MegaScale runtime flag configurations across four rounds and two platforms
(v6e generic, v6e receive-path-targeted after xprof narrowed it there, then the
best candidates repeated on tpu7x), plus 4 configurations from the public XLA
flags doc. All under the clean protocol. Noise floor first (baseline ×6: mean 192.1,
sd 11.1); every result fell inside the 2σ band, and the single-shot leaders
collapsed to parity when re-run 3× interleaved (198.3 and 198.9 against a
baseline of 198.0). The XLA-flag configs produced a **byte-identical** optimized
HLO. Same for round 2: the leaders fell to or below baseline on repetition, and
baseline had the tightest spread. See `results/optsweep/SUMMARY.txt` and
`results/optsweep2/SUMMARY.txt`.

The limit is per-host and structural, not tunable: on tpu7x a participant scan
at dim 32768 gives 161.0 / 166.1 / **168.4** Gbps for 8 / 4 / **2** devices — one
chip pair alone saturates it. Same ratio across platforms (v6e 57%, tpu7x 45%,
upstream's own tpu7x table 47%). `results/tpu7x/SUMMARY.md`.

## What changed in the package

All backward compatible — defaults unchanged.

| file | change |
|---|---|
| `dcn/k8s/entrypoint.sh` | `DCN_DEVICES_PER_SLICE=8` → env-overridable (v6e needs 4) |
| `dcn/run_slice.sh` | `--participants-per-slice` env-overridable, exposes `--participant-scan`, adds `DCN_PROFILE=1` |
| `dcn/distributed_runner.py` | `jax.profiler.trace` when `DCN_PROFILE=1` — `--profile-dir` is parsed but never used and nothing calls `jax.profiler`, so the package produced no trace at all |
| `dcn/k8s/dranet-claim.yaml`, `dcn/k8s/jobset-v6e-dranet.yaml`, `dcn/k8s/jobset-v6e-hostnet.yaml` | new |

Two other rough edges we hit but did not change:

- `results.py` hardcodes all four variants in `BASELINE_GBPS`, so running a
  subset raises `missing variant` and `entrypoint.sh` exits 1 — a good run looks
  like a failure. Given #2, running subsets is exactly what you want to do.
- `--profile-dir` is dead code (see above).

## Layout

```
METHODOLOGY.md              full write-up
scripts/                    order-matrix.sh, optsweep.sh, nicbw.sh, netpath-ab.sh + manifests
results/ordermatrix/        finding #2, 6 cases
results/optsweep/           flag sweep, read SUMMARY.txt first
results/nicbw/              raw TCP ceiling, both directions
results/netpath/            DRANET vs hostNetwork A/B
results/xprof/              clean vs poisoned trace logs (traces in the bucket)
results/hlo-dp4/            psum vs ring HLO at DP=4, the basis for ALGORITHM.md
results/optsweep2/          round-2 flag sweep, receive-path targeted
results/tpu7x/              cross-platform check on two 2x2x1 tpu7x slices
results/metrics-*.jsonl     1-NIC vs 2-NIC
```

Larger artifacts (full HLO dumps, xprof traces, ~1 GB) are public, no auth:
`https://storage.googleapis.com/yppublic/v6e-dcn-allreduce-20260826/`

## Important caveat on the tpu7x numbers

`tpu7x-standard-4t` is **dual-socket**: 2 chips + `eth1` on NUMA 0, 2 chips +
`eth2` on NUMA 1 (verified from PCI `numa_node`, see
`results/tpu7x/NUMA.md`). Every tpu7x figure here was measured with one
container requesting `google.com/tpu: 4`, so a single process drove all 8 JAX
devices and half its device-to-NIC paths crossed the UPI link.

GKE supports splitting a slice into two containers of `google.com/tpu: 2`, one
per NUMA node — and in fact *requires* that shape if you want less than a full
slice (`must be exactly 4 TPUs (full utilization)`). Internal data reports a large gap between the two layouts, so we checked
whether it applies here. Two cheap experiments say it does not:

- Plain TCP crossing UPI costs nothing measurable — 6 configurations of
  taskset-pinned neper over a local vs remote NIC all land at 189.5–189.9 Gbps.
- `megascale_transport_numa_node` in our single-process layout is a *restriction*,
  not an optimisation: it confines the transport to one node's NIC and drops
  `exchange_add` from 300.6 to ~179 Gbps (one NIC's line rate). `psum` moves
  +3–6%, inside its noise.

And `psum` at 158–168 Gbps is *below* one NIC's 190, so NIC or cross-socket
bandwidth cannot be what limits it. The 1-proc-per-NUMA rebuild is therefore not
justified by our evidence. What remains untested is the TPU-to-host-memory DMA
path specifically.

v6e is unaffected: `ct6e-standard-4t` is single-socket.

## Known gaps

- The mechanism behind #2 is not identified, though xprof narrows it to the
  receive path: poisoning leaves `send-done` untouched and multiplies
  `recv-done` by 2.3x and `barrier-cores` by 11.5x.
- DP 2, 4, 8, 10 measured; beyond that not.
- All tpu7x runs are single-process-per-host. The 1-proc-per-NUMA layout is not
  measured, but two cheap experiments argue it would not help here —
  `results/tpu7x/NUMA.md`. The TPU-to-remote-socket-memory DMA path specifically
  is still unmeasured.
