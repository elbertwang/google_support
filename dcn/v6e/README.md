# `dcn/` on TPU v6e — measurement notes

We ran the `dcn/` package from this repo on **TPU v6e** (2 × `ct6e-standard-4t`,
DP=2, GKE + DRANET) instead of TPU7x. `dcn/benchmark.py` is **untouched** — its
SHA256 still matches the value pinned in `dcn/README.md`.

**Read [`METHODOLOGY.md`](METHODOLOGY.md) for the full write-up.** This file is
the short version.

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

Six things move it, in order of size. If you are seeing ~140, start at #1.

| # | thing | effect | detail |
|---|---|---|---|
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

## Configuration levers: there are none

35 MegaScale runtime flags and 4 configurations from the public XLA flags doc,
across two rounds — the second aimed specifically at the receive path after
xprof narrowed it there. All under the clean protocol. Noise floor first (baseline ×6: mean 192.1,
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
results/optsweep2/          round-2 flag sweep, receive-path targeted
results/tpu7x/              cross-platform check on two 2x2x1 tpu7x slices
results/metrics-*.jsonl     1-NIC vs 2-NIC
```

Larger artifacts (full HLO dumps, xprof traces, ~1 GB) are public, no auth:
`https://storage.googleapis.com/yppublic/v6e-dcn-allreduce-20260826/`

## Known gaps

- The mechanism behind #2 is not identified, though xprof narrows it to the
  receive path: poisoning leaves `send-done` untouched and multiplies
  `recv-done` by 2.3x and `barrier-cores` by 11.5x.
- Only DP=2 was measured.
