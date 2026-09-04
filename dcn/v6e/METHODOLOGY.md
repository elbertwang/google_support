# Measuring DCN all-reduce on TPU v6e: the protocol, and the six things that move the number 5x

Hardware: 2 × `ct6e-standard-4t` (v6e, 4 chips / 4 JAX devices per slice, DP=2),
GKE + DRANET, jax/jaxlib 0.11.0, libtpu 0.0.44, europe-west4-a.
Benchmark: `dcn/benchmark.py` from this repo, unmodified (SHA256
`4c31f548fd32b59e6b85cd507c2e5c302919da8da6e02a8fc4c2c9e9ec342973`).

**Our number: `all_reduce` at dim 32768 (1 GiB/device) = 198.0 ± 6.0 Gbps**,
n=3 interleaved, reproduced across ~15 runs over two days.

The same benchmark on the same hardware will give you anywhere from 33 to 200
Gbps depending on six things, none of which are obvious. They are listed below
in order of how much they matter. If you are seeing ~140, start at §1.

---

## 1. Are both NICs actually inside the Pod? (up to 2x)

This is the one to check first, because **the manifests in this repo do not
plumb the NICs in.**

`k8s/jobset-static.yaml`, `k8s/jobset.yaml` and `k8s/pod-pair.yaml` all set

```yaml
- name: DCN_INTERFACES
  value: "eth1,eth2,lo"
```

which becomes `--megascale_grpc_interface_prefixes=eth1,eth2,lo`. But none of
those manifests declares `hostNetwork: true` **or** a `resourceClaims` entry.
Without one of the two, the Pod's network namespace contains only `eth0`, and
MegaScale is being pointed at interfaces that do not exist there.

Check from inside the running Pod:

```bash
for i in $(ls /sys/class/net | grep -v lo); do
  printf "%s speed=%s mtu=%s\n" "$i" \
    "$(cat /sys/class/net/$i/speed 2>/dev/null)" "$(cat /sys/class/net/$i/mtu)"
done
```

What you want to see (both at 200 Gbps):

```
eth0 speed=10000  mtu=1460      <- pod overlay veth, does not carry DCN
eth1 speed=200000 mtu=8896
eth2 speed=200000 mtu=8896
```

If `eth1`/`eth2` are missing, everything is going over one interface and you are
measuring roughly half the fabric. Our single-NIC all-reduce measured
**132.8 Gbps** against 198.0 with two.

Two ways to fix, and **they are mutually exclusive**:

**(a) DRA claim** — `k8s/dranet-claim.yaml` + `resourceClaims` on the Pod.
Requires a DRANET-enabled cluster. **Do not add `hostNetwork: true` as well**;
the combination fails with

```
NRI RunPodSandbox failed: using host network can not claim host devices
```

and the Pod retries forever while holding the ResourceClaim, so the node is
stuck. We lost a production node pool to this for 21 hours.

**(b) `hostNetwork: true` with no claim** — a hostNetwork Pod sees the node's
`eth1`/`eth2` directly, and does not need DRANET. We confirmed the interfaces
are present and at 200 Gbps this way, but **did not run the benchmark over this
path**, so there are no collective numbers for it here. Everything measured in
this document used (a).

Sanity check that MegaScale actually bound to them — the interface list is
echoed in the env dump:

```bash
grep -o 'megascale_grpc_interface_prefixes=[^ "]*' rank-0.log
```

## 2. Never run `ppermute_uni` before `all_reduce` (5.4x)

`benchmark.py` defaults to
`--variants ppermute_uni,ppermute_bidi,all_gather,all_reduce` and runs them in
that order inside one process. **Running the unidirectional `ppermute` first
leaves the MegaScale runtime in a state where every subsequent DCN collective is
6–10x slower, and it never recovers.**

Controlled matrix — identical pods, identical payload, identical
`--warmup-runs 200`, only the order changes:

| order | all_reduce |
|---|---:|
| `all_reduce` alone | **189.4** (σ 7%) |
| `all_reduce, ppermute_uni` | **181.7** (σ 7%) |
| `ppermute_bidi, all_reduce` | **199.4** (σ 8%) |
| `all_gather, all_reduce` | **196.0** (σ 11%) |
| `ppermute_uni, all_reduce` | **36.6** (σ 16%) |
| `ppermute_uni, ppermute_bidi, all_gather, all_reduce` | bidi 21.9, ag 21.0, **AR 28.2** |

A *bidirectional* predecessor is harmless — those give the best all-reduce
numbers we ever measured. A *unidirectional* one poisons everything downstream.
Putting all-reduce first is clean, and the `ppermute_uni` that follows is
unaffected, so the damage only flows one way.

It is dose-dependent — the more unidirectional dispatches run first, the worse
it gets (`ppermute_uni,all_reduce` at dim 32768):

| `--warmup-runs` | all_reduce |
|---:|---:|
| 5 | 61.0 Gbps |
| 40 | 46.6 Gbps |
| 200 | 33.2 / 36.6 Gbps |

`ppermute` with perm `[(0,1)]` is half-open: slice 0 only sends, slice 1 only
receives. Something accumulates in proportion to the number of such dispatches
and then throttles subsequent cross-slice traffic. We have not identified the
mechanism.

**Practical rule: run one variant per process.** If that is too invasive, at
least never schedule `ppermute_uni` ahead of anything you care about.

**This affects the reference table in `dcn/RESULTS.md`**, which was produced with
the default order — `ppermute_bidi` 274.932, `all_gather` 254.010 and
`all_reduce` 172.084 were all measured after `ppermute_uni`. Only
`ppermute_uni` 364.726 is clean.

## 3. `--warmup-runs 5` is too few (+11%, and σ 33% → 8%)

Measured on all_reduce alone, dim 32768:

| warmups | Gbps | σ = (max−min)/min |
|---:|---:|---:|
| 5 | 179.8 | **33%** |
| 200 | 200.2 | **8%** |
| 1000 | 199.6 | 8% |

Saturates around 200; 1000 buys nothing. This is most of the reason
`ppermute_bidi` and `all_gather` show σ of 280–570% at the larger payloads with
the defaults.

## 4. Discard the first run of a session (15%)

The first run against a freshly idle node pool consistently lands ~15% low, then
the next five cluster tightly:

```
169.9  198.1  199.2  197.7  194.4  193.3     <- baseline repeated 6x
cold   \___________ warm, mean 196.5, sd 2.5 ___________/
```

Every low outlier in our 20-config flag sweep (166.7, 169.7, 173.0, 174.3) was a
first-run-after-a-gap. Budget one throwaway run per session or you will chase
ghosts.

## 5. Report `best`, not `median`

The timing distribution has a long right tail that median does not reject. At
dim 32768 our median was ~208 ms vs a min of 191 ms — a 9% difference on a good
run, much more on a noisy one. We report `host_*_best`.

## 6. Match directionality when comparing against a ceiling

`ppermute_uni` only sends. `ppermute_bidi`, `all_gather` and `all_reduce` send
and receive concurrently. Comparing a unidirectional measurement against a
bidirectional ceiling (or vice versa) is wrong, and we got this wrong once.

Raw TCP ceiling on the same two NICs, measured with
[neper](https://github.com/google/neper) `tcp_stream` from Pods holding the
identical DRA claim:

| neper mode | 1 NIC | 2 NIC | correct peer |
|---|---:|---:|---|
| unidirectional (client `-w`, server `-r`) | 189.8 | **378.5** | `ppermute_uni` |
| bidirectional (`-rw`), per direction | 186.7 | **347.8** | `all_reduce`, `all_gather`, `ppermute_bidi` |

Gotcha: neper counts on the **read** side. In unidirectional mode the writer
logs `local_throughput=0` and the real figure is in `remote_throughput`.

Reference point: `google/dranet`'s `docs/user/gke-tpu-performance.md` publishes
180.17 + 174.73 = 354.9 Gbps on this machine type, consistent with our 347.8.

---

### Confirming the runs really were on two NICs

Not an assumption. For every run quoted here, the env dump records the
interfaces the Pod actually had, and `network_delta` records the bytes that
actually moved:

```
eth0   speed= 10000 Mbps  up     <- pod overlay veth
eth1   speed=200000 Mbps  up
eth2   speed=200000 Mbps  up
--megascale_grpc_interface_prefixes=eth1,eth2,lo
```

`all_reduce` TX split, per run:

| run | eth1 | eth2 | eth1 share | eth0 | predicted vs measured |
|---|---:|---:|---:|---:|---:|
| a-solo | 696.3 GiB | 707.5 GiB | 49.6% | 0.2 MB | 1.00 |
| c-ar-then-uni | 702.6 | 700.3 | 50.1% | 0.2 MB | 1.00 |
| e-bidi-then-ar | 692.2 | 711.4 | 49.3% | 0.1 MB | 1.00 |
| baseline-r1 | 568.1 | 634.5 | 47.2% | 0.1 MB | 1.00 |
| baseline-r3 | 621.6 | 581.2 | 51.7% | 0.1 MB | 1.00 |

47–52% split, measured total within 0.2% of
`shard_bytes × 4 devices × dispatches`, and only control traffic on `eth0`.

## The protocol we settled on

```
variants:      all_reduce ALONE          (§2)
warmups:       200                       (§3)
reps:          10-15, report best        (§5)
dim:           32768  (1 GiB/device)
NICs:          eth1 + eth2 verified inside the pod   (§1)
discard:       first run of each session (§4)
```

Gives 198.0 ± 6.0 Gbps with σ 7–8% within a run. Scripts: `optsweep.sh`,
`order-matrix.sh`, `nicbw/nicbw.sh`.

## What we verified instead of assuming

**Byte accounting, against NIC counters.** `benchmark.py` already samples
`/sys/class/net/*/statistics` around each variant. Per-variant `eth1+eth2` deltas
vs `shard_bytes(1 GiB) × 4 devices × 155 dispatches`:

| variant | predicted TX | measured TX | ratio | measured RX |
|---|---:|---:|---:|---:|
| `ppermute_uni` | 620.0 GiB | 621.3 GiB | 1.00 | **0.1 GiB** |
| `ppermute_bidi` | 620.0 GiB | 621.5 GiB | 1.00 | 624.8 GiB |
| `all_gather` | 620.0 GiB | 621.5 GiB | 1.00 | 624.9 GiB |
| `all_reduce` | 620.0 GiB | 621.5 GiB | 1.00 | 624.8 GiB |

This confirms all-reduce and ppermute really do move identical bytes, confirms
`ppermute_uni` is purely unidirectional, and validates the bandwidth formula.
Traffic also splits ~50/50 across the two NICs for every variant, so uneven NIC
use is not a factor.

**On the ring factor.** At DP=2, `2(n−1)/n = 1.0`, so `algorithm_GBps_*` and
`ring_equivalent_bus_GBps_*` are numerically identical — which is why the
counters match exactly. The factor assumes a ring; the HLO shows MegaScale does a
direct exchange instead. For n=2 both move the same bytes so it happens to be
right, but it would not necessarily hold at n>2.

**HLO.** `psum` over `dcn` lowers to a single pair of MegaScale host transfers
with the reduction folded in — the whole optimized ENTRY at dim 32768 is 8
instructions with **zero** `add`/`reduce`:

```
_xla_host_transfer_handler_name="xla_megascale_runtime"
_xla_megascale_transfer_type="ALL_REDUCE"
_xla_megascale_reduce_operation="SUM"
_xla_megascale_target="{1,2}x{0:3}"
```

versus `ppermute_uni`, which is `_xla_megascale_transfer_type="ONE_TO_ONE"`.
The full 1 GiB shard is one `send` / one `recv`; no chunking or pipelining is
visible.

## Where the clean numbers land

| | Gbps | of matched raw-TCP ceiling |
|---|---:|---:|
| `ppermute_uni` | 313–321 | 83% (of 378.5 uni) |
| `ppermute_bidi` | 237.5 | 68% (of 347.8 bidi/dir) |
| `all_gather` | 221.0 | 64% |
| **`all_reduce`** | **198.0** | **57%** |

All-reduce moves the same bytes as `ppermute` and runs at 57% of what raw TCP
does in the same direction. Doubling host NIC bandwidth from 200 to 400 Gbps
scales `ppermute` 1.52x but `all_reduce` only 1.34x.

## Configuration levers: there are none

We swept **20 MegaScale runtime flags** under the clean protocol above. Measured
noise floor first: baseline repeated 6x gives mean 192.1, sd 11.1, 2σ band
[169.9, 214.3]. Every one of the 20 results (166.7 – 206.4) fell inside that
band. The single-shot leaders were then re-run 3x interleaved with baseline:

| | r1 | r2 | r3 | mean | sd |
|---|---:|---:|---:|---:|---:|
| baseline | 200.3 | 191.1 | 202.5 | **198.0** | 6.0 |
| `megascale_grpc_use_chaotic_good=true` | 203.9 | 194.6 | 196.3 | 198.3 | 5.0 |
| `megascale_ring_threshold=0` | 205.0 | 203.8 | 187.9 | 198.9 | 9.6 |
| NUMA trio | 204.6 | 170.8 | 186.2 | 187.2 | 16.9 |

Parity. Tried and null: `use_top_level_all_gather_and_local_reduction_for_ar`,
`lower_all_reduce_to_all_gather_threshold` (0 and 16 GiB), `ring_threshold`
(0 and 16 GiB), `local_f32_accum_for_bf16_ar` (both), `eigen_threads_per_device`
(8, 64), `chunk_size` (64 MiB, 256 MiB), NUMA threadpool/transmit/work-stealing,
`grpc_num_channels` (8, 32), `enable_tpu_premapping`, `grpc_enable_multi_nic`,
`grpc_use_chaotic_good`, `grpc_dynamic_lb`.

Separately, four configurations from the public *EXTERNAL - XLA Flags* doc
(host send/recv concurrency; SparseCore all-reduce offload; that plus
`sparse_core_all_reduce_latency_multiplier=inf`; the combination) produced an
optimized HLO ENTRY that was **byte-identical to baseline in all four** — still
8 instructions, still 0 SparseCore references. SparseCore all-reduce offload is
accepted by libtpu on v6e without complaint and does nothing.

`megascale_transport_type=bamm`, which the internal MegaScale harness defaults
to, does not exist in Cloud libtpu 0.0.44 — `strings` shows only `grpc`,
`socket`, `rdma`.

## Known gaps

- The xprof trace we captured was taken with `ppermute_uni,all_reduce`, i.e. on a
  poisoned all-reduce (62 Gbps). Its time attribution (79% in `recv-done`)
  describes the poisoned state and needs re-taking under the clean protocol. The
  HLO finding is unaffected — that lowering is static.
- The mechanism behind §2 is not identified.
- Only DP=2 was measured.

## Files

| path | what |
|---|---|
| `dcn/k8s/dranet-claim.yaml`, `dcn/k8s/jobset-v6e-dranet.yaml` | §1, the DRA path |
| `dcn/v6e/results/ordermatrix/` | §2 raw data, 6 cases |
| `dcn/v6e/results/optsweep/` | flag sweep, `SUMMARY.txt` first |
| `dcn/v6e/results/nicbw/` | §6 neper ceiling, both directions |
| `dcn/v6e/results/metrics-{1nic,2nic-15rep}.jsonl` | the 1-NIC vs 2-NIC comparison in §1 |
| `dcn/v6e/scripts/` | `order-matrix.sh`, `optsweep.sh`, `nicbw.sh` + manifests |

Changes to the package itself, all backward compatible (defaults unchanged):

| file | change |
|---|---|
| `dcn/k8s/entrypoint.sh` | `DCN_DEVICES_PER_SLICE=8` → env-overridable; v6e needs 4 |
| `dcn/run_slice.sh` | `--participants-per-slice 8` → env-overridable; exposes `--participant-scan`; adds `DCN_PROFILE=1` |
| `dcn/distributed_runner.py` | wraps the run in `jax.profiler.trace` when `DCN_PROFILE=1` — `benchmark.py` parses `--profile-dir` but never uses it, and nothing in the package calls `jax.profiler`, so there was no way to get a trace |
| `dcn/k8s/dranet-claim.yaml`, `dcn/k8s/jobset-v6e-dranet.yaml` | new, see §1 |

`dcn/benchmark.py` is untouched — SHA256 still matches the value pinned in
`dcn/README.md`.

Full artifact set, public, no auth:
`https://storage.googleapis.com/yppublic/v6e-dcn-allreduce-20260826/`
