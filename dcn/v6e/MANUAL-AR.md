# A hand-written all-reduce beats the fused one: +15~66% at DP=2, +60% at DP=4

## The idea

The lowering lives inside `libtpu.so` and cannot be patched, and 46 flag
configurations across two platforms proved no external knob reaches it. But what
XLA is *asked* to emit is ours to choose. At DP=2 an all-reduce is exactly a
bidirectional exchange plus a local add:

```python
jax.lax.psum(x, "dcn")  ==  x + jax.lax.ppermute(x, "dcn", perm=[(0,1),(1,0)])
```

The right-hand side goes down the `ONE_TO_ONE` transport path instead of the
fused `ALL_REDUCE` host transfer. The local add is HBM-bound — a few ms against
a ~170-420 ms collective.

## Results

Identical bytes on the wire (NIC counters agree to 0.1%), identical results
(every variant checksums to the same value as `psum`), same clean protocol
(variant alone, warmups 200, dim 32768, best-of-10). Three rounds each with the
variant order rotated, so no ordering bias.

### v6e, 2 x ct6e-standard-4t

| variant | r1 | r2 | r3 | mean | sd | vs psum |
|---|---:|---:|---:|---:|---:|---:|
| `psum` | 190.7 | 193.8 | 197.2 | **193.9** | 3.3 | — |
| `exchange_add` | 214.7 | 231.6 | 223.7 | **223.3** | 8.5 | **+15.2%** |
| `chunked_exchange_add_4` | 216.8 | 222.8 | 223.8 | **221.2** | 3.8 | +14.1% |

### tpu7x, 2 x 2x2x1

| variant | r1 | r2 | r3 | mean | sd | vs psum |
|---|---:|---:|---:|---:|---:|---:|
| `psum` | 164.9 | 164.4 | 159.9 | **163.1** | **2.7** | — |
| `exchange_add` | 283.6 | 277.0 | 253.3 | **271.3** | 15.9 | **+66.4%** |
| `chunked_exchange_add_4` | 283.5 | 248.5 | 281.2 | **271.0** | 19.6 | +66.2% |

`psum` has sd 2.7 on tpu7x and the gap is +108 Gbps, roughly 40 sigma.

### Against the raw-TCP ceiling

| | ceiling (bidi/dir) | `psum` | `exchange_add` |
|---|---:|---:|---:|
| v6e | 347.8 | 193.9 = 56% | 223.3 = **64%** |
| tpu7x | 379.2 | 163.1 = 43% | 271.3 = **72%** |

## What this says about the fused path

Manual chunking of `psum` itself buys almost nothing (`chunked_psum_4` +3.2%,
`chunked_psum_8` +1.4% on v6e), and chunking the exchange form buys nothing over
the unchunked exchange form. So the deficit is not about message size or
pipelining granularity — it is the fused `ALL_REDUCE` host transfer itself.
The transport is capable; the fused collective does not use it.

That lines up with everything else we measured: the HLO shows one monolithic
send/recv with the SUM folded in and no `add` instruction, xprof shows 82% of a
clean `psum` sitting in `recv-done`, and a participant scan is flat in host terms
so one chip pair already saturates the path.

## Does it generalise past DP=2? Yes, but not the naive way

The identity `psum == x + ppermute([(0,1),(1,0)])` is specific to n=2. Extending
it naively — receive from each of the n-1 peers and add — costs `(n-1)*S` per
rank, while a ring costs `2(n-1)/n*S`. They coincide only at n=2; at n=4 the
naive form moves twice the bytes.

Measured four formulations. All checksum-verified against `psum`.

### DP=2, v6e

| variant | Gbps | vs `psum` | TX |
|---|---:|---:|---:|
| `psum` | 201.2 | — | 1203 GiB |
| `exchange_add` | 236.8 | +17.7% | 1203 |
| `ring_ar` (hand-written ring) | 221.1 | +9.9% | 1203 |
| `direct_add` | 236.0 | +17.3% | 1203 |
| `rs_ag` (`psum_scatter` + `all_gather`) | **158.8** | **−21.1%** | 1203 |

At n=2 the naive form and the ring move identical bytes, and the naive one wins
slightly because the ring adds orchestration for no benefit.

### DP=4, tpu7x — 4 x 2x2x1 dynamic slices, 32 devices

Three rounds, variant order rotated:

| variant | r1 | r2 | r3 | mean | sd | vs `psum` | TX |
|---|---:|---:|---:|---:|---:|---:|---:|
| `psum` | 117.5 | 118.7 | 118.8 | **118.3** | **0.7** | — | 1206 GiB |
| **`ring_ar`** | 180.8 | 202.0 | 183.9 | **188.9** | 11.4 | **+59.7%** | 1205 |
| `rs_ag` | 112.4 | 108.4 | 111.7 | 110.9 | 2.1 | −6.3% | 1206 |
| `direct_add` | 97.1 | | | 97.1 | | −17.9% | **2411** |

`psum` reproduces to sd 0.7, so the +70.6 Gbps gap is about 100 sigma.

Three things fall out:

1. **The rewrite generalises — use a ring.** `ring_ar` is +59.7% at DP=4 and
   +9.9% at DP=2, both bytes-optimal.
2. **The naive extension is a trap.** `direct_add` moves exactly 2x the bytes at
   n=4 (2411 vs 1205 GiB, matching 3S vs 1.5S) and loses 17.9%.
3. **`psum_scatter` + `all_gather` is also a trap** — −21% at DP=2 and −6% at
   DP=4. Those two fused MegaScale collectives have the same problem the fused
   all-reduce does; swapping one fused op for two does not help.

Note also that the fused `psum` degrades with scale — 163-168 Gbps at DP=2 down
to 118.3 at DP=4 — while `ring_ar` holds much better.

## Remaining caveats
- Numerics differ in association order from whatever the fused path does
  internally. Checksums pass at DP=2 and DP=4, but a real model should be
  validated.
- The v6e gain is much smaller than the tpu7x gain.
- DP=8 and above not measured. The ring cost model says it should hold, but that
  is a prediction, not a measurement.
- `ring_ar` is a straightforward textbook ring inside `shard_map`; it has not
  been tuned, overlapped with compute, or checked against a real training step.

## Reproduce

`manual_ar.py` loads `benchmark.py` through `runpy` so the pinned SHA is
untouched, and adds the variants. Mount it as a ConfigMap and point
`DCN_BENCHMARK_PATH` at it.

```bash
kubectl create configmap manual-ar --from-file=manual_ar.py=./manual_ar.py
# v6e
sed -e "s|__IMAGE__|$IMAGE|" -e "s|__DIMS__|32768|" \
    -e "s|__VARIANTS__|psum,exchange_add,chunked_exchange_add_4|" \
    jobset-manual-ar.yaml | kubectl apply -f -
```

Env: `MA_DIMS`, `MA_VARIANTS`, `MA_WARMUPS`, `MA_REPS`, `MA_BATCH`.
