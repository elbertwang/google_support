# A hand-written all-reduce beats the fused one by 15% on v6e and 66% on tpu7x

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

## Scope and caveats

- **DP=2 only.** The identity `psum == x + ppermute([(0,1),(1,0)])` holds for two
  participants. For n>2 the equivalent is reduce-scatter + all-gather
  (`jax.lax.psum_scatter` + `all_gather`); not tested, and it may not win.
- Numerics differ in association order from whatever the fused path does
  internally. Our checksum test passes at DP=2 but a real model should be
  validated.
- The v6e gain (+15%) is much smaller than the tpu7x gain (+66%).

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
