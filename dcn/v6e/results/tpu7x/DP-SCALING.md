# psum vs a hand-written ring as DP grows (tpu7x, dim 32000)

Each DP=n uses n dynamically-carved 2x2x1 tpu7x slices, 8 JAX devices each.
`dim=32000` throughout because the ring splits the shard by n a second time, and
32768 is not divisible by 10. All variants checksum-verified against `psum` at
every n.

| DP | devices | shard/dev | `psum` | `ring_ar` | gain | `rs_ag` |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 977 MiB | 169.1 | **285.6** | **+68.8%** | 155.6 (−8%) |
| 4 | 32 | 488 MiB | 113.8 | **183.8** | **+61.5%** | 108.1 (−5%) |
| 8 | 64 | 244 MiB | 95.1 ± 1.8 | **131.2 ± 10.0** | **+37.9%** | 92.7 (−3%) |
| 10 | 80 | 195 MiB | 82.0 ± 1.4 | **124.3 ± 6.1** | **+51.6%** | 65.9 (−20%) |

DP=2 and DP=4 are single runs; DP=8 and DP=10 are three runs each.
DP=8 observations: psum 93.6 / 97.1 / 94.7, ring 120.0 / 139.4 / 134.2.
DP=10 observations: psum 81.1 / 81.3 / 83.6, ring 117.7 / 129.7 / 125.4.

## What holds

- **The ring wins at every scale tested**, by 38–69%. This is not a DP=2 trick.
- **`psum_scatter` + `all_gather` never wins** and degrades with scale
  (−3% at DP=8 to −20% at DP=10). The other fused MegaScale collectives carry
  the same defect as the fused all-reduce; swapping one for two does not help.
- **Both implementations lose absolute throughput as DP grows** — `psum`
  169 → 82 Gbps, ring 286 → 124 from DP=2 to DP=10. More slices means more
  cross-slice partners per host and smaller chunks.

## What is not resolved

The gain is not monotonic in n (+68.8, +61.5, +37.9, +51.6). With 1–3 runs per
point and per-run sd of 1.4–10, the ordering of the DP=8 and DP=10 gains is not
separated. The safe statement is "38–69% across DP 2–10", not a trend line.

## Why the ring, and why not the naive form

For n participants and a shard of S bytes per rank:

| | bytes on the wire per rank |
|---|---|
| ring (reduce-scatter + all-gather) | `2(n-1)/n · S` |
| naive (send a full copy to each of n-1 peers) | `(n-1) · S` |

The ratio is `n/2`: identical at n=2, 2x worse at n=4, 5x worse at n=10. That is
why the DP=2 form (`x + ppermute(x, [(0,1),(1,0)])`) happens to be optimal and
why it stops being so immediately after. Measured at DP=4: the naive form moved
2411 GiB against the ring's 1205, exactly 2x, and lost 17.9%.

The ring's cost is latency — `2(n-1)` sequential network steps — so it is the
right choice for bandwidth-dominated (large) messages and the wrong one for
small ones. Every step is a `ppermute`, which lowers to the fast MegaScale
`ONE_TO_ONE` transfer; the additions are local HBM work.

Implementation: `make_ring_ar` in `../scripts/manual_ar.py`.
