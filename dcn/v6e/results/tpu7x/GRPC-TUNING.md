# gRPC-over-TCP tuning flags on tpu7x — measured, no effect

Four flags proposed as a short-term mitigation for a "TCP bottleneck", tested on
two 2x2x1 tpu7x slices, dim 32768, warmups 200, best-of-10. Each config measures
`psum` and `exchange_add` in the same process so the transport path has its own
control.

## Prior coverage before this run

| flag | previously tried? |
|---|---|
| `megascale_grpc_num_channels` | yes, at 8 and 32 on v6e (169.7 / 182.7, noise). 16 not tried |
| `megascale_grpc_enable_numa_aware_transmit` | only bundled with the numa threadpool trio; that bundle measured 187.2 ± 16.9 vs baseline 198.0, i.e. worse. Never alone |
| `megascale_grpc_use_event_engine_allocator` | **no** |
| `megascale_grpc_enable_memcpy_eliding` | **no** |

## Single shot

| config | psum | exchange_add |
|---|---:|---:|
| baseline | 156.9 | **292.9** |
| `grpc_num_channels=16` | 168.8 | 265.6 |
| `grpc_enable_numa_aware_transmit` | 154.6 | 283.5 |
| `grpc_use_event_engine_allocator` | 167.0 | 262.7 |
| `grpc_enable_memcpy_eliding` | 165.8 | 289.6 |
| all four | 176.6 | 295.4 |

Individually, `num_channels=16` and `use_event_engine_allocator` *hurt*
`exchange_add` (265.6 / 262.7 against 292.9). Only the four together were at or
above baseline on both variants.

## Interleaved 3x confirmation of the four together

| config | variant | r1 | r2 | r3 | mean | sd |
|---|---|---:|---:|---:|---:|---:|
| baseline | `psum` | 159.9 | 168.5 | 174.3 | 167.6 | 7.3 |
| all four | `psum` | 175.4 | 163.1 | 182.2 | 173.6 | 9.7 |
| baseline | `exchange_add` | 279.9 | 250.3 | 276.5 | 268.9 | 16.2 |
| all four | `exchange_add` | 273.4 | 271.8 | 284.8 | 276.6 | 7.1 |

+3.6% on psum and +2.9% on exchange_add, against standard deviations of 7.3 and
16.2. Not distinguishable from noise. The single-shot 176.6 was a high draw.

## Why TCP tuning cannot fix all-reduce

The premise does not hold for this workload. The *same* gRPC-over-TCP transport,
on the same two hosts, in the same process, moving the *same* bytes:

    psum          167.6 Gbps
    exchange_add  268.9 Gbps      +60%

Both go over gRPC/TCP on eth1+eth2. The transport is not the limit — the fused
`ALL_REDUCE` host transfer is. Tuning the transport moves neither number.

The lever that does work is the graph rewrite in `../MANUAL-AR.md`: at DP=2,
`x + ppermute(x, [(0,1),(1,0)])` instead of `psum`, which is +66% on tpu7x.

Running total across the whole investigation: **52 MegaScale flag configurations
on two platforms plus 4 XLA-doc configurations, none with a measurable effect.**
