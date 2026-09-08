# gRPC flow-control knobs on tpu7x: defaults are already best, with one exception

Sourced from an internal write-up that gave the **default values** — which turned
out to matter more than the flag list, because three of our earlier tests were
pointing the wrong way.

## Defaults, and what that means for earlier tests

| flag | default | what we had tested | re-reading |
|---|---|---|---|
| `megascale_chunk_size` | 8 MiB | 4 / 64 / 256 MiB | 4 MiB is a *halving* (collapsed to 16.6 Gbps); 64/256 MiB are 8x/32x increases (nothing). Default is at or above the knee |
| `megascale_grpc_premap_memory_bytes` | 16 GiB | 8 GiB | we tested a *decrease* |
| `megascale_grpc_enable_memcpy_eliding` | **true** | set to true | no-op |
| `megascale_grpc_use_process_numa_local_interfaces_only` | **true** | set to true | no-op |
| `megascale_grpc_enable_multi_nic` | **true** | set to true | no-op |
| `megascale_grpc_num_channels` | 8 | 8 / 16 / 32 | — |
| `megascale_grpc_dynamic_lb_max_outstanding_bytes` | 8 MiB | never | tested below |
| `megascale_grpc_dynamic_lb_min_outstanding_rpcs` | 32 | never | tested below |

## Single shot, tpu7x DP=2 dim 32000, both variants in one process

| config | `psum` | `exchange_add` |
|---|---:|---:|
| baseline | 161.4 | **289.2** |
| `dynamic_lb_max_outstanding_bytes=64Mi` | 160.2 | 263.7 |
| `dynamic_lb_max_outstanding_bytes=256Mi` | 151.6 | 258.3 |
| `dynamic_lb_min_outstanding_rpcs=128` | 157.2 | 284.8 |
| `dynamic_lb=false` | 165.9 | 254.5 |
| `grpc_enable_rpc_receive_coalescing=true` | 152.5 | 270.3 |
| `premap_memory_bytes=32Gi` | 156.6 | 287.0 |
| **combo** (64Mi + rpcs128 + coalesce + chan32) | **176.9** | 280.4 |

Nothing beats the default on `exchange_add`. Raising the in-flight cap makes it
monotonically worse, so "the 8 MiB flow-control window is throttling a 1 GiB
transfer" is wrong — more bytes in flight just adds queueing.

## The combo survives confirmation, and it is a trade-off

Three rounds, interleaved:

| | baseline | combo | delta |
|---|---:|---:|---:|
| `psum` | 163.3 ± 6.6 | **179.6 ± 3.5** | **+10.0%** |
| `exchange_add` | 281.3 ± 25.4 | 243.0 ± 9.7 | **−13.6%** |

Every round agrees in both directions, and the combo has the tighter spread.
This is the **first flag configuration in 60+ tried that survives repetition**.

It is mechanistically interesting: the fused `ALL_REDUCE` path and the
`ONE_TO_ONE` transport path want **opposite** tuning. That is another independent
sign that they are different execution paths inside the runtime.

It is also the one result that is directly deployable for a GSPMD training job,
where the collective is compiler-generated and cannot be swapped but
`LIBTPU_INIT_ARGS` can be set.

Caveat on magnitude: even tuned, `psum` at 179.6 remains 35–56% behind the
hand-written ring at 243–281.

## Which sub-flag drives it? None of them — it is an interaction

Three rounds, interleaved, all four configurations in one session:

| | `psum` | vs baseline | `exchange_add` | vs baseline |
|---|---:|---:|---:|---:|
| baseline | 160.3 ± 2.9 | — | 275.2 ± 9.3 | — |
| `num_channels=32` | 165.5 ± 2.5 | **+3.3%** | 258.4 ± 42.6 | −6.1% |
| `num_channels=64` | 157.4 ± 5.0 | −1.8% | 262.0 ± 43.6 | −4.8% |
| `num_channels=32` + `min_outstanding_rpcs=128` | 163.9 ± 1.8 | +2.2% | 233.1 ± 23.0 | −15.3% |
| combo, all four (measured separately) | 179.6 ± 3.5 | **+10.0%** | 243.0 ± 9.7 | −13.6% |

`num_channels=32` alone recovers only a third of the combo's gain. The rest comes
from the other three together, even though each of them measured *negative* on
its own. The coherent reading: 32 channels need a larger per-channel in-flight
budget to stay fed. Raising the in-flight cap while still on 8 channels just adds
queueing, which is what the single-flag runs showed.

`num_channels=64` is worse than 32, so the knee is between.

Note also that raising the channel count blows up the variance on the transport
path — `exchange_add` sd goes from 9.3 to 42.6 (individual runs 306.0 / 223.6 /
245.8). More channels makes `ONE_TO_ONE` both slower and erratic.

## Recommendation

For a GSPMD training job where the all-reduce is compiler-generated and cannot be
replaced, the deployable setting is the full combination, not `num_channels`
alone:

```
--megascale_grpc_num_channels=32
--megascale_grpc_dynamic_lb_max_outstanding_bytes=67108864
--megascale_grpc_dynamic_lb_min_outstanding_rpcs=128
--grpc_enable_rpc_receive_coalescing=true
```

~+10% on the fused all-reduce, measured on tpu7x at DP=2, dim 32000, three
interleaved rounds. Two caveats:

1. It costs ~14% on `ppermute`-style point-to-point traffic, so it is the wrong
   setting if your critical collective is a permute or all-gather.
2. It is a microbenchmark result on an isolated collective. Whether it moves a
   real training step depends on how much of the all-reduce is already overlapped
   with backward compute — MaxText enables `xla_tpu_overlap_compute_collective_tc`
   and `xla_tpu_enable_async_collective_fusion` by default, and we have not
   measured a training step.
