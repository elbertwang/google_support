# What the fused all-reduce actually does, and how many rings a ring needs

Measured on tpu7x, DP=4 (4 x 2x2x1 slices, 32 devices), dim 32000, 488 MiB per
device. HLO dumps in `tpu7x/hlo-dp4/`.

## The fused `psum` builds no rings at all

Its entire optimized ENTRY, at DP=4, is eight instructions:

```hlo
ENTRY %main.0_spmd (param.1: bf16[8000,32000]) -> bf16[8000,32000] {
  %param.1   = bf16[8000,32000] parameter(0), sharding={devices=[4,1,8]<=[32] last_tile_dim_replicate}
  %after-all = token[] after-all(), frontend_attributes={MegascaleRendezvousKeyName="psum.7_0"}
  %bitcast   = bf16[250000,8,128] bitcast(%param.1)
  %send      = (...) send(%bitcast, %after-all), is_host_transfer=true,
               _xla_megascale_transfer_type="ALL_REDUCE"
               _xla_megascale_reduce_operation="SUM"
               _xla_megascale_target="{1,2,3,4}x{0:7}"
  %recv      = (...) recv(%after-all), is_host_transfer=true, same attributes
  %send-done, %recv-done
  ROOT %copy.3 = bf16[8000,32000] copy(%bitcast.1)
}
```

One `send`, one `recv`, the whole 488 MiB shard, one group descriptor covering
all 4 slices x 8 devices. Byte-for-byte the same structure as at DP=2 — only the
group string changes from `{1,2}x{0:3}` to `{1,2,3,4}x{0:7}`. There is **no
algorithmic decomposition visible at any n**: XLA hands the buffer to the
MegaScale runtime and receives the reduced buffer back.

## The hand-written ring builds exactly one

| module | variant | operand shape | transfer type | host-transfer instrs | total instrs |
|---|---|---|---|---:|---:|
| 0042 | `psum` | bf16[8000,32000] (whole shard) | `ALL_REDUCE` | **4** = 1 pair | 8 |
| 0046 | `ring_ar` | bf16[2000,32000] (one chunk) | `ONE_TO_ONE` | **24** = 6 pairs | 224 |
| 0054 | `ring2_ar` | bf16[1000,32000] | `ONE_TO_ONE` | **48** = 12 pairs | 361 |

`ring_ar` emits 2(n-1) = 6 sequential `collective-permute` steps at n=4, each a
pairwise `ONE_TO_ONE` transfer of a `S/n` chunk. That matches the algorithm
exactly. `ring2_ar` runs two counter-rotating rings on half the buffer each, so
12 steps.

## Two counter-rotating rings buy nothing

| variant | Gbps | vs `psum` |
|---|---:|---:|
| `psum` | 110.4 | — |
| `ring_ar` (1 ring) | 183.5 | +66.2% |
| `ring2_ar` (2 counter-rotating rings) | 185.9 | +68.3% |

+1.3% over the single ring, i.e. nothing. The reason is in the counters below:
one ring already saturates both the duplex and both NICs, so a second ring only
splits the same budget into more flows.

## Both NICs and both directions are already fully used — by everything

Per-run NIC counters, rank-0, across DP 2/4/8/10:

| variant | eth1 share of TX | RX / TX |
|---|---:|---:|
| `psum` | 49.7 – 51.2% | **1.00** |
| `ring_ar` | 49.7 – 50.0% | **1.00** |
| `ring2_ar` | 50.2% | **1.00** |

Every variant stripes evenly across `eth1`/`eth2` and moves as much in as out.
The striping is done by MegaScale from
`--megascale_grpc_interface_prefixes=eth1,eth2,lo` and is independent of which
collective is running. A ring is inherently full-duplex: rank i sends to i+1
while receiving from i-1, two different peers, both directions busy.

So "does it use the bidirectional bandwidth of both NICs" — yes, and that is not
where the difference between `psum` and the ring comes from.

## Where the difference does come from

At DP=4 both move **the same total bytes**: `psum` 1150 GiB TX, `ring_ar` 1150
GiB TX. `2(n-1)/n · S` for n=4 is 1.5S, and that is what both move. So the fused
runtime is *not* using a bytes-suboptimal algorithm internally — it moves what a
ring would move.

Same bytes, same NICs, same duplex utilisation, 110.4 vs 183.5 Gbps.

**The deficit is in execution, not in the algorithm.** That also explains two
earlier negative results: chunking `psum` by hand bought +1–3% because it just
produces N copies of the same monolithic op, and none of the 52 flag
configurations moved it because the flags do not reach whatever the runtime does
between `send` and `recv-done`.
