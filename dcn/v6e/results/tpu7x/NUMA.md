# tpu7x is dual-NUMA, and every number we published was measured single-process

Verified on `gke-tpu-66ce10fa-mh9d` (tpu7x-standard-4t, prod cluster), not taken
on faith.

## 1. The topology claim holds on our hardware

```
Intel Xeon 8581C, 2 sockets, 224 vCPU, NUMA node0 = cpu 0-55,112-167
                                       NUMA node1 = cpu 56-111,168-223

TPU accelerators (vendor 0x1ae0, class 0xff0000):
  0000:00:03.0  0000:00:03.1  0000:00:04.0  0000:00:04.1   -> numa_node 0
  0000:c0:00.0  0000:c0:00.1  0000:c0:02.0  0000:c0:02.1   -> numa_node 1

NICs:
  eth0 -> numa_node 0      eth1 -> numa_node 0      eth2 -> numa_node 1
```

Two chips (four JAX devices) plus one 200 Gbps NIC per socket. Exactly the
layout the internal write-up describes.

**Consequence for our results.** Every tpu7x number in this directory was taken
with one container requesting `google.com/tpu: 4`, i.e. a single process driving
all 8 JAX devices. Half of its device-to-NIC paths necessarily cross the UPI
link. That is the degraded configuration.

Note this is invisible to the neper ceiling we measured (379.2 Gbps): neper is
plain TCP with 16 threads x 32 flows scheduled across all cores and never
touches the TPU-to-host-memory-to-NIC path.

## 2. The platform enforces the two-container form

A single container asking for half a slice is rejected outright:

```
admission webhook "mjobset.kb.io" denied the request: invalid jobset
"dcn-numa2-eth1" replicated job "slice": configuration results in 2 TPUs
requested, but must be exactly 4 TPUs (full utilization)
```

So `google.com/tpu: 2` only exists as *two* containers in one Pod summing to the
full slice. It is not an optional style — it is the only sub-slice shape GKE
allows.

## 3. GKE injects the topology for you

Probed a real two-container Pod. The device plugin does more than the internal
scripts assume:

| env | container `numa0` | container `numa1` |
|---|---|---|
| `TPU_VISIBLE_CHIPS` | `0,1` | `2,3` |
| `TPU_WORKER_ID` | 0 | 1 |
| `TPU_PROCESS_PORT` | 8471 | 8472 |
| `TPU_CHIPS_PER_HOST_BOUNDS` | `1,2,1` | `1,2,1` |
| `TPU_HOST_BOUNDS` | `2,1,1` | `2,1,1` |
| `TPU_PROCESS_ADDRESSES` | `...:8471,...:8472` | same |

`TPU_HOST_BOUNDS=2,1,1` means **GKE already presents one physical machine as two
logical hosts**. The manual `HOSTS_PER_SLICE = PHYSICAL_HOSTS * 2` and
`--deepsea_host_bounds` from the internal scripts are for the non-GKE path; here
the plugin has done it.

## 4. What GKE does *not* do — and this is the gap

```
numa0 cpuset: 0-223
numa1 cpuset: 0-223
```

**Neither container is CPU- or memory-pinned.** The device plugin partitions
chip *visibility* only. So the caller still has to supply:

- `numactl --cpunodebind=N --membind=N` per container
- `--megascale_transport_numa_node=N`
- distinct `--megascale_port` per container
- NIC IRQ / XPS affinity (the internal `network_settings.sh` step), which needs
  host-level tooling from a privileged hostNetwork container

Without those, splitting into two containers gets you chip affinity but leaves
threads and buffers free to land on the wrong socket.

## 5. Flag coverage check

`--megascale_transport_numa_node` was **never tested** in any of our four flag
rounds. What we did test was `use_numa_aware_threadpool`,
`grpc_enable_numa_aware_transmit` and `grpc_enable_numa_work_stealing` — related
but not the same thing, and all on a single-process layout where there is no
correct NUMA node to bind to. On v6e the flag is meaningless anyway:
`ct6e-standard-4t` is single-socket.

## 6. What this would change

The internal bug b/462225501 reports single-process ~128 Gbps aggregate against
~363 Gbps for 1-proc-per-NUMA on the same hardware. Our single-process
measurements sit between those: `psum` 163-168 and `ring_ar` 271-286 Gbps at
DP=2.

If the NUMA split lifts the floor, two of our conclusions need re-checking:

- the tpu7x-versus-v6e efficiency gap (45% versus 57% of the raw-TCP ceiling)
  may be an artefact of the layout rather than a property of the generation
- the `psum` versus hand-written-ring gap may narrow, widen, or hold — the ring
  issues 2(n-1) small pairwise transfers while `psum` issues one large one, and
  those two shapes need not respond to NUMA locality the same way

Everything measured on v6e is unaffected: `ct6e-standard-4t` has a single socket.
