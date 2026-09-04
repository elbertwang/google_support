# tpu7x (2 x 2x2x1, DP=2) vs v6e — same clean protocol
gke-tpu-train-us-central1-1-prod / us-central1-c, dynamic slicing + Kueue,
hostNetwork, eth1+eth2, --megascale_grpc_interface_prefixes=eth1,eth2,lo verified in the env dump.
all_reduce ALONE, warmups 200, reps 10-15, best-of-N.

## Reproducing the colleagues' number
dim 16384: ours 156.4 vs theirs 157.8 Gbps  -> 0.9% apart. Nothing wrong with their setup.
They used warmups=10000 (~18 min); 200 gives the same answer.

## Payload sweep (participants=8)
dim 16384  256 MiB/dev   156.4
dim 32768  1.00 GiB/dev  170.6
dim 40960  1.56 GiB/dev  157.3
dim 49152  2.25 GiB/dev  167.3
-> plateau 165 +/- 8. Bigger payload does not get you to 190.

## Participant scan @ dim 32768
participants=8 (4 chips)  161.0 Gbps host,  20.1 per device
participants=4 (2 chips)  166.1 Gbps host,  41.5 per device
participants=2 (1 chip)   168.4 Gbps host,  84.2 per device
-> host total is flat. One chip pair alone already hits the ceiling.
   Aggregation width is NOT the bottleneck.

## Raw TCP ceiling on tpu7x nodes (neper, hostNetwork)
bidirectional per direction: eth1 189.8, eth2 189.6, both 379.2
unidirectional both: 380.6
CAVEAT: measured between two free uncarved nodes, not necessarily the two nodes
the benchmark slices landed on.

## Platform comparison, same protocol
                 raw TCP bidi/dir   all_reduce   ratio
v6e   4 dev/slice     347.8            198.0      57%
tpu7x 8 dev/slice     379.2            170.6      45%

tpu7x has the faster fabric and the slower all-reduce.
190+ is a v6e number; it is not reachable on tpu7x here.
