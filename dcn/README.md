# TPU7x DP2 DCN collective reproduction

This package reproduces the Google-style two-slice baseline for:

- unidirectional `ppermute([(0, 1)])`;
- bidirectional `ppermute([(0, 1), (1, 0)])`;
- tiled `all_gather` over `dcn`;
- `psum` all-reduce over `dcn`.

The required hardware shape is **two independent physical TPU7x `2x2x1`
slices**, not one `2x2x2` ICI slice. Each slice has four physical chips and
eight JAX devices. The global JAX mesh is `[dcn=2, ici=8]`, with input
sharding `P("dcn", None)`.

`benchmark.py` is byte-for-byte identical to
`benchmarks/ironwood/dcn_google_baseline.py` at tpu-microbenchmarks commit
`ccb9ab32250194a8e639a602099d583b9af63927`. Its SHA256 is:

```text
4c31f548fd32b59e6b85cd507c2e5c302919da8da6e02a8fc4c2c9e9ec342973
```

## Measurement contract

The sweep uses BF16 square matrices of dimensions 8192, 16384, 24576, and
32768. Each case has five warmups and five timed repetitions. A timed
repetition dispatches ten operations and waits once; global barriers bracket
the batch.

The reference medians are:

| Variant | Reference host bandwidth |
|---|---:|
| `ppermute_uni` | 364.726 Gbps |
| `ppermute_bidi` | 274.932 Gbps/direction |
| `all_gather` | 254.010 Gbps/direction |
| `all_reduce` | 172.084 Gbps/direction (ring-equivalent) |

The two transport cases and all-gather report transmitted bytes. All-reduce
reports input bytes times the ring-equivalent factor `2*(DP-1)/DP`; that factor
is exactly one for DP2.

### Instrumented timing and XLA flag A/B/A

`timing_benchmark.py` preserves the original benchmark and replaces only its
timing function. It emits JSON records from every rank with:

- one separately measured first call, including compilation and execution;
- a compiled warmup phase;
- dispatch, device-wait, collective-only, and trailing-barrier times;
- per-repetition `eth1` and `eth2` byte deltas.

The A/B/A runner also emits a `DCN_HLO_SUMMARY` after each block so that
collective-related tokens in optimized HLO can be compared across flags.

Use the paired critical-path analyzer instead of averaging rank-local medians.
For each repetition it takes the slower rank's collective-only time, excluding
the trailing host barrier:

```bash
python3 dcn/timing_results.py /tmp/rank-0.log /tmp/rank-1.log
python3 dcn/timing_results.py --json /tmp/rank-0.log /tmp/rank-1.log
```

The analyzer defaults to the dim-16384 all-reduce payload used here: 17.179869184
Gbit per host and a 400 Gbps aggregate link. Override
`--payload-gbits-per-host` and `--link-gbps` for a different workload.

## Falcon run

Requirements: authenticated `falcon`, `jq`, two available reservation-backed
TPU7x `2x2x1` slices in `tpu-training-antgroup-v2`, and access to the Falcon
artifact bucket.

```bash
./dcn/falcon/run.sh
```

Optional parameters:

```bash
DCN_FALCON_CLUSTER=my-cluster \
DCN_DIMS=8192,16384,24576,32768 \
DCN_VARIANTS=ppermute_uni,ppermute_bidi,all_gather,all_reduce \
DCN_WARMUPS=5 DCN_REPS=5 DCN_BATCH=10 \
./dcn/falcon/run.sh
```

The script performs the complete lifecycle: submit two holders, wait for both,
copy this package with `falcon exp cp`, launch both slices with the same JAX and
MegaScale coordinator, collect both artifacts, extract rank-0 JSONL, compare
with the reference, and run `operator-analysis` on the slice-0 artifact.
Outputs are written under `dcn/results/<run-id>/`.

If a local process is interrupted after both holders are submitted, reuse them
without allocating new TPU slices:

```bash
DCN_RUN_ID=<existing-run-id> \
DCN_EXP0=exp-... DCN_EXP1=exp-... \
./dcn/falcon/run.sh
```

For collection/analysis-only recovery after the workload has already finished:

```bash
DCN_RUN_ID=<existing-run-id> DCN_EXP0=exp-... DCN_EXP1=exp-... \
DCN_SKIP_LAUNCH=1 DCN_ANALYSIS_ID=an-... ./dcn/falcon/run.sh
```

## Kubernetes delivery for Google

### 1. Platform preflight

Confirm all of the following before applying a workload:

1. There are two distinct TPU7x physical `2x2x1` slices. A `2x2x1` slice is
   one VM with four physical chips and eight JAX devices.
2. GKE is at least `1.34.1-gke.1829001`; older releases have a TPU7x `2x2x1`
   chip-count admission bug.
3. Each benchmark container requests `google.com/tpu: 4`, not 8. TPU7x exposes
   two JAX devices per physical chip, but Kubernetes schedules physical chips.
4. Both high-throughput NICs are visible as `eth1` and `eth2`. The workload
   passes `--megascale_grpc_interface_prefixes=eth1,eth2,lo`.
5. TCP ports 1234 (JAX coordinator) and 8081 (MegaScale coordinator) are open
   between the two hosts.

Relevant Google documentation:

- [TPU7x topology and chip counts](https://docs.cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus)
- [TPU Multislice JobSet](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/tpu-multislice)
- [TPU7x 2x2x1 admission bug and fixed GKE version](https://docs.cloud.google.com/kubernetes-engine/docs/troubleshooting/known-issues#tpu7x-admission)
- [Dynamic TPU sub-slicing](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/create-dynamic-slices)
- [JobSet DNS, coordinator, and topology placement](https://jobset.sigs.k8s.io/docs/concepts/)

Google's conventional GKE Multislice flow historically required each slice to
be multi-host, while this benchmark deliberately uses two single-host `2x2x1`
slices. Use one of the following supported deployment arrangements and have the
cluster owner confirm that MegaScale/DCN is enabled across the two slices:

- two explicitly selected static node pools (`pod-pair.yaml`);
- JobSet over two pre-created static node pools (`jobset-static.yaml`);
- GKE dynamic sub-slicing with JobSet (`jobset.yaml`), which requires GKE
  `1.36.0-gke.3712000` or later and JobSet 0.12.0 or later.

### 2. Build the self-contained image

From the repository root:

```bash
export IMAGE=REGION-docker.pkg.dev/PROJECT/REPOSITORY/dcn-google-baseline:ccb9ab3
docker build -f dcn/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
```

The image pins JAX/JAXLIB 0.11.0 and libtpu 0.0.44. No Falcon component is
included or required.

### 3A. JobSet on dynamic `2x2x1` sub-slices

This manifest targets a Kueue TAS dynamic-slicing cluster. The Pod annotation
requests `2x2x1`; do not add a fixed `cloud.google.com/gke-tpu-topology`
selector or a JobSet `exclusive-topology` annotation. Kueue admission owns the
partition selection and injects the required topology affinity. The template
uses `hostNetwork: true`, exposes `eth1` and `eth2`, and makes no DRA claim.

```bash
./dcn/k8s/render.sh jobset-dynamic "$IMAGE" /tmp/dcn-jobset.yaml
kubectl apply -f /tmp/dcn-jobset.yaml
kubectl get jobset,pods -w
kubectl wait --for=condition=Completed jobset/dcn-google-baseline --timeout=2h
./dcn/k8s/collect.sh jobset /tmp/dcn-metrics.jsonl
```

To reproduce the XLA-requested measurement protocol (isolated first call,
10,000 compiled warmups, then 20 paired batch-10 samples), render the dedicated
instrumented template:

```bash
./dcn/k8s/render.sh jobset-dynamic-hostnetwork-10k "$IMAGE" \
  /tmp/dcn-dynamic-hostnetwork-10k.yaml
kubectl apply -f /tmp/dcn-dynamic-hostnetwork-10k.yaml
kubectl wait --for=condition=Completed \
  jobset/dcn-dyn-hn-10k --timeout=2h
kubectl logs -l app=dcn-dyn-hn-10k \
  --all-containers=true --prefix=true --tail=-1 > /tmp/dcn-dynamic-10k.log
python3 dcn/timing_results.py /tmp/dcn-dynamic-10k.log
```

The Kueue local queue is named `default` in the supplied template. Change the
`kueue.x-k8s.io/queue-name` label if the target cluster uses a different queue.

Google also recommends increasing the TCP receive-buffer maximum. The supplied
TPU7x adaptation preserves the upstream value and logs the old and new values:

```bash
kubectl apply -f dcn/k8s/tpu7x-increase-rmem.yaml
kubectl rollout status daemonset/tcp-increase-rmem -n kube-system
kubectl logs -n kube-system -l k8s-app=tcp-increase-rmem \
  -c tcp-increase-rmem --prefix=true
```

The source is GoogleCloudPlatform/ai-on-gke
`scripts/network-setup/v6e-increase-rmem.yaml` at commit
`51bf3dcab6ff658cf62cc32867f96860bf58dfdc`; only the accelerator selector was
changed from TPU-v6e to TPU7x, with before/after logging added.

### 3B. JobSet on pre-created static `2x2x1` node pools

```bash
./dcn/k8s/render.sh jobset-static "$IMAGE" /tmp/dcn-jobset.yaml
kubectl apply -f /tmp/dcn-jobset.yaml
kubectl get jobset,pods -w
kubectl wait --for=condition=Completed jobset/dcn-google-baseline --timeout=2h
./dcn/k8s/collect.sh jobset /tmp/dcn-metrics.jsonl
```

For the static template, the JobSet exclusive-topology annotation must assign
its two replicated Jobs to different node pools. For the dynamic template,
Kueue TAS must assign distinct slice/partition IDs. Verify the selected nodes
before trusting a result:

```bash
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=dcn-google-baseline \
  -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName,SLICE:.metadata.labels.jobset\.sigs\.k8s\.io/job-index
```

The manifests declare slice 0 / pod 0 as the JobSet coordinator and consume
the controller-injected `jobset.sigs.k8s.io/coordinator` stable DNS label; no
Pod IP or generated Pod name is hard-coded.

### 3C. Host-network A/B/A for one libtpu flag

This mode targets two pre-created node pools that each expose `eth1` and `eth2`
through additional node networks. The Pods use `hostNetwork: true` and do not
request a DRA network resource. It runs baseline / candidate / baseline in
three fresh libtpu processes, with 10,000 warmups per block:

```bash
./dcn/k8s/render.sh jobset-hostnetwork-flag-aba "$IMAGE" \
  TPU_NODEPOOL_0 TPU_NODEPOOL_1 \
  sparse_core_collective_aggregator \
  '--xla_tpu_enable_sparse_core_collective_aggregator=true' \
  /tmp/dcn-xla-flag-aba.yaml
kubectl apply -f /tmp/dcn-xla-flag-aba.yaml
kubectl wait --for=condition=Completed jobset/dcn-xla-flag-aba --timeout=2h
kubectl logs -l jobset.sigs.k8s.io/jobset-name=dcn-xla-flag-aba \
  --all-containers=true --prefix=true --tail=-1 > /tmp/dcn-xla-flag-aba.log
python3 dcn/timing_results.py /tmp/dcn-xla-flag-aba.log
```

Test only one flag per rendered JobSet. Delete or rename the completed JobSet
before rendering the next candidate. The same template can test the DCN
all-reduce combiner threshold by changing the experiment and flag arguments:

```bash
./dcn/k8s/render.sh jobset-hostnetwork-flag-aba "$IMAGE" \
  TPU_NODEPOOL_0 TPU_NODEPOOL_1 \
  dcn_all_reduce_combiner_1g \
  '--xla_tpu_dcn_all_reduce_combiner_threshold_bytes=1073741824' \
  /tmp/dcn-combiner-flag-aba.yaml
```

### 3D. Two explicit Pods

Use this when Google gives you the names of two distinct static TPU node pools:

```bash
./dcn/k8s/render.sh pods "$IMAGE" TPU_NODEPOOL_0 TPU_NODEPOOL_1 /tmp/dcn-pods.yaml
kubectl apply -f /tmp/dcn-pods.yaml
kubectl get pods -l app=dcn-google-baseline -o wide -w
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/dcn-google-baseline-s0 --timeout=2h
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/dcn-google-baseline-s1 --timeout=2h
./dcn/k8s/collect.sh pods /tmp/dcn-metrics.jsonl
```

### 4. Validate the run

Rank 0 must first print a preflight equivalent to:

```text
process_count: 2
local_device_count: 8
global_device_count: 16
slice_counts: {0: 8, 1: 8}
```

The movement check must report:

```text
bidi block means: [2.0, 1.0]
uni block means:  [0.0, 1.0]
all-reduce means: [3.0, 3.0]
```

Do not compare bandwidth if either check differs. Extracted JSONL can be
reprocessed at any time with:

```bash
python3 dcn/results.py /tmp/dcn-metrics.jsonl
python3 dcn/results.py /tmp/dcn-metrics.jsonl --json
```

For failures, first inspect both ranks:

```bash
kubectl logs -l jobset.sigs.k8s.io/jobset-name=dcn-google-baseline \
  --all-containers=true --prefix=true --tail=-1
kubectl describe jobset dcn-google-baseline
kubectl get events --sort-by=.lastTimestamp
```
