#!/usr/bin/env bash
# Hunt for a MegaScale runtime flag that speeds up DCN all-reduce on v6e.
#
# Clean protocol, learned the hard way:
#   - all_reduce ALONE. Running ppermute_uni first poisons it 5x (see ordermatrix/).
#   - warmups 200. At 5 the spread is 33%; at 200 it is 8% and saturated.
#   - dim 32768 only (1 GiB/device), the size the whole investigation is about.
# Clean reference on this cluster: 181-200 Gbps, sigma 7-8%.
#
# These flags are not in the public XLA flags doc; they came from `strings` on
# libtpu 0.0.44. Defaults are unknown, so booleans are tried both ways and
# numerics at more than one magnitude. libtpu hard-fails on an unknown flag name
# or an illegal value, so rows=0 means "rejected", not "no effect".
set -uo pipefail

PROJECT=tpu-launchpad-playground
CLUSTER=dcnbw-mn-ew4
IMAGE=us-docker.pkg.dev/$PROJECT/yunpeng-image-repo/dcn-google-baseline:v6e-20260826
DIM=32768
WARMUPS="${WARMUPS:-200}"
REPS="${REPS:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/optsweep"
export KUBECONFIG="$HERE/../kubeconfigs/$CLUSTER.kubeconfig"

tok() { export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say() { echo "[$(date +%T)] $*"; }

CASES=(
"baseline|"
"verbose|--megascale_verbose=2 --megascale_info=true"
"aglocal|--megascale_use_top_level_all_gather_and_local_reduction_for_ar=true"
"ar2ag-0|--megascale_lower_all_reduce_to_all_gather_threshold=0"
"ar2ag-big|--megascale_lower_all_reduce_to_all_gather_threshold=17179869184"
"ring-0|--megascale_ring_threshold=0"
"ring-big|--megascale_ring_threshold=17179869184"
"nof32acc|--megascale_local_f32_accum_for_bf16_ar=false"
"f32acc|--megascale_local_f32_accum_for_bf16_ar=true"
"eigen8|--megascale_eigen_threads_per_device=8"
"eigen64|--megascale_eigen_threads_per_device=64"
"chunk64m|--megascale_chunk_size=67108864"
"chunk256m|--megascale_chunk_size=268435456"
"numa|--megascale_use_numa_aware_threadpool=true --megascale_grpc_enable_numa_aware_transmit=true --megascale_grpc_enable_numa_work_stealing=true"
"chan8|--megascale_grpc_num_channels=8"
"chan32|--megascale_grpc_num_channels=32"
"premap|--megascale_enable_tpu_premapping=true"
"multinic|--megascale_grpc_enable_multi_nic=true"
"chaotic|--megascale_grpc_use_chaotic_good=true"
"streamsend|--megascale_grpc_dynamic_lb=true"
)

mkdir -p "$OUT"
WANT="${ONLY:-}"
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; FLAGS="${entry#*|}"
  if [ -n "$WANT" ]; then case " $WANT " in *" $NAME "*) ;; *) continue;; esac; fi
  JS="dcn-opt-$NAME"
  say "===== $NAME   ${FLAGS:-<none>}"
  tok
  kubectl delete jobset "$JS" --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 20); do
    [ "$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done
  rm -f "$OUT/$NAME.log" "$OUT/$NAME.jsonl"

  sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__DIMS__|$DIM|g" -e "s|__VARIANTS__|all_reduce|g" \
      -e "s|__EXTRA_LIBTPU__|$FLAGS|g" -e "s|v6e-dranet-2nic-hlo-20260826|opt-$NAME|g" \
      -e "s|name: dcn-flagsweep|name: $JS|" \
      -e "/name: DCN_WARMUPS/{n;s|value: \"5\"|value: \"$WARMUPS\"|;}" \
      -e "/name: DCN_REPS/{n;s|value: \"15\"|value: \"$REPS\"|;}" \
      "$HERE/jobset-v6e-flagsweep.yaml" > "/tmp/opt-$NAME.yaml"
  kubectl apply -f "/tmp/opt-$NAME.yaml" >/dev/null 2>&1

  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 75); do
    n=0
    for p in $(kubectl get pods -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}' 2>/dev/null); do
      kubectl exec "$p" -- test -f /tmp/dcn-artifacts/DONE >/dev/null 2>&1 && n=$((n+1))
    done
    [ "$n" = "2" ] && break
    sleep 20
    [ $((i % 12)) = 0 ] && tok
  done

  P=$(kubectl get pods -l "$SEL,jobset.sigs.k8s.io/job-index=0" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  kubectl logs "$P" > "$OUT/$NAME.log" 2>&1
  sed -n '/^DCN_METRICS_JSONL_BEGIN$/,/^DCN_METRICS_JSONL_END$/p' "$OUT/$NAME.log" | sed '1d;$d' > "$OUT/$NAME.jsonl"
  if [ "$(wc -l < "$OUT/$NAME.jsonl")" = "0" ]; then
    say "  REJECTED/FAILED: $(grep -ioE 'Illegal value[^\"]*|Unknown flag[^\"]*|retired flag [^ ]*' "$OUT/$NAME.log" | head -1)"
  else
    say "  $(python3 -c "
import json;r=json.load(open('$OUT/$NAME.jsonl'))if 0 else [json.loads(l) for l in open('$OUT/$NAME.jsonl')][0]
t=r['time_ms_all'];print(f\"{r['host_ring_equivalent_bus_GBps_best']*8:7.1f} Gbps  min={min(t):.0f}ms  sigma={(max(t)-min(t))/min(t):.0%}\")")"
  fi
  kubectl delete jobset "$JS" --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 20); do
    [ "$(kubectl get pods -l "$SEL" --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done
done
say "OPT SWEEP DONE"
