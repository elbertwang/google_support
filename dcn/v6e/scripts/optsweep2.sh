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
OUT="$HERE/optsweep2"
export KUBECONFIG="$HERE/../kubeconfigs/$CLUSTER.kubeconfig"

tok() { export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say() { echo "[$(date +%T)] $*"; }

CASES=(
"baseline|"
# --- receive path: xprof says 82% of clean all_reduce sits in recv-done ---
"evmgr|--megascale_use_dedicated_eventmanager=true"
"evmgr-h2d|--megascale_use_dedicated_h2d_eventmanager=true"
"evmgr-d2h|--megascale_use_dedicated_d2h_eventmanager=true"
"evmgr-all|--megascale_use_dedicated_eventmanager=true --megascale_use_dedicated_h2d_eventmanager=true --megascale_use_dedicated_d2h_eventmanager=true"
# --- DMA / message sizing ---
"dma64m|--megascale_target_dma_size=67108864"
"dma4m|--megascale_target_dma_size=4194304"
"hiprio-dma|--megascale_host_command_high_priority_dma_max_size=1073741824"
"msg2g|--megascale_grpc_max_message_size=2147483648"
"premap8g|--megascale_grpc_premap_memory_bytes=8589934592 --megascale_enable_tpu_premapping=true"
"xfercount|--megascale_dcn_transfer_count_threshold=1"
# --- scheduling / async ---
"asynchc|--megascale_enable_async_host_commands=true"
"preact|--megascale_preactivate_graphs=true"
# --- fewer bytes on the wire: changes numerics, kept separate ---
"quant8|--megascale_quantization_exponent_bits=5 --megascale_quantization_mantissa_bits=2"
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
