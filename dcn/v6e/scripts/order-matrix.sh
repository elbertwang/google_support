#!/usr/bin/env bash
# DP=2 all-reduce: isolate the ordering effect.
#
# Same hardware, same payload (dim 32768 = 1 GiB/device), same warmups. The only
# thing that changes is which collectives run before all_reduce inside the same
# process. Observed so far: all_reduce alone ~200 Gbps, ppermute_uni then
# all_reduce ~33 Gbps. This matrix pins down what actually causes it.
set -uo pipefail

PROJECT=tpu-launchpad-playground
CLUSTER=dcnbw-mn-ew4
IMAGE=us-docker.pkg.dev/$PROJECT/yunpeng-image-repo/dcn-google-baseline:v6e-20260826
DIM=32768
WARMUPS="${WARMUPS:-200}"
REPS="${REPS:-15}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/ordermatrix"
export KUBECONFIG="$HERE/../kubeconfigs/$CLUSTER.kubeconfig"

tok() { export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say() { echo "[$(date +%T)] $*"; }

# label | variant list (order matters, benchmark runs them in this order)
CASES=(
"a-solo|all_reduce"
"b-uni-then-ar|ppermute_uni,all_reduce"
"c-ar-then-uni|all_reduce,ppermute_uni"
"d-all4|ppermute_uni,ppermute_bidi,all_gather,all_reduce"
"e-bidi-then-ar|ppermute_bidi,all_reduce"
"f-ag-then-ar|all_gather,all_reduce"
)

mkdir -p "$OUT"
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; VARS="${entry#*|}"
  JS="dcn-om-$NAME"
  say "===== $NAME   variants=$VARS"
  tok
  kubectl delete jobset "$JS" --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 20); do
    [ "$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done
  rm -f "$OUT/$NAME.log" "$OUT/$NAME.jsonl"

  sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__DIMS__|$DIM|g" -e "s|__VARIANTS__|$VARS|g" \
      -e "s|__EXTRA_LIBTPU__||g" -e "s|v6e-dranet-2nic-hlo-20260826|order-$NAME|g" \
      -e "s|name: dcn-flagsweep|name: $JS|" \
      -e "/name: DCN_WARMUPS/{n;s|value: \"5\"|value: \"$WARMUPS\"|;}" \
      -e "/name: DCN_REPS/{n;s|value: \"15\"|value: \"$REPS\"|;}" \
      "$HERE/jobset-v6e-flagsweep.yaml" > "/tmp/om-$NAME.yaml"
  kubectl apply -f "/tmp/om-$NAME.yaml" >/dev/null 2>&1

  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  ok=0
  for i in $(seq 1 90); do
    n=0
    for p in $(kubectl get pods -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}' 2>/dev/null); do
      kubectl exec "$p" -- test -f /tmp/dcn-artifacts/DONE >/dev/null 2>&1 && n=$((n+1))
    done
    [ "$n" = "2" ] && { ok=1; break; }
    sleep 20
    [ $((i % 12)) = 0 ] && tok
  done

  P=$(kubectl get pods -l "$SEL,jobset.sigs.k8s.io/job-index=0" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  kubectl logs "$P" > "$OUT/$NAME.log" 2>&1
  sed -n '/^DCN_METRICS_JSONL_BEGIN$/,/^DCN_METRICS_JSONL_END$/p' "$OUT/$NAME.log" | sed '1d;$d' > "$OUT/$NAME.jsonl"
  say "  rows=$(wc -l < "$OUT/$NAME.jsonl")"
  kubectl delete jobset "$JS" --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 20); do
    [ "$(kubectl get pods -l "$SEL" --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done
done
say "ORDER MATRIX DONE"
