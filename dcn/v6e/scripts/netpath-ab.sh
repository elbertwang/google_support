#!/usr/bin/env bash
# A/B the two ways of getting the second NIC into the Pod, on the same hardware:
#   dranet    resourceClaimTemplate 2-netdev, no hostNetwork
#   hostnet   hostNetwork: true, no claim
# Everything else identical: all_reduce alone (never after ppermute_uni),
# warmups 200, dim 32768, best-of-N. Interleaved so drift hits both equally.
set -uo pipefail

PROJECT=tpu-launchpad-playground
CLUSTER=dcnbw-mn-ew4
IMAGE=us-docker.pkg.dev/$PROJECT/yunpeng-image-repo/dcn-google-baseline:v6e-20260826
DIM=32768; WARMUPS=200; REPS=10
ROUNDS="${ROUNDS:-3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/netpath"
export KUBECONFIG="$HERE/../kubeconfigs/$CLUSTER.kubeconfig"

tok() { export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say() { echo "[$(date +%T)] $*"; }

run() {  # $1=path (dranet|hostnet)  $2=tag
  local path=$1 tag=$2 tmpl JS="dcn-np-$1"
  case $path in
    dranet)  tmpl="$HERE/jobset-v6e-flagsweep.yaml" ;;
    hostnet) tmpl="$HERE/jobset-v6e-hostnet.yaml" ;;
  esac
  tok
  kubectl delete jobset "$JS" --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done
  sed -e "s|__IMAGE__|$IMAGE|g" -e "s|__DIMS__|$DIM|g" -e "s|__VARIANTS__|all_reduce|g" \
      -e "s|__EXTRA_LIBTPU__||g" -e "s|v6e-dranet-2nic-hlo-20260826|netpath-$tag|g" \
      -e "s|name: dcn-flagsweep|name: $JS|" \
      -e "/name: DCN_WARMUPS/{n;s|value: \"5\"|value: \"$WARMUPS\"|;}" \
      -e "/name: DCN_REPS/{n;s|value: \"15\"|value: \"$REPS\"|;}" \
      "$tmpl" > "/tmp/np-$tag.yaml"
  kubectl apply -f "/tmp/np-$tag.yaml" >/dev/null 2>&1

  local SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 75); do
    n=0
    for p in $(kubectl get pods -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}' 2>/dev/null); do
      kubectl exec "$p" -- test -f /tmp/dcn-artifacts/DONE >/dev/null 2>&1 && n=$((n+1))
    done
    [ "$n" = "2" ] && break
    sleep 20
    [ $((i % 12)) = 0 ] && tok
  done
  local P
  P=$(kubectl get pods -l "$SEL,jobset.sigs.k8s.io/job-index=0" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  kubectl logs "$P" > "$OUT/$tag.log" 2>&1
  sed -n '/^DCN_METRICS_JSONL_BEGIN$/,/^DCN_METRICS_JSONL_END$/p' "$OUT/$tag.log" | sed '1d;$d' > "$OUT/$tag.jsonl"
  if [ ! -s "$OUT/$tag.jsonl" ]; then
    say "  $tag FAILED"
    kubectl get pods -l "$SEL" --no-headers 2>&1 | sed 's/^/      /'
    kubectl describe pod "$P" 2>&1 | sed -n '/^Events:/,$p' | head -6 | sed 's/^/      /'
  else
    say "  $tag  $(python3 -c "
import json;r=[json.loads(l) for l in open('$OUT/$tag.jsonl')][0];t=r['time_ms_all']
nd=r['network_delta'];a=nd.get('eth1',{}).get('tx_bytes',0);b=nd.get('eth2',{}).get('tx_bytes',0)
print(f\"{r['host_ring_equivalent_bus_GBps_best']*8:6.1f} Gbps  min={min(t):.0f}ms sigma={(max(t)-min(t))/min(t):.0%}  eth1/eth2={a/2**30:.0f}/{b/2**30:.0f} GiB\")")"
  fi
  kubectl delete jobset "$JS" --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -l "$SEL" --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done
}

mkdir -p "$OUT"
say "throwaway run (cold start)"; run dranet cold
for r in $(seq 1 "$ROUNDS"); do
  say "== round $r"
  run dranet  "dranet-r$r"
  run hostnet "hostnet-r$r"
done
say "NETPATH A/B DONE"
