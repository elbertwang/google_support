#!/usr/bin/env bash
# Capture an xprof trace of the DCN all-reduce vs ppermute comparison.
# Short config on purpose: a trace of the full 4x4 sweep would be enormous and
# we only need to attribute time inside one all_reduce and one ppermute.
set -uo pipefail

PROJECT=tpu-launchpad-playground
CLUSTER=dcnbw-mn-ew4
IMAGE=us-docker.pkg.dev/$PROJECT/yunpeng-image-repo/dcn-google-baseline:v6e-xprof-20260826
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/xprof-clean"
JS=dcn-xprof-clean
export KUBECONFIG="$HERE/../kubeconfigs/$CLUSTER.kubeconfig"

tok() { export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say() { echo "[$(date +%T)] $*"; }

tok
kubectl delete jobset "$JS" --ignore-not-found --wait=true >/dev/null 2>&1
sed -e "s|__IMAGE__|$IMAGE|g" \
    -e "s|__DIMS__|32768|g" \
    -e "s|__VARIANTS__|all_reduce|g" \
    -e "s|__EXTRA_LIBTPU__||g" \
    -e "s|v6e-dranet-2nic-hlo-20260826|xprof-20260826|g" \
    -e "s|name: dcn-flagsweep|name: $JS|" \
    -e 's|                value: "15"|                value: "3"|' \
    "$HERE/jobset-v6e-flagsweep.yaml" > /tmp/dcn-xprof.yaml
# turn the profiler on
python3 - <<'PY'
import pathlib
p=pathlib.Path('/tmp/dcn-xprof.yaml'); s=p.read_text()
s=s.replace('              - name: DCN_BATCH',
            '              - name: DCN_PROFILE\n                value: "1"\n              - name: DCN_BATCH')
p.write_text(s)
PY
grep -A1 "DCN_PROFILE\b" /tmp/dcn-xprof.yaml
kubectl apply -f /tmp/dcn-xprof.yaml >/dev/null 2>&1

SEL="jobset.sigs.k8s.io/jobset-name=$JS"
ok=0
for i in $(seq 1 60); do
  n=0
  for p in $(kubectl get pods -l "$SEL" -o jsonpath='{range .items[*]}{.metadata.name}{" "}{end}' 2>/dev/null); do
    kubectl exec "$p" -- test -f /tmp/dcn-artifacts/DONE >/dev/null 2>&1 && n=$((n+1))
  done
  say "  parked=$n/2 [$(kubectl get pods -l "$SEL" --no-headers 2>/dev/null | awk '{printf "%s ", $3}')]"
  [ "$n" = "2" ] && { ok=1; break; }
  sleep 20
  [ $((i % 12)) = 0 ] && tok
done

rm -rf "$OUT"; mkdir -p "$OUT"
for idx in 0 1; do
  P=$(kubectl get pods -l "$SEL,jobset.sigs.k8s.io/job-index=$idx" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
  kubectl logs "$P" > "$OUT/rank-$idx.log" 2>&1
  kubectl cp "default/$P:/tmp/dcn-artifacts/rank-$idx/xprof" "$OUT/rank-$idx" >/dev/null 2>&1
  say "rank $idx: $(du -sh "$OUT/rank-$idx" 2>/dev/null | cut -f1)  files=$(find "$OUT/rank-$idx" -type f 2>/dev/null | wc -l)"
done
grep -c DCN_XPROF_TRACE_DONE "$OUT/rank-0.log"
find "$OUT" -name "*.xplane.pb" -o -name "*.trace.json.gz" | head
kubectl delete jobset "$JS" --ignore-not-found >/dev/null 2>&1
say "XPROF DONE"
