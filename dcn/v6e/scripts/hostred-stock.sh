#!/usr/bin/env bash
# Does --xla_tpu_use_megascale_host_reduction=false work on STOCK libtpu 0.0.44?
# If yes this is deployable today with no custom binary, and the patched .so is
# irrelevant to the win. Compares against the stock baseline measured on the same
# nodes an hour earlier: psum 162.1+-3.6, exchange_add 287.8+-10.9.
# The fp8 arm re-runs the compression flags and keeps BOTH slices' logs, since
# the failing worker was the peer and slice-0's log had no root cause.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/hostred"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }

RUNS=("hostred-r1|--xla_tpu_use_megascale_host_reduction=false"
      "hostred-r2|--xla_tpu_use_megascale_host_reduction=false"
      "hostred-r3|--xla_tpu_use_megascale_host_reduction=false"
      "fp8diag|--megascale_compression_threshold=0 --megascale_quantization_exponent_bits=5 --megascale_quantization_mantissa_bits=2")

for entry in "${RUNS[@]}"; do
  NAME="${entry%%|*}"; FLAGS="${entry#*|}"
  JS="dcn-hs-$NAME"; tok
  echo "[$(date +%T)] ===== $NAME  (stock libtpu)"
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
  JS="$JS" NAME="$NAME" FLAGS="$FLAGS" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
# NO TPU_LIBRARY_PATH override: stock pip libtpu 0.0.44
cmd=c['command'][2].replace('manualar_7x','hs_'+os.environ['NAME'].replace('-','_'))
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
c['env']=list(env.values())
json.dump(d,open('/tmp/hs.json','w'))
PY
  kubectl apply -f /tmp/hs.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 60); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    [ -n "$P" ] && [ "$(kubectl logs "$P" -n $NS 2>/dev/null|grep -c MANUAL_AR_RESULT)" = 2 ] && break
    case "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|awk '{printf "%s ",$3}')" in *Error*|*CrashLoop*) sleep 10; break;; esac
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  for s in 0 1; do
    PP=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep "slice-$s"|awk '{print $1}'|head -1)
    [ -n "$PP" ] && kubectl logs "$PP" -n $NS > "$OUT/$NAME-slice$s.log" 2>&1
  done
  python3 - "$OUT/$NAME-slice0.log" "$NAME" <<'PY'
import re,json,sys
s=open(sys.argv[1]).read()
rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))
      for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',s)}
if not rows: print(f"  -> {sys.argv[2]:<10} FAILED")
else:
    f=lambda k: f"{rows[k]['host_gbps_best']:6.1f}" if k in rows else "     -"
    print(f"  -> {sys.argv[2]:<10} psum={f('psum')}  exchange_add={f('exchange_add')}")
PY
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done
echo "stock baseline (same nodes, 1h earlier): psum 162.1+-3.6  exchange_add 287.8+-10.9"
echo "HOSTRED STOCK DONE"
