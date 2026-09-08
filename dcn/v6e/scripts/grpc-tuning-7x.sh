#!/usr/bin/env bash
# gRPC-over-TCP tuning flags on tpu7x, measured against BOTH psum and
# exchange_add.
#
# Rationale for testing both: manual_ar showed the same gRPC/TCP transport
# carries 271-283 Gbps when driven by ppermute versus 163 for psum, so TCP is
# not what limits all-reduce. But the transport itself still sits at 72% of the
# 379.2 Gbps raw-TCP ceiling, so if these flags do anything it should show up on
# exchange_add. psum runs in the same process as an internal control.
set -uo pipefail

NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/grpctune"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config

tok() { export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say() { echo "[$(date +%T)] $*"; }

CASES=(
"baseline|"
"chan16|--megascale_grpc_num_channels=16"
"numatx|--megascale_grpc_enable_numa_aware_transmit=true"
"eealloc|--megascale_grpc_use_event_engine_allocator=true"
"memcpyelide|--megascale_grpc_enable_memcpy_eliding=true"
"all4|--megascale_grpc_num_channels=16 --megascale_grpc_enable_numa_aware_transmit=true --megascale_grpc_use_event_engine_allocator=true --megascale_grpc_enable_memcpy_eliding=true"
)

mkdir -p "$OUT"
WANT="${ONLY:-}"
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; FLAGS="${entry#*|}"
  if [ -n "$WANT" ]; then case " $WANT " in *" $NAME "*) ;; *) continue;; esac; fi
  JS="dcn-gt-$NAME"
  say "===== $NAME   ${FLAGS:-<none>}"
  tok
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done

  JS="$JS" NAME="$NAME" FLAGS="$FLAGS" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('manualar_7x','gt_'+os.environ['NAME'])
f=os.environ['FLAGS']
if f:
    cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                    'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%f,1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
c['env']=list(env.values())
json.dump(d,open('/tmp/gt.json','w'))
PY
  kubectl apply -f /tmp/gt.json >/dev/null 2>&1

  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 50); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null | grep 'slice-0' | awk '{print $1}' | head -1)
    [ -n "$P" ] && [ "$(kubectl logs "$P" -n $NS 2>/dev/null | grep -c MANUAL_AR_RESULT)" = "2" ] && break
    st=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null | awk '{printf "%s ",$3}')
    case "$st" in *Error*) break;; esac
    sleep 20
    [ $((i % 12)) = 0 ] && tok
  done
  kubectl logs "$P" -n $NS > "$OUT/$NAME.log" 2>&1
  python3 - "$OUT/$NAME.log" "$NAME" <<'PY'
import re,json,sys
s=open(sys.argv[1]).read()
rows={json.loads(m.group(1))['variant']: json.loads(m.group(1))
      for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',s)}
if not rows:
    err=re.findall(r'(Illegal value[^\n"]*|Unknown flag[^\n"]*)',s)
    print(f"  {sys.argv[2]:<12} FAILED {err[:1]}")
else:
    p=rows.get('psum'); e=rows.get('exchange_add')
    f=lambda r: f"{r['host_gbps_best']:6.1f}" if r else "     -"
    print(f"  {sys.argv[2]:<12} psum={f(p)}  exchange_add={f(e)}")
PY
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done
done
say "GRPC TUNING SWEEP DONE"
