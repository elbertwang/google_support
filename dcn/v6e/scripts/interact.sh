#!/usr/bin/env bash
# Do the earlier conclusions survive now that host reduction is off?
#
# Everything before today was measured with host reduction ON, i.e. against a
# system whose real bottleneck was elsewhere. Two things need re-checking:
#
#   1. the +10% gRPC combo from FLOWCTRL.md. It tuned the transport while the
#      transport was NOT the limit. Now it is, so the gain could grow or vanish.
# The hand-written ring is deliberately not measured here: it is not a
# deployable option (the DP all-reduce is GSPMD-inserted, not hand-written), and
# exchange_add already serves as the in-run reference for how much headroom is
# left on the same transport.
# Variant order is fixed across arms (ordering effects are real here).
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/interact"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }

HR="--xla_tpu_use_megascale_host_reduction=false"
COMBO="--megascale_grpc_num_channels=32 --megascale_grpc_dynamic_lb_max_outstanding_bytes=67108864 --megascale_grpc_dynamic_lb_min_outstanding_rpcs=128 --grpc_enable_rpc_receive_coalescing=true"

for r in 1 2; do
for ARM in hr hrcombo; do
  case $ARM in
    hr)      FLAGS="$HR";;
    hrcombo) FLAGS="$HR $COMBO";;
  esac
  JS="dcn-ix-$ARM-r$r"; tok
  echo "[$(date +%T)] ===== $ARM r$r"
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
  JS="$JS" ARM="$ARM" FLAGS="$FLAGS" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('manualar_7x','ix_'+os.environ['JS'].replace('-','_'))
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
c['env']=list(env.values())
json.dump(d,open('/tmp/ix.json','w'))
PY
  kubectl apply -f /tmp/ix.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 70); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    [ -n "$P" ] && [ "$(kubectl logs "$P" -n $NS 2>/dev/null|grep -c MANUAL_AR_RESULT)" = 2 ] && break
    case "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|awk '{printf "%s ",$3}')" in *Error*|*CrashLoop*) sleep 10; break;; esac
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  kubectl logs "$P" -n $NS > "$OUT/$ARM-r$r.log" 2>&1
  python3 - "$OUT/$ARM-r$r.log" "$ARM r$r" <<'PY'
import re,json,sys
s=open(sys.argv[1]).read()
rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))
      for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',s)}
if not rows: print(f"  -> {sys.argv[2]:<12} FAILED")
else:
    f=lambda k: f"{rows[k]['host_gbps_best']:6.1f}" if k in rows else "     -"
    print(f"  -> {sys.argv[2]:<12} psum={f('psum')}  exchange_add={f('exchange_add')}")
PY
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done; done

python3 - "$OUT" <<'PY'
import re,json,glob,sys,statistics as st
d={}
for f in glob.glob(sys.argv[1]+'/*-r*.log'):
    arm=f.split('/')[-1].split('-r')[0]
    for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',open(f).read()):
        r=json.loads(m.group(1)); d.setdefault((arm,r['variant']),[]).append(r['host_gbps_best'])
print()
for v in ['psum','exchange_add']:
    a=d.get(('hr',v),[]); b=d.get(('hrcombo',v),[])
    if not a: continue
    ma=st.mean(a); mb=st.mean(b) if b else float('nan')
    print(f"  {v:<13} hostred={ma:6.1f} {a}   +combo={mb:6.1f} {b}   {(mb/ma-1)*100:+.1f}%")
print("\n  参考: 无 hostred 时 psum 162.1+-3.6, combo 约 +10%")
PY
echo "INTERACT DONE"
