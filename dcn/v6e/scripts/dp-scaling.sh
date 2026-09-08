#!/usr/bin/env bash
# psum vs hand-written ring as DP grows. dim 32000 throughout so the ring's
# two-level split (dim/n, then /n again) stays integral at n=10.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/dpscale"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say(){ echo "[$(date +%T)] $*"; }
for N in ${DPS:-2 4 8 10}; do
  JS="dcn-dp$N-7x"; say "===== DP=$N"
  tok
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 36); do [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
  N=$N JS=$JS python3 - <<'PY'
import json,os
n=int(os.environ['N'])
d=json.load(open('/tmp/dp4.json'))
d['metadata']['name']=os.environ['JS']
rj=d['spec']['replicatedJobs'][0]; rj['replicas']=n
ps=rj['template']['spec']['template']; ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2]
import re
cmd=re.sub(r'export DCN_PROCESS_COUNT=\d+', f'export DCN_PROCESS_COUNT={n}', cmd)
cmd=re.sub(r'dp\d*_7x', f'dp{n}_7x', cmd)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_SLICES']={'name':'MA_SLICES','value':str(n)}
env['MA_DIMS']={'name':'MA_DIMS','value':'32000'}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,ring_ar,rs_ag'}
env['MA_WARMUPS']={'name':'MA_WARMUPS','value':'100'}
env['MA_REPS']={'name':'MA_REPS','value':'10'}
c['env']=list(env.values())
json.dump(d,open('/tmp/dpN.json','w'))
PY
  kubectl apply -f /tmp/dpN.json >/dev/null 2>&1
  for i in $(seq 1 60); do
    P=$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|grep 'slice-0'|awk '{print $1}'|head -1)
    [ -n "$P" ] && [ "$(kubectl logs $P -n $NS 2>/dev/null|grep -c MANUAL_AR_RESULT)" = 3 ] && break
    [ -n "$P" ] && kubectl logs $P -n $NS 2>/dev/null | grep -qE "Traceback|AssertionError" && break
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  kubectl logs "$P" -n $NS > "$OUT/dp$N.log" 2>&1
  python3 - "$OUT/dp$N.log" "$N" <<'PY'
import re,json,sys
s=open(sys.argv[1]).read()
pf=re.search(r'DCN_MULTISLICE_PREFLIGHT (\{.*)',s)
rows=[json.loads(m.group(1)) for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',s)]
if not rows:
    e=re.findall(r'(AssertionError[^\n]*|ValueError[^\n]*|Illegal value[^\n]*)',s)
    print(f"  DP={sys.argv[2]} FAILED {e[:1]}"); raise SystemExit
b={r['variant']:r for r in rows}
g=lambda v: b[v]['host_gbps_best'] if v in b else 0
p=json.loads(pf.group(1)) if pf else {}
print(f"  DP={sys.argv[2]:<3} devices={p.get('global_device_count','?'):<4} shard/dev={b['psum']['shard_bytes_per_device']/2**20:6.0f}MiB   "
      f"psum={g('psum'):6.1f}  ring_ar={g('ring_ar'):6.1f} ({g('ring_ar')/g('psum')-1:+.1%})  rs_ag={g('rs_ag'):6.1f}")
PY
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 36); do [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done
say "DP SCALING DONE"
