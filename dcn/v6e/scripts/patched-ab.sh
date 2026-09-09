#!/usr/bin/env bash
# Patched libtpu (v7x-dcn-fullduplex-poc-v1) vs stock 0.0.44, interleaved A/B.
#
# The patch removes send_d->AddControlDependencyTo(recv_d) in
# cross_slice_rewrites.cc, and the HLO dump shows it lands ONLY on the fused
# path: ALL_REDUCE/ALL_GATHER lose %send-done from control-predecessors while
# ONE_TO_ONE keeps it. So within a single process:
#   psum         -> patched code path   (the treatment)
#   exchange_add -> untouched code path (the control)
# A gain on psum with exchange_add flat cannot be explained by the CL-977903203
# vs 0.0.44 snapshot gap, which would move both.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/patched"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }

for r in 1 2 3; do
for ARM in stock patched; do
  JS="dcn-ab-$ARM-r$r"; tok
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
  JS="$JS" ARM="$ARM" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('manualar_7x','ab_'+os.environ['JS'].replace('-','_'))
if os.environ['ARM']=='patched':
    inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu.so'
         ' -o /tmp/libtpu_patched.so\nexport TPU_LIBRARY_PATH=/tmp/libtpu_patched.so\n')
    cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
c['env']=list(env.values())
json.dump(d,open('/tmp/ab.json','w'))
PY
  kubectl apply -f /tmp/ab.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 60); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    [ -n "$P" ] && [ "$(kubectl logs "$P" -n $NS 2>/dev/null|grep -c MANUAL_AR_RESULT)" = 2 ] && break
    case "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|awk '{printf "%s ",$3}')" in *Error*) break;; esac
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  kubectl logs "$P" -n $NS > "$OUT/$ARM-r$r.log" 2>&1
  python3 - "$OUT/$ARM-r$r.log" "r$r $ARM" <<'PY'
import re,json,sys
rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))
      for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',open(sys.argv[1]).read())}
f=lambda k: f"{rows[k]['host_gbps_best']:6.1f}" if k in rows else "     -"
print(f"  {sys.argv[2]:<12} psum={f('psum')}  exchange_add={f('exchange_add')}",flush=True)
PY
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done; done

python3 - "$OUT" <<'PY'
import re,json,sys,glob,statistics as st
d={}
for f in glob.glob(sys.argv[1]+'/*-r*.log'):
    arm=f.split('/')[-1].split('-r')[0]
    for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',open(f).read()):
        r=json.loads(m.group(1)); d.setdefault((arm,r['variant']),[]).append(r['host_gbps_best'])
print()
for v,label in [('psum','psum        (ALL_REDUCE, patch 生效)'),
                ('exchange_add','exchange_add(ONE_TO_ONE, 对照组)')]:
    s=d.get(('stock',v),[]); p=d.get(('patched',v),[])
    if not s or not p: continue
    ms,mp=st.mean(s),st.mean(p)
    sd=lambda a: st.stdev(a) if len(a)>1 else 0.0
    print(f"{label}")
    print(f"   stock   {' '.join(f'{x:6.1f}' for x in s)}  mean={ms:6.1f} sd={sd(s):5.1f}")
    print(f"   patched {' '.join(f'{x:6.1f}' for x in p)}  mean={mp:6.1f} sd={sd(p):5.1f}   {(mp/ms-1)*100:+.1f}%")
PY
echo "AB DONE"
