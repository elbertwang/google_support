#!/usr/bin/env bash
# Confirm the chunk-size result. The screen gave exchange_add 275.5 -> 321.9 at
# 32 MiB, which would be the best number measured in this whole investigation --
# 85% of the 379.2 Gbps neper ceiling, against 73% at the default 8 MiB. It is
# n=1, and this session's spread on exchange_add is +-15, so it needs repeats.
#
# psum rides along but is NOT expected to move: instrumentation shows the flag
# only reaches ppermute (ONE_TO_ONE). The all-reduce path stays pinned at 8 MiB
# whatever the flag says. So here psum is the negative control -- if psum moves
# with it, something global changed and the exchange_add delta is suspect.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/chunkreps"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
HR="--xla_tpu_use_megascale_host_reduction=false"
declare -A F=( [base]="$HR" [c16]="$HR --megascale_chunk_size=16777216" [c32]="$HR --megascale_chunk_size=33554432" )
for r in 1 2 3; do
for ARM in base c16 c32; do
  JS="dcn-cr-$ARM-r$r"; tok
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
  JS="$JS" FLAGS="${F[$ARM]}" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('manualar_7x','cr_'+os.environ['JS'].replace('-','_'))
inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so -o /tmp/libtpu_v2.so\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\n')
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
c['env']=list(env.values())
json.dump(d,open('/tmp/cr.json','w'))
PY
  kubectl apply -f /tmp/cr.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 90); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    [ -n "$P" ] && [ "$(kubectl logs "$P" -n $NS 2>/dev/null|grep -c MANUAL_AR_RESULT)" = 2 ] && break
    case "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|awk '{printf "%s ",$3}')" in
      *OOMKilled*|*Error*|*CrashLoop*) sleep 10; break;; esac
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  kubectl logs "$P" -n $NS > "$OUT/$ARM-r$r.log" 2>&1
  python3 - "$OUT/$ARM-r$r.log" "$ARM r$r" <<'PY'
import re,json,sys
rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))['host_gbps_best']
      for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',open(sys.argv[1]).read())}
if len(rows)<2: print(f"  {sys.argv[2]:<9} FAILED"); raise SystemExit
print(f"  {sys.argv[2]:<9} psum={rows['psum']:6.1f}  exch={rows['exchange_add']:6.1f}",flush=True)
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
    rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))['host_gbps_best']
          for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',open(f).read())}
    if len(rows)==2: d.setdefault(arm,[]).append((rows['psum'],rows['exchange_add']))
sd=lambda a: st.stdev(a) if len(a)>1 else 0.0
print("\n  arm    psum(应不动)        exchange_add(目标)")
for a in ['base','c16','c32']:
    v=d.get(a,[])
    if not v: continue
    p=[x[0] for x in v]; e=[x[1] for x in v]
    print(f"  {a:<6} {st.mean(p):6.1f} ± {sd(p):4.1f}      {st.mean(e):6.1f} ± {sd(e):4.1f}")
if d.get('base'):
    be=st.mean([x[1] for x in d['base']])
    for a in ['c16','c32']:
        if d.get(a):
            m=st.mean([x[1] for x in d[a]])
            print(f"  {a} exchange_add: {(m/be-1)*100:+.1f}%   （占裸TCP天花板 379.2 的 {m/379.2:.0%}）")
PY
echo "CHUNK REPS DONE"
