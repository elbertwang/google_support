#!/usr/bin/env bash
# P1 and P2 are both confirmed active in HLO (P2 collapses the two phases into
# one with a local add; P1 k=4 quadruples the send/recv count). Neither moved
# throughput at n=1. This run repeats all three arms 3x interleaved to say so
# with numbers, using the in-run exchange_add ratio to cancel session drift --
# the n=1 pass already showed a 6% swing that was pure drift.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/v2reps"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
HR=--xla_tpu_use_megascale_host_reduction=false
declare -A F=( [base]="$HR"
               [p2]="$HR --megascale_specialize_two_slice_allreduce=true"
               [p1k4]="$HR --megascale_rs_ag_pipeline_chunks=4" )
for r in 1 2 3; do
for ARM in base p2 p1k4; do
  JS="dcn-vr-$ARM-r$r"; tok
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
  JS="$JS" ARM="$ARM" FLAGS="${F[$ARM]}" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('manualar_7x','vr_'+os.environ['JS'].replace('-','_'))
inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so -o /tmp/libtpu_v2.so\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\n')
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
c['env']=list(env.values())
json.dump(d,open('/tmp/vr.json','w'))
PY
  kubectl apply -f /tmp/vr.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 70); do
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
if len(rows)<2: print(f"  {sys.argv[2]:<10} FAILED"); raise SystemExit
print(f"  {sys.argv[2]:<10} psum={rows['psum']:6.1f}  exch={rows['exchange_add']:6.1f}  ratio={rows['psum']/rows['exchange_add']:.3f}",flush=True)
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
print("\n  arm     psum(mean±sd)      exch(mean±sd)      psum/exch")
sd=lambda a: st.stdev(a) if len(a)>1 else 0.0
for arm in ['base','p2','p1k4']:
    v=d.get(arm,[])
    if not v: continue
    p=[x[0] for x in v]; e=[x[1] for x in v]; r=[x[0]/x[1] for x in v]
    print(f"  {arm:<7} {st.mean(p):6.1f} ± {sd(p):4.1f}      {st.mean(e):6.1f} ± {sd(e):4.1f}      {st.mean(r):.3f} ± {sd(r):.3f}")
b=d.get('base')
if b:
    br=st.mean([x[0]/x[1] for x in b])
    for arm in ['p2','p1k4']:
        v=d.get(arm)
        if v: print(f"  {arm} 相对 base 的比值变化: {(st.mean([x[0]/x[1] for x in v])/br-1)*100:+.1f}%")
PY
echo "V2 REPS DONE"
