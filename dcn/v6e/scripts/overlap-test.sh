#!/usr/bin/env bash
# The xprof trace shows send-done and recv-done are strictly serial under
# hostred=false: phase 1 spends 41.9ms in send-done and only then 63.3ms in
# recv-done; phase 2 the same. Union of op spans equals sum of op durations,
# i.e. zero overlap anywhere in the step.
#
# That is exactly what the original send_done->recv_done control dependency
# predicts. The v1 patch that removed it measured +1.6% -- but that was under
# host reduction ON, where psum lowers to a single fused ALL_REDUCE with a
# different structure. The combination that matters, hostred=false plus
# allow_send_recv_overlap=true, has never been tested.
#
# If the two overlap: phase1 max(41.9,63.3) instead of 105.2, phase2
# max(10.2,131.1) instead of 141.2 -> step 292.4ms -> ~240ms, about +22%.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/overlap"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
HR=--xla_tpu_use_megascale_host_reduction=false
declare -A F=( [base]="$HR"
               [ovl]="$HR --megascale_allow_send_recv_overlap=true" )
for r in 1 2 3; do
for ARM in base ovl; do
  JS="dcn-ov-$ARM-r$r"; tok
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
cmd=c['command'][2].replace('manualar_7x','ov_'+os.environ['JS'].replace('-','_'))
inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so -o /tmp/libtpu_v2.so\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\n')
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
c['env']=list(env.values())
json.dump(d,open('/tmp/ov.json','w'))
PY
  kubectl apply -f /tmp/ov.json >/dev/null 2>&1
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
if len(rows)<2: print(f"  {sys.argv[2]:<9} FAILED"); raise SystemExit
print(f"  {sys.argv[2]:<9} psum={rows['psum']:6.1f}  exch={rows['exchange_add']:6.1f}  ratio={rows['psum']/rows['exchange_add']:.3f}",flush=True)
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
print("\n  arm    psum(mean±sd)      exch(mean±sd)      psum/exch")
for arm in ['base','ovl']:
    v=d.get(arm,[])
    if not v: continue
    p=[x[0] for x in v]; e=[x[1] for x in v]; r=[x[0]/x[1] for x in v]
    print(f"  {arm:<6} {st.mean(p):6.1f} ± {sd(p):4.1f}      {st.mean(e):6.1f} ± {sd(e):4.1f}      {st.mean(r):.3f} ± {sd(r):.3f}")
if d.get('base') and d.get('ovl'):
    b=[x[0] for x in d['base']]; o=[x[0] for x in d['ovl']]
    print(f"  psum 提升 {(st.mean(o)/st.mean(b)-1)*100:+.1f}%   （trace 预测 +22%）")
PY
echo "OVERLAP DONE"
