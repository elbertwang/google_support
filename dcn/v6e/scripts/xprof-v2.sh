#!/usr/bin/env bash
# HLO dump + xprof trace for a handful of iterations, to see what actually
# overlaps. Two arms:
#   base   hostred=false                       -> ALL_TO_ALL then ALL_GATHER
#   p1k4   hostred=false + pipeline_chunks=4   -> same, split into 4 chunks
# P1 gave no throughput gain despite being confirmed active in HLO. If the four
# chunks serialise instead of pipelining, the trace will show it directly.
#
# Few iterations on purpose: warmups 15, reps 1, batch 5 -> 5 timed steps in the
# trace. A 200-warmup trace would be enormous and no more informative.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/xprof-ovl"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
HR=--xla_tpu_use_megascale_host_reduction=false
CASES=("ovl|$HR --megascale_allow_send_recv_overlap=true")
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; FLAGS="${entry#*|}"
  JS="dcn-xp-$NAME"; tok
  echo "[$(date +%T)] ===== $NAME"
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
cmd=c['command'][2].replace('manualar_7x','xp_'+os.environ['NAME'])
inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so -o /tmp/libtpu_v2.so\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\n')
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
patch=open('/tmp/profpatch.txt').read()
cmd=cmd.replace('export DCN_CODE_ROOT=/tmp/google_support',
                patch+'export DCN_CODE_ROOT=/tmp/google_support',1)
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
cmd=cmd.rstrip()+'\ntouch /tmp/PARKED\necho PARKED_OK\nsleep 2400\n'
c['command'][2]=cmd
env={e['name']:e for e in c['env']}

env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum'}
env['MA_WARMUPS']={'name':'MA_WARMUPS','value':'15'}
env['MA_REPS']={'name':'MA_REPS','value':'1'}
env['MA_BATCH']={'name':'MA_BATCH','value':'5'}
c['env']=list(env.values())
json.dump(d,open('/tmp/xp.json','w'))
PY
  kubectl apply -f /tmp/xp.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"; P=""
  for i in $(seq 1 45); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    [ -n "$P" ] && kubectl exec "$P" -n $NS -- test -f /tmp/PARKED 2>/dev/null && break
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  echo "     产物清单:"
  kubectl exec "$P" -n $NS -- sh -c 'R=$(ls -d /tmp/dcn-artifacts/*/rank-0 2>/dev/null|head -1)
    echo "       xprof: $(find /tmp/xprof -name "*.xplane.pb" 2>/dev/null|wc -l) 个 xplane"
    du -sh /tmp/xprof 2>/dev/null
    echo "       hlo:   $(ls $R/compiler/hlo 2>/dev/null|wc -l) 个文件"' 2>&1
  mkdir -p "$OUT/$NAME"
  kubectl exec "$P" -n $NS -- sh -c 'R=$(ls -d /tmp/dcn-artifacts/*/rank-0 2>/dev/null|head -1)
    tar czf /tmp/prof.tgz -C /tmp xprof -C $R compiler/hlo 2>/dev/null; echo ok' >/dev/null 2>&1
  kubectl cp "$NS/$P:/tmp/prof.tgz" "$OUT/$NAME/prof.tgz" >/dev/null 2>&1
  ls -lh "$OUT/$NAME/prof.tgz" 2>/dev/null | awk '{print "     已下载 "$5}'
  kubectl logs "$P" -n $NS > "$OUT/$NAME.log" 2>&1
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done
echo "XPROF V2 DONE"
