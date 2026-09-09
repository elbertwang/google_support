#!/usr/bin/env bash
# Structural check before any more perf work: do P1/P2 change the emitted graph?
# Perf said "no effect", but that has two very different explanations -- the
# change does nothing, or the flag never reached the compiler. HLO separates
# them, and it is what settled the first patched libtpu.
# Cheap runs: one variant, small dim, 5 warmups. We only need it to compile.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/v2hlo"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
HR=--xla_tpu_use_megascale_host_reduction=false
CASES=("H1-hostred|$HR"
       "H2-P2|$HR --megascale_specialize_two_slice_allreduce=true"
       "H3-P1k4|$HR --megascale_rs_ag_pipeline_chunks=4")
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; FLAGS="${entry#*|}"
  JS="dcn-vh-$(echo "$NAME"|tr '[:upper:]' '[:lower:]')"; tok
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
tag='vh_'+os.environ['NAME'].replace('-','_')
cmd=c['command'][2].replace('manualar_7x',tag)
inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so -o /tmp/libtpu_v2.so\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\necho "LIBTPU_ARGS_SET=%s"\n' % os.environ['FLAGS'])
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
cmd=cmd.rstrip()+'\ntouch /tmp/PARKED\necho PARKED_OK\nsleep 1800\n'
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum'}
env['MA_DIMS']={'name':'MA_DIMS','value':'16384'}
env['MA_WARMUPS']={'name':'MA_WARMUPS','value':'5'}
env['MA_REPS']={'name':'MA_REPS','value':'2'}
c['env']=list(env.values())
json.dump(d,open('/tmp/vh.json','w'))
PY
  kubectl apply -f /tmp/vh.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"; P=""
  for i in $(seq 1 40); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    [ -n "$P" ] && kubectl exec "$P" -n $NS -- test -f /tmp/PARKED 2>/dev/null && break
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  echo -n "     实际加载: "
  kubectl exec "$P" -n $NS -- sh -c 'for p in /proc/[0-9]*/maps; do grep -ho "/[^ ]*libtpu[^ ]*\.so" $p 2>/dev/null; done|sort -u' 2>&1|tr '\n' ' '; echo
  kubectl exec "$P" -n $NS -- sh -c "
D=\$(ls -d /tmp/dcn-artifacts/*/rank-0/compiler/hlo 2>/dev/null|head -1)
echo \"     transfer 结构:\"
for f in \$(grep -l xla_megascale_runtime \$D/*before_buffer_assignment* 2>/dev/null); do
  tt=\$(grep -o '_xla_megascale_transfer_type=\"[A-Z_]*\"' \$f|sed 's/.*=\"//;s/\"//'|sort|uniq -c|tr '\n' ' ')
  ns=\$(grep -cE '= .*(send|recv)\(' \$f)
  na=\$(grep -cE '^ *%?[a-z0-9_.-]+ = .* (add|reduce)\(' \$f)
  echo \"       \$(basename \$f|cut -d. -f1)  [\$tt] send/recv指令=\$ns  add=\$na\"
done" 2>&1 | tail -8
  kubectl logs "$P" -n $NS > "$OUT/$NAME.log" 2>&1
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done
echo "V2 HLO DONE"
