#!/usr/bin/env bash
# Chunk size under hostred=false. Never tested in this configuration.
#
# The earlier sweep covered 4M / 64M / 256M and target_dma 4M / 64M, all with
# host reduction ON, and all looked slightly negative (152-159 vs 163 baseline).
# Those conclusions are void: with host reduction on, the host CPU sum was the
# bottleneck, so nothing about the transfer granularity could show up.
# 16M and 32M were never tested at all -- the old sweep jumped 4M -> 64M. P5
# instrumentation measured the current default at 8 MiB, so 16M/32M are the
# nearest unexplored neighbours.
#
# Instrumentation is on in every arm so we can confirm the chunk size actually
# changed, the same way HLO confirmed P1/P2 were live. A flag that is silently
# ignored looks exactly like a flag that does nothing.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/chunk"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
HR="--xla_tpu_use_megascale_host_reduction=false --megascale_enable_transport_instrumentation=true"
CASES=(
"base|$HR"
"chunk16m|$HR --megascale_chunk_size=16777216"
"chunk32m|$HR --megascale_chunk_size=33554432"
"chunk64m|$HR --megascale_chunk_size=67108864"
"dma32m|$HR --megascale_target_dma_size=33554432"
)
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; FLAGS="${entry#*|}"
  JS="dcn-ck-$NAME"; tok
  echo "[$(date +%T)] ===== $NAME"
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
  JS="$JS" FLAGS="$FLAGS" NAME="$NAME" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('manualar_7x','ck_'+os.environ['NAME'])
inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so -o /tmp/libtpu_v2.so\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\n'
     'export TPU_STDERR_LOG_LEVEL=0\nexport TPU_MIN_LOG_LEVEL=0\n'
     'export GLOG_logtostderr=1\nexport GLOG_stderrthreshold=0\n')
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
env['MA_WARMUPS']={'name':'MA_WARMUPS','value':'60'}
c['env']=list(env.values())
json.dump(d,open('/tmp/ck.json','w'))
PY
  kubectl apply -f /tmp/ck.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 70); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    [ -n "$P" ] && [ "$(kubectl logs "$P" -n $NS 2>/dev/null|grep -c MANUAL_AR_RESULT)" = 2 ] && break
    case "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|awk '{printf "%s ",$3}')" in
      *OOMKilled*|*Error*|*CrashLoop*) sleep 10; break;; esac
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  kubectl logs "$P" -n $NS > "$OUT/$NAME.log" 2>&1
  python3 - "$OUT/$NAME.log" "$NAME" <<'PY'
import re,json,sys,collections
s=open(sys.argv[1]).read()
rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))['host_gbps_best']
      for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',s)}
sz=collections.Counter(int(x) for x in re.findall(r'MEGASCALE_TRANSPORT_INSTR\] SendRpc:.* size=(\d+)',s))
big=[(k,v) for k,v in sz.most_common(3) if k>65536]
if len(rows)<2:
    print(f"  -> {sys.argv[2]:<9} FAILED"); raise SystemExit
print(f"  -> {sys.argv[2]:<9} psum={rows['psum']:6.1f}  exch={rows['exchange_add']:6.1f}  "
      f"ratio={rows['psum']/rows['exchange_add']:.3f}   实际chunk={[(k//1048576,v) for k,v in big]} MiB")
PY
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done
echo "参照: 默认 chunk = 8 MiB（P5 实测）"
echo "CHUNK SWEEP DONE"
