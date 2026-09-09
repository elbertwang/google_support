#!/usr/bin/env bash
# Test the BlockAllocator theory for ppermute_uni poisoning with a binary
# predicate, no compile needed.
#
# Claim: ppermute_uni pushes next_allocation_offset to the end of the premapped
# pool; if any allocation is still outstanding the offset never resets, the
# later all_reduce fails to get a premapped buffer, and FallbackAllocator
# silently drops to plain heap memory -- which cannot do zero-copy DMA.
#
# P5 instrumentation already prints memcpy_needed on the receive path. On a
# clean run it was false for 615/616 samples. If the theory holds, the poisoned
# run must show memcpy_needed=true. The instrumentation also prints the op key,
# so within the poisoned run we can split ppermute samples from all-reduce ones
# and see the flip happen inside a single process.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/poison"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
INSTR="--megascale_enable_transport_instrumentation=true"
CASES=("C1-clean|all_reduce" "C2-poisoned|ppermute_uni,all_reduce")
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; VARIANTS="${entry#*|}"
  JS="dcn-pi-$(echo "$NAME"|tr '[:upper:]' '[:lower:]')"; tok
  echo "[$(date +%T)] ===== $NAME  variants=$VARIANTS"
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
  JS="$JS" NAME="$NAME" VARIANTS="$VARIANTS" FLAGS="$INSTR" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/dimsweep3.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('yp_dimsweep_0904','pi_'+os.environ['NAME'].replace('-','_'))
inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so -o /tmp/libtpu_v2.so\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\n'
     'export TPU_STDERR_LOG_LEVEL=0\nexport TPU_MIN_LOG_LEVEL=0\n'
     'export GLOG_logtostderr=1\nexport GLOG_stderrthreshold=0\n')
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%os.environ['FLAGS'],1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['DCN_DIMS']['value']='16384'; env['DCN_VARIANTS']['value']=os.environ['VARIANTS']
env['DCN_WARMUPS']['value']='60'; env['DCN_REPS']['value']='5'; env['DCN_BATCH']['value']='10'
c['env']=list(env.values())
json.dump(d,open('/tmp/pi.json','w'))
PY
  kubectl apply -f /tmp/pi.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  N=$(echo "$VARIANTS"|tr ',' '\n'|wc -l)
  for i in $(seq 1 100); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    [ -n "$P" ] && [ "$(kubectl logs "$P" -n $NS 2>/dev/null|grep -c DCN_GOOGLE_BASELINE_RESULT)" -ge "$N" ] && break
    case "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|awk '{printf "%s ",$3}')" in
      *OOMKilled*|*Error*|*CrashLoop*) sleep 10; break;; esac
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  kubectl logs "$P" -n $NS > "$OUT/$NAME.log" 2>&1
  python3 - "$OUT/$NAME.log" "$NAME" <<'PY'
import re,json,sys,collections
s=open(sys.argv[1]).read()
res=[]
for m in re.finditer(r'DCN_GOOGLE_BASELINE_RESULT (\{.*)',s):
    r=json.loads(m.group(1))
    g=r.get('host_ring_equivalent_bus_GBps_best') or r.get('host_tx_GBps_best')
    res.append(f"{r.get('variant')}={g*8:.1f}")
print(f"  -> {sys.argv[2]:<12} {'  '.join(res)}")
# 按 op key 分组统计 memcpy_needed —— Gemini 理论的二值判据
g=collections.defaultdict(collections.Counter)
for m in re.finditer(r'MEGASCALE_TRANSPORT_INSTR\] Recv: key=([^|_.]+)[^ ]* .*memcpy_needed=(\w+)',s):
    g[m.group(1)][m.group(2)]+=1
if g:
    print("     Recv memcpy_needed（按 op 分组）:")
    for k,c in sorted(g.items()):
        tot=sum(c.values()); t=c.get('true',0)
        print(f"       {k:<16} true={t:<6} false={c.get('false',0):<6} 总={tot:<6} true占比={t/tot:.1%}")
else:
    print("     (无 instrumentation 采样)")
PY
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done
echo "判据: 若 BlockAllocator 理论成立，C2 的 all_reduce 相关 recv 必须出现 memcpy_needed=true"
echo "POISON INSTR DONE"
