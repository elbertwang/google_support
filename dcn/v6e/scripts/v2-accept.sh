#!/usr/bin/env bash
# Acceptance run for libtpu_v2.so (P1/P2/P3/P5, all behind default-off flags).
#
# Gate 1 comes first and is a hard stop. The whole value of the default-off
# design is that the same binary is its own control; if the binary with every
# new flag at its default does not reproduce the stock numbers, that property is
# gone and no later delta can be attributed. So Gate 1 failing aborts the run.
#
# Every job measures psum AND exchange_add. exchange_add is the in-process
# control: P1/P2 touch only the fused all-reduce lowering, so exchange_add must
# stay put. If it moves, something global changed and the psum delta is suspect.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/v2"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }

SHA=ec252605e2a56e7c77c1ec53b48cb8d5a68c374732c27025cfc69be648998a17
HR=--xla_tpu_use_megascale_host_reduction=false

#      name              flags
CASES=(
"G1a-default|"
"G1b-hostred|$HR"
"G2-P2-specialize|$HR --megascale_specialize_two_slice_allreduce=true"
"G3-P1-k4|$HR --megascale_rs_ag_pipeline_chunks=4"
)
WANT="${ONLY:-}"
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; FLAGS="${entry#*|}"
  if [ -n "$WANT" ]; then case " $WANT " in *" $NAME "*) ;; *) continue;; esac; fi
  JS="dcn-v2-$(echo "$NAME"|tr '[:upper:]' '[:lower:]')"; tok
  echo "[$(date +%T)] ===== $NAME   ${FLAGS:-<全 flag 默认>}"
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done

  JS="$JS" NAME="$NAME" FLAGS="$FLAGS" SHA="$SHA" python3 - <<'PY'
import json,os
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('manualar_7x','v2_'+os.environ['NAME'].replace('-','_'))
inj=('\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so'
     ' -o /tmp/libtpu_v2.so\n'
     'echo "%s  /tmp/libtpu_v2.so" | sha256sum -c - || { echo "SHA_MISMATCH"; exit 9; }\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\n' % os.environ['SHA'])
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)
f=os.environ['FLAGS']
if f:
    cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                    'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%f,1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['MA_VARIANTS']={'name':'MA_VARIANTS','value':'psum,exchange_add'}
c['env']=list(env.values())
json.dump(d,open('/tmp/v2.json','w'))
PY
  kubectl apply -f /tmp/v2.json >/dev/null 2>&1
  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 90); do
    P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
    if [ -n "$P" ]; then
      if [ ! -f "$OUT/$NAME.mapped" ]; then
        kubectl exec "$P" -n $NS -- sh -c 'for p in /proc/[0-9]*/maps; do grep -ho "/[^ ]*libtpu[^ ]*\.so" $p 2>/dev/null; done|sort -u' \
          > "$OUT/$NAME.mapped" 2>/dev/null || rm -f "$OUT/$NAME.mapped"
      fi
      [ "$(kubectl logs "$P" -n $NS 2>/dev/null|grep -c MANUAL_AR_RESULT)" = 2 ] && break
    fi
    st=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|awk '{printf "%s ",$3}')
    case "$st" in *OOMKilled*|*Error*|*CrashLoop*) echo "     [pod: $st]"; sleep 10; break;; esac
    sleep 20; [ $((i%12)) = 0 ] && tok
  done
  for s in 0 1; do
    PP=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep "slice-$s"|awk '{print $1}'|head -1)
    [ -n "$PP" ] && kubectl logs "$PP" -n $NS > "$OUT/$NAME-slice$s.log" 2>&1
  done
  python3 - "$OUT/$NAME-slice0.log" "$NAME" <<'PY'
import re,json,sys,os
p=sys.argv[1]
if not os.path.exists(p): print(f"  -> {sys.argv[2]:<18} NO-LOG"); raise SystemExit
s=open(p).read()
if 'SHA_MISMATCH' in s: print(f"  -> {sys.argv[2]:<18} SHA 校验失败"); raise SystemExit
mp=sys.argv[1].replace('-slice0.log','.mapped')
so=open(mp).read().split() if os.path.exists(mp) else ['?']
rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))['host_gbps_best']
      for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',s)}
if not rows:
    c='CRASH' if re.search(r'unrecoverable|SIGABRT|Aborting the coordinator',s) else 'NO-RESULT'
    err=re.findall(r'(Unknown flag[^\n"]*|Illegal value[^\n"]*)',s)
    print(f"  -> {sys.argv[2]:<18} {c} {err[:1]}")
else:
    f=lambda k: f"{rows[k]:6.1f}" if k in rows else "     -"
    print(f"  -> {sys.argv[2]:<18} psum={f('psum')}  exchange_add={f('exchange_add')}   so={so or '?'}")
PY
  # 第一道闸不过就停
  if [ "$NAME" = "G1a-default" ] || [ "$NAME" = "G1b-hostred" ]; then
    python3 - "$OUT/$NAME-slice0.log" "$NAME" <<'PY' || { echo "!!! 第一道闸未通过，中止。二进制不能自证对照，后续 delta 无法归因。"; exit 1; }
import re,json,sys,os
p,name=sys.argv[1],sys.argv[2]
if not os.path.exists(p): raise SystemExit(1)
s=open(p).read()
rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))['host_gbps_best']
      for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',s)}
if 'psum' not in rows: raise SystemExit(1)
lo,hi=(150,178) if name=='G1a-default' else (228,272)   # 基线 162±4 / 250±10，放宽到约 2.5 sigma
ok = lo <= rows['psum'] <= hi
print(f"     闸门判据 psum ∈ [{lo},{hi}]: {rows['psum']:.1f} -> {'通过' if ok else '未通过'}")
raise SystemExit(0 if ok else 1)
PY
  fi
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done
done
echo "参照: 默认 psum 162±3.6 / hostred 250±9.6 / exchange_add 275-290"
echo "V2 ACCEPT DONE"
