#!/usr/bin/env bash
# Many configs in ONE pod.
#
# Previous rounds paid, per config: Kueue admission (minutes to half an hour),
# venv + pip install of jax/libtpu, and a 650 MB libtpu_v2 download. The actual
# measurement is the cheap part. Running the configs sequentially inside a
# single pod pays all of that once. It also interleaves naturally inside one
# TPU allocation, which removes the session-to-session drift that forced the
# in-run exchange_add ratio in earlier rounds.
#
# Coordinator port is varied per iteration so a lingering socket from the
# previous run cannot block the next one.
#
# What is being tested: Gemini traced the all-reduce path's 8 MiB cap to
# --megascale_max_reduction_shard_size (default 8 MiB), applied in GraphBuilder
# BEFORE ApplyChunking. Since ApplyChunking can only subdivide
# (n_chunks = max(transfer_size/chunk_size, 1)), raising the shard size alone
# should get undone by the still-8-MiB chunk_size. So mrss16 alone is predicted
# to do nothing, and mrss16+chunk16 is predicted to be the combination that
# actually moves psum to 16 MiB. Instrumentation reads the real chunk size out
# of every arm, which answers the structural question in one pass without
# needing statistics.
set -uo pipefail
NS=default
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/multiconf"; mkdir -p "$OUT"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config
tok(){ export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
JS="${JS:-dcn-mc}"
# 每行: 名字;flags   —— 两个 slice 必须跑完全相同的序列
CONFIGS="${CONFIGS:-\
base;--xla_tpu_use_megascale_host_reduction=false
mrss16;--xla_tpu_use_megascale_host_reduction=false --megascale_max_reduction_shard_size=16777216
mrss16_ck16;--xla_tpu_use_megascale_host_reduction=false --megascale_max_reduction_shard_size=16777216 --megascale_chunk_size=16777216
ck16;--xla_tpu_use_megascale_host_reduction=false --megascale_chunk_size=16777216}"
WARM="${WARM:-200}"; REPS="${REPS:-6}"; DIM="${DIM:-32768}"; PASSES="${PASSES:-1}"

tok
kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
for i in $(seq 1 24); do
  [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null|wc -l)" = 0 ] && break; sleep 5; done

JS="$JS" CONFIGS="$CONFIGS" WARM="$WARM" REPS="$REPS" DIM="$DIM" PASSES="$PASSES" STOCK="${STOCK:-}" python3 - <<'PY'
import json,os,base64
d=json.load(open('/tmp/manualar7x.json'))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2]

# 一次性准备：patch 版 libtpu + 打开 libtpu 的 INFO 日志（否则 instrumentation 静默）
stock = os.environ.get('STOCK','')=='1'
inj=('' if stock else
     '\ncurl -sL --retry 3 https://storage.googleapis.com/yppublic/custom_libtpu/libtpu_v2.so -o /tmp/libtpu_v2.so\n'
     'export TPU_LIBRARY_PATH=/tmp/libtpu_v2.so\n') + (
     'export TPU_STDERR_LOG_LEVEL=0\nexport TPU_MIN_LOG_LEVEL=0\n'
     'export GLOG_logtostderr=1\nexport GLOG_stderrthreshold=0\n')
cmd=cmd.replace('export DCN_SLICE_ID=',inj+'export DCN_SLICE_ID=',1)

cfg_b64=base64.b64encode(os.environ['CONFIGS'].encode()).decode()
loop = f'''
echo "{cfg_b64}" | base64 -d > /tmp/configs.txt
export MA_DIMS={os.environ['DIM']}
export MA_WARMUPS={os.environ['WARM']}
export MA_REPS={os.environ['REPS']}
export MA_BATCH=10
export MA_VARIANTS=psum,exchange_add
BASE_ART="$DCN_ARTIFACT_ROOT"
port=1300
for pass in $(seq 1 {os.environ['PASSES']}); do
  while IFS=';' read -r cname cflags; do
    [ -z "$cname" ] && continue
    port=$((port+1))
    export DCN_COORDINATOR_ADDRESS="${{DCN_COORDINATOR_HOST}}:$port"
    export DCN_ARTIFACT_ROOT="$BASE_ART/$cname-p$pass"
    mkdir -p "$DCN_ARTIFACT_ROOT"
    export LIBTPU_INIT_ARGS="$cflags"
    echo "MC_CONFIG_BEGIN name=$cname pass=$pass port=$port flags=$cflags"
    bash /tmp/google_support/dcn/run_slice.sh || echo "MC_CONFIG_FAILED $cname rc=$?"
    echo "MC_CONFIG_END name=$cname pass=$pass"
    sleep 8
  done < <(cat /tmp/configs.txt; echo)
done
echo "MC_ALL_DONE"
sleep 600
'''
assert cmd.rstrip().endswith('bash /tmp/google_support/dcn/run_slice.sh')
cmd = cmd.rstrip()[:-len('bash /tmp/google_support/dcn/run_slice.sh')] + loop
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
for k in ['MA_VARIANTS','MA_DIMS','MA_WARMUPS','MA_REPS','MA_BATCH']:
    env.pop(k,None)
c['env']=list(env.values())
json.dump(d,open('/tmp/mc.json','w'))
print("configs:"); print(os.environ['CONFIGS'])
PY

kubectl apply -f /tmp/mc.json >/dev/null 2>&1
SEL="jobset.sigs.k8s.io/jobset-name=$JS"
echo "[$(date +%T)] 已提交，等待准入…"
LAST=""
for i in $(seq 1 400); do
  tok >/dev/null 2>&1
  P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null|grep slice-0|awk '{print $1}'|head -1)
  if [ -n "$P" ]; then
    L=$(kubectl logs "$P" -n $NS 2>/dev/null | grep -cE "MC_CONFIG_END")
    if [ "$L" != "$LAST" ]; then echo "[$(date +%T)] 已完成 $L 个配置"; LAST="$L"; fi
    kubectl logs "$P" -n $NS 2>/dev/null | grep -q MC_ALL_DONE && break
  else
    [ $((i % 20)) = 0 ] && echo "[$(date +%T)] Kueue 队列中"
  fi
  sleep 20
done
kubectl logs "$P" -n $NS > "$OUT/run.log" 2>&1
python3 - "$OUT/run.log" <<'PY'
import re,json,sys,collections,statistics as st
s=open(sys.argv[1]).read()
blocks=re.split(r'MC_CONFIG_BEGIN name=(\S+) pass=(\S+)',s)
res=collections.defaultdict(list); chunks={}
for i in range(1,len(blocks),3):
    name,pas,body=blocks[i],blocks[i+1],blocks[i+2]
    rows={json.loads(m.group(1))['variant']:json.loads(m.group(1))['host_gbps_best']
          for m in re.finditer(r'MANUAL_AR_RESULT (\{.*)',body)}
    if len(rows)==2: res[name].append((rows['psum'],rows['exchange_add']))
    g=collections.defaultdict(collections.Counter)
    for m in re.finditer(r'MEGASCALE_TRANSPORT_INSTR\] SendRpc: key=([^|_.]+)[^ ]* size=(\d+)',body):
        g[m.group(1)][int(m.group(2))//1048576]+=1
    chunks[name]={k:sorted(c.items(),reverse=True)[:2] for k,c in g.items() if sum(c.values())>10}
sd=lambda a: st.stdev(a) if len(a)>1 else 0.0
print(f"\n{'配置':<14}{'psum':>18}{'exchange_add':>20}   实际 chunk (MiB×条)")
for name in res:
    v=res[name]; p=[x[0] for x in v]; e=[x[1] for x in v]
    ck=chunks.get(name,{})
    cs=' '.join(f"{k}={v2}" for k,v2 in sorted(ck.items()))
    print(f"{name:<14}{st.mean(p):8.1f} ± {sd(p):4.1f}{st.mean(e):12.1f} ± {sd(e):4.1f}   {cs}")
print("\n预测: mrss16 单独无效（chunk_size 仍 8 MiB 会切回去）；mrss16_ck16 才让 psum 变 16 MiB")
PY
kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
echo "MULTICONF DONE"
