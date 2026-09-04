#!/usr/bin/env bash
# Flag sweep on tpu7x (2 x 2x2x1, dynamic slicing + Kueue), same clean protocol
# as v6e: all_reduce ALONE, warmups 200, dim 32768, best-of-N.
#
# Built by cloning the reference JobSet that already works on this cluster and
# changing only the flags, so the Kueue / slice plumbing is identical:
#   - JobSet label kueue.x-k8s.io/queue-name
#   - pod annotations cloud.google.com/gke-tpu-slice-topology + the three
#     kueue.x-k8s.io/podset-slice-* ones
#   - nodeSelector must NOT pin gke-tpu-topology, or TAS only sees already-carved
#     nodes and cannot admit two slices
#
# The reference command does `unset LIBTPU_INIT_ARGS`, so extra flags are
# exported after that line.
set -uo pipefail

NS=default
BASE_JSON="${BASE_JSON:-/tmp/dimsweep3.json}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/tpu7x/flagsweep"
export KUBECONFIG=~/.kube/gke-tpu-train-us-central1-1-prod.config

tok() { export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token); }
say() { echo "[$(date +%T)] $*"; }

CASES=(
"baseline|"
"aglocal|--megascale_use_top_level_all_gather_and_local_reduction_for_ar=true"
"ring0|--megascale_ring_threshold=0"
"chaotic|--megascale_grpc_use_chaotic_good=true"
"eigen64|--megascale_eigen_threads_per_device=64"
"evmgr-d2h|--megascale_use_dedicated_d2h_eventmanager=true"
"evmgr-all|--megascale_use_dedicated_eventmanager=true --megascale_use_dedicated_h2d_eventmanager=true --megascale_use_dedicated_d2h_eventmanager=true"
"dma64m|--megascale_target_dma_size=67108864"
"premap8g|--megascale_grpc_premap_memory_bytes=8589934592 --megascale_enable_tpu_premapping=true"
"preact|--megascale_preactivate_graphs=true"
"asynchc|--megascale_enable_async_host_commands=true"
"chunk64m|--megascale_chunk_size=67108864"
)

mkdir -p "$OUT"
WANT="${ONLY:-}"
for entry in "${CASES[@]}"; do
  NAME="${entry%%|*}"; FLAGS="${entry#*|}"
  if [ -n "$WANT" ]; then case " $WANT " in *" $NAME "*) ;; *) continue;; esac; fi
  JS="dcn-fs7x-$(echo "$NAME" | tr '_' '-')"
  say "===== $NAME   ${FLAGS:-<none>}"
  tok
  kubectl delete jobset "$JS" -n $NS --ignore-not-found --wait=true >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l jobset.sigs.k8s.io/jobset-name=$JS --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done

  JS="$JS" NAME="$NAME" FLAGS="$FLAGS" BASE_JSON="$BASE_JSON" python3 - <<'PY'
import json,os
d=json.load(open(os.environ['BASE_JSON']))
d['metadata']['name']=os.environ['JS']
ps=d['spec']['replicatedJobs'][0]['template']['spec']['template']
ps['metadata']['labels']['app']=os.environ['JS']
c=ps['spec']['containers'][0]
cmd=c['command'][2].replace('yp_dimsweep_0904','fs7x_'+os.environ['NAME'].replace('-','_'))
f=os.environ['FLAGS']
if f:
    cmd=cmd.replace('unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS',
                    'unset LIBTPU_INIT_ARGS GRPC_EXPERIMENTS\nexport LIBTPU_INIT_ARGS="%s"'%f,1)
c['command'][2]=cmd
env={e['name']:e for e in c['env']}
env['DCN_DIMS']['value']='32768'
env['DCN_VARIANTS']['value']='all_reduce'
env['DCN_WARMUPS']['value']='200'
env['DCN_REPS']['value']='10'
env['DCN_BATCH']['value']='10'
c['env']=list(env.values())
json.dump(d,open('/tmp/fs7x.json','w'))
PY
  kubectl apply -f /tmp/fs7x.json >/dev/null 2>&1

  SEL="jobset.sigs.k8s.io/jobset-name=$JS"
  for i in $(seq 1 60); do
    st=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null | awk '{printf "%s ",$3}')
    case "$st" in *Completed*|*Error*) break;; esac
    sleep 20
    [ $((i % 12)) = 0 ] && tok
  done
  P=$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null | grep 'slice-0' | awk '{print $1}' | head -1)
  kubectl logs "$P" -n $NS > "$OUT/$NAME.log" 2>&1
  python3 - "$OUT/$NAME.log" "$NAME" <<'PY'
import re,json,sys
s=open(sys.argv[1]).read()
m=list(re.finditer(r'DCN_GOOGLE_BASELINE_RESULT (\{.*)',s))
if not m:
    err=re.findall(r'(Illegal value[^\n"]*|Unknown flag[^\n"]*|Check failed[^\n]*)',s)
    print(f"  {sys.argv[2]}: FAILED  {err[:1]}")
else:
    r=json.loads(m[-1].group(1)); t=r['time_ms_all']
    ifc=re.search(r'--megascale_grpc_interface_prefixes=[^ \\"]*',s)
    print(f"  {sys.argv[2]}: {r['host_ring_equivalent_bus_GBps_best']*8:6.1f} Gbps  min={min(t):.0f}ms  sigma={(max(t)-min(t))/min(t):.0%}  [{ifc.group(0) if ifc else '?'}]")
PY
  kubectl delete jobset "$JS" -n $NS --ignore-not-found >/dev/null 2>&1
  for i in $(seq 1 24); do
    [ "$(kubectl get pods -n $NS -l "$SEL" --no-headers 2>/dev/null | wc -l)" = "0" ] && break
    sleep 5
  done
done
say "TPU7X FLAG SWEEP DONE"
