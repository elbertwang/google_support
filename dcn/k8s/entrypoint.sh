#!/usr/bin/env bash
set -euo pipefail

ulimit -c 0
export DCN_SLICE_ID="${DCN_SLICE_ID:-${JOBSET_JOB_INDEX:-}}"
: "${DCN_SLICE_ID:?DCN_SLICE_ID or JOBSET_JOB_INDEX must be set}"
case "$DCN_SLICE_ID" in 0|1) ;; *) echo "slice id must be 0 or 1" >&2; exit 2;; esac

export DCN_PROCESS_ID="$DCN_SLICE_ID"
export DCN_PROCESS_COUNT=2
export DCN_DEVICES_PER_SLICE="${DCN_DEVICES_PER_SLICE:-8}"
export DCN_ARTIFACT_ROOT="${DCN_ARTIFACT_ROOT:-/tmp/dcn-artifacts}"
if [ -z "${DCN_COORDINATOR_ADDRESS:-}" ]; then
  : "${DCN_COORDINATOR_HOST:?DCN_COORDINATOR_ADDRESS or DCN_COORDINATOR_HOST must be set}"
  export DCN_COORDINATOR_ADDRESS="${DCN_COORDINATOR_HOST}:1234"
fi
export FALCON_EXP_ID="${DCN_RUN_ID:-k8s-dcn-google-baseline}"

mkdir -p /tmp/tpu_logs/tmp /tmp/tpu_logs/pip-cache \
  /tmp/tpu_logs/uv-cache /tmp/tpu_logs/jax-compilation-cache "$DCN_ARTIFACT_ROOT"
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export JAX_COMPILATION_CACHE_DIR=/tmp/tpu_logs/jax-compilation-cache

python_version="$($DCN_PYTHON - <<'PY'
import importlib.metadata as md
print({name: md.version(name) for name in ("jax", "jaxlib", "libtpu")})
PY
)"
echo "DCN_VERSION_CHECK $python_version"

bash "$DCN_CODE_ROOT/dcn/run_slice.sh"

if [ "$DCN_SLICE_ID" = "0" ]; then
  metrics="$DCN_ARTIFACT_ROOT/rank-0/benchmark/metrics.jsonl"
  echo DCN_METRICS_JSONL_BEGIN
  cat "$metrics"
  echo DCN_METRICS_JSONL_END
  echo DCN_BASELINE_COMPARISON_BEGIN
  "$DCN_PYTHON" "$DCN_CODE_ROOT/dcn/results.py" "$metrics"
  echo DCN_BASELINE_COMPARISON_END
fi
