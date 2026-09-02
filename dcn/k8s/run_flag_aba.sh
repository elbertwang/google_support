#!/usr/bin/env bash
set -euo pipefail

: "${DCN_FLAG_ARGS:?set DCN_FLAG_ARGS to one libtpu flag under test}"
export DCN_SLICE_ID="${DCN_SLICE_ID:-${JOBSET_JOB_INDEX:-}}"
: "${DCN_SLICE_ID:?DCN_SLICE_ID or JOBSET_JOB_INDEX must be set}"
: "${DCN_COORDINATOR_HOST:?set DCN_COORDINATOR_HOST from the JobSet coordinator label}"

case "$DCN_SLICE_ID" in 0|1) ;; *) echo "slice id must be 0 or 1" >&2; exit 2;; esac

export DCN_PROCESS_ID="$DCN_SLICE_ID"
export DCN_PROCESS_COUNT=2
export DCN_DEVICES_PER_SLICE=8
export DCN_COORDINATOR_ADDRESS="${DCN_COORDINATOR_HOST}:1234"
export DCN_CODE_ROOT="${DCN_CODE_ROOT:-/workspace/google_support}"
export DCN_PYTHON="${DCN_PYTHON:-/opt/dcn-venv/bin/python}"
export DCN_SOURCE_BENCHMARK="$DCN_CODE_ROOT/dcn/benchmark.py"
export DCN_BENCHMARK_PATH="$DCN_CODE_ROOT/dcn/timing_benchmark.py"

artifact_base="${DCN_ARTIFACT_ROOT:-/tmp/dcn-artifacts}"
candidate_name="${DCN_FLAG_EXPERIMENT:-xla_flag_candidate}"
experiment_names=(xla_flag_baseline_start "$candidate_name" xla_flag_baseline_end)
experiment_args=("" "$DCN_FLAG_ARGS" "")

mkdir -p /tmp/tpu_logs/tmp /tmp/tpu_logs/pip-cache \
  /tmp/tpu_logs/uv-cache "$artifact_base"
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export UV_CACHE_DIR=/tmp/tpu_logs/uv-cache

python_version="$($DCN_PYTHON - <<'PY'
import importlib.metadata as md
print({name: md.version(name) for name in ("jax", "jaxlib", "libtpu")})
PY
)"
echo "DCN_VERSION_CHECK $python_version"

for index in "${!experiment_names[@]}"; do
  (
    export DCN_EXPERIMENT="${experiment_names[$index]}"
    export LIBTPU_INIT_ARGS="${experiment_args[$index]}"
    export DCN_ARTIFACT_ROOT="$artifact_base/$DCN_EXPERIMENT"
    export JAX_COMPILATION_CACHE_DIR="$DCN_ARTIFACT_ROOT/jax-compilation-cache"
    export FALCON_EXP_ID="${DCN_RUN_ID:-k8s-dcn-$DCN_EXPERIMENT}"
    unset GRPC_EXPERIMENTS
    mkdir -p "$DCN_ARTIFACT_ROOT" "$JAX_COMPILATION_CACHE_DIR"
    echo "DCN_EXPERIMENT_BEGIN $DCN_EXPERIMENT extra_args=$LIBTPU_INIT_ARGS"
    bash "$DCN_CODE_ROOT/dcn/run_slice.sh"
    "$DCN_PYTHON" "$DCN_CODE_ROOT/dcn/hlo_summary.py" \
      "$DCN_ARTIFACT_ROOT/rank-$DCN_PROCESS_ID/compiler/hlo" \
      --experiment "$DCN_EXPERIMENT" --rank "$DCN_PROCESS_ID"
    echo "DCN_EXPERIMENT_END $DCN_EXPERIMENT"
  )
done
