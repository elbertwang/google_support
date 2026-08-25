#!/usr/bin/env bash
set -euo pipefail

status_file=/tmp/dcn-status
done_file=/tmp/dcn-done
rm -f "$status_file" "$done_file"
exec > >(tee /tmp/dcn-workload.log) 2>&1

finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$status_file"
  touch "$done_file"
  exit "$rc"
}
trap finish EXIT

test -f /tmp/dcn-ready
export DCN_ARTIFACT_ROOT="${ARTIFACT_LOCAL_DIR:?ARTIFACT_LOCAL_DIR must be set}"
export DCN_PYTHON=/tmp/dcn-venv/bin/python
export DCN_CODE_ROOT=/tmp/google_support
export TPU_MICROBENCH_SOURCE_COMMIT="${DCN_SOURCE_COMMIT:-working-tree}"
export FALCON_EXP_ID="${DCN_RUN_ID:?set DCN_RUN_ID}"

bash /tmp/google_support/dcn/run_slice.sh

if [ "$DCN_SLICE_ID" = "0" ]; then
  metrics="$DCN_ARTIFACT_ROOT/rank-0/benchmark/metrics.jsonl"
  echo DCN_METRICS_JSONL_BEGIN
  cat "$metrics"
  echo DCN_METRICS_JSONL_END
fi
