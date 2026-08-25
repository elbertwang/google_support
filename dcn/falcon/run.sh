#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cluster="${DCN_FALCON_CLUSTER:-tpu-training-antgroup-v2}"
run_id="${DCN_RUN_ID:-dcn-google-baseline-$(date -u +%Y%m%dT%H%M%SZ)}"
poll_seconds="${DCN_POLL_SECONDS:-15}"
timeout_seconds="${DCN_TIMEOUT_SECONDS:-7200}"
state_dir="${DCN_STATE_DIR:-$repo_root/dcn/results/$run_id}"
mkdir -p "$state_dir"

for command in falcon jq tar sed awk; do
  command -v "$command" >/dev/null || { echo "missing command: $command" >&2; exit 2; }
done

render_manifest() {
  slice_id="$1"
  sed -e "s/__NAME__/${run_id}-s${slice_id}/g" \
      -e "s/__CLUSTER__/${cluster}/g" \
      "$repo_root/dcn/falcon/holder.yaml" > "$state_dir/holder-s${slice_id}.yaml"
}

submit() {
  slice_id="$1"
  response="$(falcon workflow profile submit -f "$state_dir/holder-s${slice_id}.yaml" --output json)"
  printf '%s\n' "$response" > "$state_dir/submit-s${slice_id}.json"
  printf '%s\n' "$response" | jq -e '.ok == true and .ids.exp_id != null' >/dev/null
  printf '%s\n' "$response" | jq -r '.ids.exp_id'
}

wait_ready() {
  exp_id="$1"
  deadline=$(( $(date +%s) + timeout_seconds ))
  while true; do
    if holder_logs="$(falcon exp logs "$exp_id" --container task --tail 100 2>/dev/null)" \
      && printf '%s\n' "$holder_logs" | grep -q '^DCN_HOLDER_READY$'; then
      return 0
    fi
    if ! exp_json="$(falcon exp get "$exp_id" --output json)"; then
      echo "transient Falcon read failure for $exp_id; retrying" >&2
      [ "$(date +%s)" -lt "$deadline" ] || { echo "timeout waiting for $exp_id" >&2; return 1; }
      sleep "$poll_seconds"
      continue
    fi
    status="$(printf '%s\n' "$exp_json" | jq -r '.status // .ids.status // "unknown"' | tr '[:upper:]' '[:lower:]')"
    case "$status" in failed|aborted|deleted|succeeded)
      falcon exp logs "$exp_id" --tail 500 || true
      echo "$exp_id became $status before holder readiness" >&2
      return 1
    esac
    [ "$(date +%s)" -lt "$deadline" ] || { echo "timeout waiting for $exp_id" >&2; return 1; }
    sleep "$poll_seconds"
  done
}

render_manifest 0
render_manifest 1
exp0="${DCN_EXP0:-}"
exp1="${DCN_EXP1:-}"
if [ -z "$exp0" ] || [ -z "$exp1" ]; then
  exp0="$(submit 0)"
  exp1="$(submit 1)"
fi
printf 'slice_id\texp_id\n0\t%s\n1\t%s\n' "$exp0" "$exp1" | tee "$state_dir/experiments.tsv"

wait_ready "$exp0"
wait_ready "$exp1"

launch_slice() {
  exp_id="$1"
  slice_id="$2"
  falcon exp exec "$exp_id" --rank 0 --no-wait -- env \
    DCN_SLICE_ID="$slice_id" \
    DCN_PROCESS_ID="$slice_id" \
    DCN_PROCESS_COUNT=2 \
    DCN_DEVICES_PER_SLICE=8 \
    DCN_COORDINATOR_ADDRESS="$coordinator_address" \
    DCN_RUN_ID="$run_id" \
    DCN_SOURCE_COMMIT="$source_commit" \
    DCN_DIMS="${DCN_DIMS:-8192,16384,24576,32768}" \
    DCN_VARIANTS="${DCN_VARIANTS:-ppermute_uni,ppermute_bidi,all_gather,all_reduce}" \
    DCN_BATCH="${DCN_BATCH:-10}" \
    DCN_WARMUPS="${DCN_WARMUPS:-5}" \
    DCN_REPS="${DCN_REPS:-5}" \
    bash /tmp/google_support/dcn/falcon/remote_run.sh
}

if [ "${DCN_SKIP_LAUNCH:-0}" != "1" ]; then
  bundle="$(mktemp -t dcn-google-support.XXXXXX.tar.gz)"
  trap 'rm -f "$bundle"' EXIT
  COPYFILE_DISABLE=1 tar -C "$repo_root" \
    --exclude='dcn/results' --exclude='*/__pycache__' -czf "$bundle" dcn
  for exp_id in "$exp0" "$exp1"; do
    falcon exp cp "$bundle" "$exp_id:/tmp/dcn-bundle.tar.gz"
    falcon exp exec "$exp_id" --rank 0 -- mkdir -p /tmp/google_support
    falcon exp exec "$exp_id" --rank 0 -- tar -xzf /tmp/dcn-bundle.tar.gz -C /tmp/google_support
  done

  coordinator_output="$(falcon exp exec "$exp0" --rank 0 -- hostname -i)"
  coordinator_ip="$(printf '%s\n' "$coordinator_output" | awk '!found && match($0, /([0-9]{1,3}\.){3}[0-9]{1,3}/) { print substr($0, RSTART, RLENGTH); found=1 } END { if (!found) exit 1 }')"
  [ -n "$coordinator_ip" ] || { echo "unable to resolve slice-0 pod IP" >&2; exit 1; }
  coordinator_address="${coordinator_ip}:1234"
  source_commit="${DCN_BENCHMARK_COMMIT:-ccb9ab32250194a8e639a602099d583b9af63927}"

  launch_slice "$exp1" 1
  launch_slice "$exp0" 0
fi

collect_one() {
  exp_id="$1"
  output="$2"
  deadline=$(( $(date +%s) + timeout_seconds ))
  while true; do
    set +e
    response="$(falcon workflow profile collect "$exp_id" --timeout "${timeout_seconds}s" --output json)"
    rc=$?
    set -e
    printf '%s\n' "$response" | tee "$output"
    status="$(printf '%s\n' "$response" | jq -r '.ids.status // "UNKNOWN"')"
    if [ "$rc" -eq 0 ] && [ "$status" = "SUCCEEDED" ]; then
      return 0
    fi
    retry_safe="$(printf '%s\n' "$response" | jq -r '.data.retry_safe // false')"
    error_code="$(printf '%s\n' "$response" | jq -r '.error.code // empty')"
    if [ "$retry_safe" != "true" ] && [ "$error_code" != "runtime" ] && [ "$status" != "PENDING" ]; then
      return 1
    fi
    [ "$(date +%s)" -lt "$deadline" ] || return 1
    echo "retrying profile collect for $exp_id after status=$status error=$error_code" >&2
    sleep "$poll_seconds"
  done
}
collect_one "$exp0" "$state_dir/collect-s0.json"
collect_one "$exp1" "$state_dir/collect-s1.json"

falcon exp logs "$exp0" --output json > "$state_dir/logs-s0.json"
jq -r '.data.content // .content // empty' "$state_dir/logs-s0.json" \
  | sed -n 's/^DCN_GOOGLE_BASELINE_RESULT //p' > "$state_dir/metrics.jsonl"
[ -s "$state_dir/metrics.jsonl" ] || {
  echo "no result rows in Falcon logs; inspect $state_dir/logs-s0.json" >&2
  exit 1
}
python3 "$repo_root/dcn/results.py" "$state_dir/metrics.jsonl" | tee "$state_dir/comparison.md"

analysis_id="${DCN_ANALYSIS_ID:-}"
if [ -z "$analysis_id" ]; then
  sed "s/__EXP_ID__/$exp0/g" "$repo_root/dcn/falcon/operator-analysis.tmpl.yaml" > "$state_dir/operator-analysis.yaml"
  analysis_response="$(falcon workflow analysis create -f "$state_dir/operator-analysis.yaml" --output json)"
  printf '%s\n' "$analysis_response" > "$state_dir/analysis-create.json"
  analysis_id="$(printf '%s\n' "$analysis_response" | jq -r '.ids.analysis_id')"
fi
falcon workflow analysis wait "$analysis_id" --timeout 30m --output json | tee "$state_dir/analysis-wait.json"
falcon workflow analysis outputs "$analysis_id" --output json | tee "$state_dir/analysis-outputs.json"
if jq -e '.data.outputs[]? | select(.path == "report.md")' "$state_dir/analysis-outputs.json" >/dev/null; then
  falcon workflow analysis cat "$analysis_id" report.md --output json \
    | jq -r '.data.content' > "$state_dir/operator-report.md"
fi

echo "Falcon reproduction completed: $state_dir"
