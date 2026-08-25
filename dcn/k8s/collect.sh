#!/usr/bin/env bash
set -euo pipefail

mode="${1:-jobset}"
output="${2:-metrics.jsonl}"
case "$mode" in
  jobset)
    pod="$(kubectl get pods \
      -l 'jobset.sigs.k8s.io/jobset-name=dcn-google-baseline,jobset.sigs.k8s.io/job-index=0' \
      -o jsonpath='{.items[0].metadata.name}')"
    ;;
  pods) pod=dcn-google-baseline-s0 ;;
  *) echo "Usage: $0 [jobset|pods] [OUTPUT]" >&2; exit 2 ;;
esac

kubectl logs "$pod" \
  | sed -n '/^DCN_METRICS_JSONL_BEGIN$/,/^DCN_METRICS_JSONL_END$/p' \
  | sed '1d;$d' > "$output"
[ -s "$output" ] || { echo "no metrics found in $pod logs" >&2; exit 1; }
python3 "$(dirname "$0")/../results.py" "$output"
