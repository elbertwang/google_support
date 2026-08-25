#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 jobset-dynamic IMAGE [OUTPUT]" >&2
  echo "       $0 jobset-static  IMAGE [OUTPUT]" >&2
  echo "       $0 pods IMAGE SLICE0_NODEPOOL SLICE1_NODEPOOL [OUTPUT]" >&2
  exit 2
}

[ "$#" -ge 2 ] || usage
mode="$1"
image="$2"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$mode" in
  jobset-dynamic)
    output="${3:-dcn-jobset.yaml}"
    sed "s|__IMAGE__|$image|g" "$script_dir/jobset.yaml" > "$output"
    ;;
  jobset-static)
    output="${3:-dcn-jobset.yaml}"
    sed "s|__IMAGE__|$image|g" "$script_dir/jobset-static.yaml" > "$output"
    ;;
  pods)
    [ "$#" -ge 4 ] || usage
    output="${5:-dcn-pods.yaml}"
    sed -e "s|__IMAGE__|$image|g" \
        -e "s|__SLICE0_NODEPOOL__|$3|g" \
        -e "s|__SLICE1_NODEPOOL__|$4|g" \
        "$script_dir/pod-pair.yaml" > "$output"
    ;;
  *) usage ;;
esac
echo "$output"
