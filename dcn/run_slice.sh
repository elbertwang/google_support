#!/usr/bin/env bash
set -euo pipefail

ulimit -c 0

: "${DCN_SLICE_ID:?set DCN_SLICE_ID to 0 or 1}"
: "${DCN_COORDINATOR_ADDRESS:?set host:port for slice 0}"
: "${DCN_ARTIFACT_ROOT:?set the writable artifact directory}"

export DCN_PROCESS_ID="${DCN_PROCESS_ID:-$DCN_SLICE_ID}"
export DCN_PROCESS_COUNT="${DCN_PROCESS_COUNT:-2}"
export DCN_DEVICES_PER_SLICE="${DCN_DEVICES_PER_SLICE:-8}"
export DCN_CODE_ROOT="${DCN_CODE_ROOT:-/workspace/google_support}"
export DCN_BENCHMARK_PATH="${DCN_BENCHMARK_PATH:-$DCN_CODE_ROOT/dcn/benchmark.py}"

export JAX_PLATFORMS=tpu
export MEGASCALE_COORDINATOR_ADDRESS="${DCN_COORDINATOR_ADDRESS%:*}:8081"
export MEGASCALE_NUM_SLICES="$DCN_PROCESS_COUNT"
export MEGASCALE_SLICE_ID="$DCN_SLICE_ID"
export MEGASCALE_PORT=8081

DCN_LIBTPU_ARGS="--xla_tpu_use_enhanced_launch_barrier=true \
--xla_tpu_prefer_async_allgather_to_allreduce=true \
--xla_tpu_data_parallel_opt_different_sized_ops=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true \
--megascale_coordinator_address=${MEGASCALE_COORDINATOR_ADDRESS} \
--megascale_slice_id=${MEGASCALE_SLICE_ID} \
--megascale_num_slices=${MEGASCALE_NUM_SLICES} \
--megascale_transport_type=grpc \
--megascale_port=${MEGASCALE_PORT} \
--megascale_use_insecure_grpc \
--megascale_grpc_interface_prefixes=${DCN_INTERFACES:-eth1,eth2,lo}"
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }$DCN_LIBTPU_ARGS"

rank_out="$DCN_ARTIFACT_ROOT/rank-$DCN_PROCESS_ID"
mkdir -p "$rank_out/benchmark" "$rank_out/profiling" "$rank_out/compiler/hlo"
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_dump_to=$rank_out/compiler/hlo --xla_dump_hlo_as_text"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/tpu_logs/jax-compilation-cache}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

if [ -n "${DCN_PROFILE:-}" ]; then
  export DCN_PROFILE_DIR="$rank_out/xprof"
  mkdir -p "$DCN_PROFILE_DIR"
fi

python_bin="${DCN_PYTHON:-python3}"
"$python_bin" "$DCN_CODE_ROOT/dcn/distributed_runner.py" \
  --output-dir "$rank_out" \
  --storage-dtype bf16 \
  --expected-num-slices 2 \
  --participants-per-slice "${DCN_PARTICIPANTS:-8}" \
  --participant-scan "${DCN_PARTICIPANT_SCAN:-}" \
  --dims "${DCN_DIMS:-8192,16384,24576,32768}" \
  --variants "${DCN_VARIANTS:-ppermute_uni,ppermute_bidi,all_gather,all_reduce}" \
  --batch "${DCN_BATCH:-10}" \
  --warmup-runs "${DCN_WARMUPS:-5}" \
  --sample-runs "${DCN_REPS:-5}" \
  --verify
