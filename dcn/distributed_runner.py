"""Initialize one process per physical TPU slice, then run the benchmark."""

from __future__ import annotations

import json
import os
import runpy

import jax


def _required_int(name: str) -> int:
  value = os.environ.get(name)
  if value is None:
    raise RuntimeError(f"missing required environment variable: {name}")
  return int(value)


def main() -> None:
  slice_id = _required_int("DCN_SLICE_ID")
  process_id = _required_int("DCN_PROCESS_ID")
  process_count = _required_int("DCN_PROCESS_COUNT")
  expected_devices_per_slice = _required_int("DCN_DEVICES_PER_SLICE")

  jax.distributed.initialize(
      coordinator_address=os.environ["DCN_COORDINATOR_ADDRESS"],
      num_processes=process_count,
      process_id=process_id,
      cluster_detection_method="deactivate",
      initialization_timeout=1800,
      partition_index=slice_id,
  )

  slice_counts: dict[int, int] = {}
  for device in jax.devices():
    device_slice = getattr(device, "slice_index", None)
    if device_slice is None:
      raise RuntimeError(f"device has no slice_index: {device}")
    slice_counts[int(device_slice)] = slice_counts.get(int(device_slice), 0) + 1

  preflight = {
      "process_id": jax.process_index(),
      "process_count": jax.process_count(),
      "local_device_count": jax.local_device_count(),
      "global_device_count": jax.device_count(),
      "slice_counts": slice_counts,
  }
  print("DCN_MULTISLICE_PREFLIGHT " + json.dumps(preflight, sort_keys=True), flush=True)

  assert jax.process_count() == process_count, preflight
  assert jax.local_device_count() == expected_devices_per_slice, preflight
  assert jax.device_count() == process_count * expected_devices_per_slice, preflight
  assert len(slice_counts) == process_count, preflight
  assert set(slice_counts.values()) == {expected_devices_per_slice}, preflight

  # Optional xprof capture. benchmark.py parses --profile-dir but never uses it
  # (it is a compat shim for the Falcon wrapper), and nothing in the upstream
  # package ever calls jax.profiler. Wrapping here keeps benchmark.py byte-identical
  # to the SHA the repo pins.
  profile_dir = os.environ.get("DCN_PROFILE_DIR", "")
  if profile_dir:
    os.makedirs(profile_dir, exist_ok=True)
    print(f"DCN_XPROF_TRACE_START {profile_dir}", flush=True)
    with jax.profiler.trace(profile_dir):
      runpy.run_path(os.environ["DCN_BENCHMARK_PATH"], run_name="__main__")
    print("DCN_XPROF_TRACE_DONE", flush=True)
  else:
    runpy.run_path(os.environ["DCN_BENCHMARK_PATH"], run_name="__main__")


if __name__ == "__main__":
  main()
