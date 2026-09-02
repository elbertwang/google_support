"""Add detailed timing instrumentation to the source DCN benchmark.

The source benchmark intentionally stays byte-for-byte identical to the
tpu-microbenchmarks reference. This wrapper replaces only ``timed_batches`` so
that compilation, warmup, collective execution, and the trailing host barrier
are measured separately. Every rank emits JSON records that can be paired with
``dcn/timing_results.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import time
from typing import Any


SOURCE_BENCHMARK = os.environ.get(
    "DCN_SOURCE_BENCHMARK", str(Path(__file__).with_name("benchmark.py"))
)
source = runpy.run_path(SOURCE_BENCHMARK, run_name="dcn_source_benchmark")
jax = source["jax"]
np = source["np"]
multihost_utils = source["multihost_utils"]
network_snapshot = source["network_snapshot"]
network_delta = source["network_delta"]


def emit(kind: str, payload: dict[str, Any]) -> None:
    print(f"DCN_TIMING_{kind} " + json.dumps(payload, sort_keys=True), flush=True)


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    median = float(np.median(array))
    stdev = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    percentiles = np.percentile(array, [5, 10, 25, 75, 90, 95])
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "p05": float(percentiles[0]),
        "p10": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "median": median,
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "max": float(np.max(array)),
        "mean": mean,
        "stdev": stdev,
        "cv": stdev / mean if mean else 0.0,
        "mad": float(np.median(np.abs(array - median))),
    }


def dcn_bytes(delta: dict[str, dict[str, int]], direction: str) -> int:
    return sum(
        int(delta.get(interface, {}).get(direction, 0))
        for interface in ("eth1", "eth2")
    )


def instrumented_timed_batches(
    fn,
    x,
    *,
    label: str,
    batch: int,
    reps: int,
    warmups: int,
) -> list[float]:
    """Return legacy samples while emitting collective-only paired timing."""

    rank = int(jax.process_index())
    experiment = os.environ.get("DCN_EXPERIMENT", "unknown")

    pre_barrier_started = time.perf_counter()
    multihost_utils.sync_global_devices(f"{experiment}_{label}_compile_start")
    compile_started = time.perf_counter()
    result = fn(x)
    compile_dispatched = time.perf_counter()
    jax.block_until_ready(result)
    compile_ready = time.perf_counter()
    multihost_utils.sync_global_devices(f"{experiment}_{label}_compile_done")
    compile_finished = time.perf_counter()
    emit(
        "FIRST_CALL",
        {
            "experiment": experiment,
            "label": label,
            "rank": rank,
            "pre_barrier_ms": (compile_started - pre_barrier_started) * 1e3,
            "dispatch_ms": (compile_dispatched - compile_started) * 1e3,
            "device_wait_ms": (compile_ready - compile_dispatched) * 1e3,
            "compile_execute_ms": (compile_ready - compile_started) * 1e3,
            "post_barrier_ms": (compile_finished - compile_ready) * 1e3,
        },
    )

    multihost_utils.sync_global_devices(f"{experiment}_{label}_warmup_start")
    warmup_started = time.perf_counter()
    for _ in range(warmups):
        result = fn(x)
    warmup_dispatched = time.perf_counter()
    jax.block_until_ready(result)
    warmup_ready = time.perf_counter()
    multihost_utils.sync_global_devices(f"{experiment}_{label}_warmup_done")
    emit(
        "WARMUP",
        {
            "experiment": experiment,
            "label": label,
            "rank": rank,
            "warmups": warmups,
            "dispatch_ms_total": (warmup_dispatched - warmup_started) * 1e3,
            "device_wait_ms_total": (warmup_ready - warmup_dispatched) * 1e3,
            "collective_ms_total": (warmup_ready - warmup_started) * 1e3,
            "collective_ms_per_iteration": (
                (warmup_ready - warmup_started) * 1e3 / warmups
            ),
        },
    )

    rows: list[dict[str, Any]] = []
    previous_network = network_snapshot()
    for rep in range(reps):
        multihost_utils.sync_global_devices(
            f"{experiment}_{label}_b{batch}_r{rep}_start"
        )
        started = time.perf_counter()
        for _ in range(batch):
            result = fn(x)
        dispatched = time.perf_counter()
        jax.block_until_ready(result)
        ready = time.perf_counter()
        multihost_utils.sync_global_devices(f"{experiment}_{label}_b{batch}_r{rep}_end")
        finished = time.perf_counter()
        current_network = network_snapshot()
        delta = network_delta(previous_network, current_network)
        previous_network = current_network

        collective_ms_total = (ready - started) * 1e3
        legacy_ms_total = (finished - started) * 1e3
        row = {
            "experiment": experiment,
            "label": label,
            "rank": rank,
            "batch": batch,
            "rep": rep,
            "dispatch_ms_per_iteration": (dispatched - started) * 1e3 / batch,
            "device_wait_ms_per_iteration": (ready - dispatched) * 1e3 / batch,
            "collective_ms_per_iteration": collective_ms_total / batch,
            "end_barrier_ms": (finished - ready) * 1e3,
            "end_barrier_ms_per_iteration": (finished - ready) * 1e3 / batch,
            "legacy_ms_per_iteration": legacy_ms_total / batch,
            "dcn_tx_bytes": dcn_bytes(delta, "tx_bytes"),
            "dcn_rx_bytes": dcn_bytes(delta, "rx_bytes"),
            "network_delta": delta,
        }
        rows.append(row)
        emit("REP", row)

    metric_names = (
        "dispatch_ms_per_iteration",
        "device_wait_ms_per_iteration",
        "collective_ms_per_iteration",
        "end_barrier_ms",
        "end_barrier_ms_per_iteration",
        "legacy_ms_per_iteration",
        "dcn_tx_bytes",
        "dcn_rx_bytes",
    )
    emit(
        "SUMMARY",
        {
            "experiment": experiment,
            "label": label,
            "rank": rank,
            "batch": batch,
            "reps": reps,
            "warmups": warmups,
            "metrics": {
                metric: distribution([float(row[metric]) for row in rows])
                for metric in metric_names
            },
        },
    )

    # Preserve the source benchmark's original barrier-inclusive output. The
    # paired analyzer uses collective_ms_per_iteration from the records above.
    return [float(row["legacy_ms_per_iteration"]) for row in rows]


emit(
    "CONFIG",
    {
        "experiment": os.environ.get("DCN_EXPERIMENT", "unknown"),
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "grpc_experiments": os.environ.get("GRPC_EXPERIMENTS", ""),
    },
)
source["main"].__globals__["timed_batches"] = instrumented_timed_batches
source["main"]()
