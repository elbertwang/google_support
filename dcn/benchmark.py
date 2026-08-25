"""Google-style multislice DCN transport baseline adapted to TPU v7x.

This keeps the measurement structure of the v6e reference benchmark:

* mesh axes ``("dcn", "ici")`` with input sharding ``P("dcn", None)``;
* unidirectional and bidirectional ``ppermute``, ``all_gather``, and
  ``all_reduce``;
* a batch of asynchronous dispatches bracketed by global barriers;
* per-participant and per-host transmit bandwidth reporting.

On v7x, one physical chip exposes two JAX devices.  A physical ``2x2x1``
slice therefore contributes eight participants from four chips.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map as _shard_map


_SMAP_PARAMS = set(inspect.signature(_shard_map).parameters)
if "check_rep" in _SMAP_PARAMS:
    _CHECK_KW = {"check_rep": False}
elif "check_vma" in _SMAP_PARAMS:
    _CHECK_KW = {"check_vma": False}
else:
    _CHECK_KW = {}


def smap(body, mesh: Mesh, in_specs: P, out_specs: P):
    return _shard_map(
        body,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        **_CHECK_KW,
    )


def log(*values: Any) -> None:
    if jax.process_index() == 0:
        print(*values, flush=True)


def device_coordinate(device: Any) -> tuple[int, int, int, int]:
    coords = tuple(int(value) for value in getattr(device, "coords", ()))
    core = getattr(device, "core_on_chip", None)
    if len(coords) != 3 or core is None:
        return (int(device.process_index), int(device.id), 0, 0)
    return (*coords, int(core))


def physical_chip_coordinate(device: Any) -> tuple[int, int, int]:
    return device_coordinate(device)[:3]


def devices_by_slice() -> list[list[Any]]:
    groups: dict[int, list[Any]] = {}
    for device in jax.devices():
        slice_index = getattr(device, "slice_index", None)
        if slice_index is None:
            raise RuntimeError(f"device has no slice_index: {device}")
        groups.setdefault(int(slice_index), []).append(device)
    for devices in groups.values():
        devices.sort(key=device_coordinate)
    return [groups[index] for index in sorted(groups)]


def build_mesh(
    n_slices: int, participants_per_slice: int
) -> tuple[Mesh, list[list[Any]]]:
    groups = devices_by_slice()
    if len(groups) < n_slices:
        raise RuntimeError(f"need {n_slices} slices, found {len(groups)}")
    selected = [devices[:participants_per_slice] for devices in groups[:n_slices]]
    for slice_index, devices in enumerate(selected):
        if len(devices) != participants_per_slice:
            raise RuntimeError(
                f"slice {slice_index} has {len(devices)} devices; "
                f"need {participants_per_slice}"
            )
    return Mesh(np.asarray(selected, dtype=object), ("dcn", "ici")), selected


def timed_batches(
    fn,
    x,
    *,
    label: str,
    batch: int,
    reps: int,
    warmups: int,
) -> list[float]:
    """Return barrier-delimited per-iteration milliseconds."""

    multihost_utils.sync_global_devices(f"{label}_warmup_start")
    for _ in range(warmups):
        result = fn(x)
    jax.block_until_ready(result)
    multihost_utils.sync_global_devices(f"{label}_warmup_done")

    per_iteration_ms: list[float] = []
    for rep in range(reps):
        multihost_utils.sync_global_devices(f"{label}_rep_{rep}_start")
        started = time.perf_counter()
        for _ in range(batch):
            result = fn(x)
        jax.block_until_ready(result)
        # Required for the one-way case: the sender can complete before the
        # receiver has consumed the payload.
        multihost_utils.sync_global_devices(f"{label}_rep_{rep}_end")
        finished = time.perf_counter()
        per_iteration_ms.append((finished - started) * 1e3 / batch)
    return per_iteration_ms


def make_ppermute(mesh: Mesh, perm: list[tuple[int, int]]):
    spec = P("dcn", None)

    @jax.jit
    def operation(x):
        def body(value):
            return jax.lax.ppermute(value, "dcn", perm=perm)

        return smap(body, mesh, spec, spec)(x)

    return operation


def make_all_gather(mesh: Mesh):
    @jax.jit
    def operation(x):
        def body(value):
            return jax.lax.all_gather(value, "dcn", tiled=True)

        return smap(body, mesh, P("dcn", None), P(None, None))(x)

    return operation


def make_all_reduce(mesh: Mesh):
    spec = P("dcn", None)

    @jax.jit
    def operation(x):
        def body(value):
            return jax.lax.psum(value, "dcn")

        return smap(body, mesh, spec, spec)(x)

    return operation


def make_input(mesh: Mesh, dim: int):
    sharding = NamedSharding(mesh, P("dcn", None))
    return jax.jit(
        lambda: jnp.ones((dim, dim), dtype=jnp.bfloat16),
        out_shardings=sharding,
    )()


def verify_movement(mesh: Mesh, n_slices: int) -> dict[str, float]:
    dim = 512
    sharding = NamedSharding(mesh, P("dcn", None))
    host = np.zeros((dim, dim), dtype=np.float32)
    rows = dim // n_slices
    for slice_index in range(n_slices):
        host[slice_index * rows : (slice_index + 1) * rows, :] = slice_index + 1
    x = jax.device_put(host, sharding)

    @partial(jax.jit, out_shardings=NamedSharding(mesh, P()))
    def block_means(value):
        return value.reshape(n_slices, rows, dim).mean(axis=(1, 2))

    bidi = np.asarray(
        block_means(make_ppermute(mesh, [(0, 1), (1, 0)])(x))
    )
    uni = np.asarray(block_means(make_ppermute(mesh, [(0, 1)])(x)))
    all_reduce = np.asarray(block_means(make_all_reduce(mesh)(x)))
    expected_sum = float(n_slices * (n_slices + 1) // 2)
    if not np.allclose(all_reduce, expected_sum):
        raise AssertionError(
            f"AllReduce verification failed: expected {expected_sum}, "
            f"block means={all_reduce.tolist()}"
        )
    return {
        "bidi_block0_mean": float(bidi[0]),
        "bidi_block1_mean": float(bidi[1]),
        "uni_block0_mean": float(uni[0]),
        "uni_block1_mean": float(uni[1]),
        "all_reduce_block0_mean": float(all_reduce[0]),
        "all_reduce_block1_mean": float(all_reduce[1]),
        "all_reduce_expected_sum": expected_sum,
    }


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def network_snapshot() -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for interface_path in sorted(Path("/sys/class/net").glob("*")):
        name = interface_path.name
        snapshot[name] = {
            "speed_mbps": read_text(interface_path / "speed"),
            "operstate": read_text(interface_path / "operstate"),
            "tx_bytes": int(read_text(interface_path / "statistics/tx_bytes") or 0),
            "rx_bytes": int(read_text(interface_path / "statistics/rx_bytes") or 0),
        }
    return snapshot


def network_delta(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, dict[str, int]]:
    return {
        name: {
            "tx_bytes": int(values["tx_bytes"]) - int(before.get(name, {}).get("tx_bytes", 0)),
            "rx_bytes": int(values["rx_bytes"]) - int(before.get(name, {}).get("rx_bytes", 0)),
        }
        for name, values in after.items()
    }


def parse_int_list(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"expected positive comma-separated integers, got {value!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Compatibility with the existing Falcon holder wrapper.
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--bucket-mib", type=int, default=512)
    parser.add_argument("--storage-dtype", default="bf16")
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--sample-runs", type=int, default=5)
    parser.add_argument("--trace-sample-runs", type=int, default=1)
    parser.add_argument("--logical-slice-shape", default="")
    parser.add_argument("--expected-physical-shape", default="")
    parser.add_argument("--expected-num-slices", type=int, required=True)
    parser.add_argument("--verify", action="store_true")

    parser.add_argument("--participants-per-slice", type=int, default=0)
    parser.add_argument("--participant-scan", default="")
    parser.add_argument("--dims", default="8192,16384,24576,32768")
    parser.add_argument(
        "--variants",
        default="ppermute_uni,ppermute_bidi,all_gather,all_reduce",
    )
    parser.add_argument("--batch", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.storage_dtype != "bf16":
        raise ValueError("the Google baseline currently requires BF16")
    if args.expected_num_slices != 2:
        raise ValueError("the uni/bidi baseline currently requires exactly two slices")
    if args.warmup_runs <= 0 or args.sample_runs <= 0 or args.batch <= 0:
        raise ValueError("warmup, sample, and batch counts must be positive")

    slice_groups = devices_by_slice()
    if len(slice_groups) != args.expected_num_slices:
        raise ValueError(
            f"found {len(slice_groups)} slices, expected {args.expected_num_slices}"
        )
    full_participants = min(len(devices) for devices in slice_groups)
    participants_per_slice = args.participants_per_slice or full_participants
    participant_scan = (
        parse_int_list(args.participant_scan)
        if args.participant_scan
        else [participants_per_slice]
    )
    if any(count > full_participants for count in participant_scan):
        raise ValueError(
            f"participant scan {participant_scan} exceeds {full_participants} devices/slice"
        )

    dims = parse_int_list(args.dims)
    variants = args.variants.split(",")
    unknown_variants = set(variants) - {
        "ppermute_uni",
        "ppermute_bidi",
        "all_gather",
        "all_reduce",
    }
    if unknown_variants:
        raise ValueError(f"unknown variants: {sorted(unknown_variants)}")

    output_dir = Path(args.output_dir or os.environ.get("ARTIFACT_LOCAL_DIR", "/tmp"))
    benchmark_dir = output_dir / "benchmark"
    profiling_dir = output_dir / "profiling"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    profiling_dir.mkdir(parents=True, exist_ok=True)

    environment = {
        "benchmark": "dcn_google_baseline",
        "jax_version": jax.__version__,
        "process_index": jax.process_index(),
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "device_kind": jax.devices()[0].device_kind,
        "slices_detected": len(slice_groups),
        "devices_per_slice": [len(devices) for devices in slice_groups],
        "physical_chips_per_slice": [
            len({physical_chip_coordinate(device) for device in devices})
            for devices in slice_groups
        ],
        "libtpu_init_args": os.environ.get("LIBTPU_INIT_ARGS", ""),
        "network_before": network_snapshot(),
    }
    log("DCN_GOOGLE_BASELINE_ENV " + json.dumps(environment, sort_keys=True))

    full_mesh, _ = build_mesh(args.expected_num_slices, full_participants)
    if args.verify:
        movement = verify_movement(full_mesh, args.expected_num_slices)
        environment["movement_check"] = movement
        log("DCN_GOOGLE_BASELINE_MOVEMENT " + json.dumps(movement, sort_keys=True))

    results: list[dict[str, Any]] = []
    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    for participant_count in participant_scan:
        mesh, selected = build_mesh(args.expected_num_slices, participant_count)
        chips_per_slice = len(
            {physical_chip_coordinate(device) for device in selected[0]}
        )
        for dim in dims:
            x = make_input(mesh, dim)
            shard_bytes = dim * dim * itemsize / args.expected_num_slices
            factories = {
                "ppermute_uni": lambda: make_ppermute(mesh, [(0, 1)]),
                "ppermute_bidi": lambda: make_ppermute(mesh, [(0, 1), (1, 0)]),
                "all_gather": lambda: make_all_gather(mesh),
                "all_reduce": lambda: make_all_reduce(mesh),
            }
            for variant in variants:
                label = f"google_{variant}_p{participant_count}_d{dim}"
                before = network_snapshot()
                samples_ms = timed_batches(
                    factories[variant](),
                    x,
                    label=label,
                    batch=args.batch,
                    reps=args.sample_runs,
                    warmups=args.warmup_runs,
                )
                after = network_snapshot()
                median_ms = statistics.median(samples_ms)
                minimum_ms = min(samples_ms)
                per_device_median = shard_bytes / 1e9 / (median_ms / 1e3)
                per_device_best = shard_bytes / 1e9 / (minimum_ms / 1e3)
                ring_factor = (
                    2.0
                    * (args.expected_num_slices - 1)
                    / args.expected_num_slices
                )
                row = {
                    "benchmark": "dcn_google_baseline",
                    "variant": variant,
                    "rank": jax.process_index(),
                    "n_slices": args.expected_num_slices,
                    "participants_per_slice": participant_count,
                    "physical_chips_per_slice": chips_per_slice,
                    "matrix_dim": dim,
                    "dtype": "bfloat16",
                    "shard_bytes_per_device": shard_bytes,
                    "batch": args.batch,
                    "reps": args.sample_runs,
                    "warmups": args.warmup_runs,
                    "time_ms_median": median_ms,
                    "time_ms_min": minimum_ms,
                    "time_ms_all": samples_ms,
                    "network_delta": network_delta(before, after),
                    "run_id": os.environ.get("FALCON_EXP_ID", ""),
                    "source_commit": os.environ.get(
                        "TPU_MICROBENCH_SOURCE_COMMIT", ""
                    ),
                }
                if variant == "all_reduce":
                    row.update(
                        {
                            "bucket_semantics": "all_reduce_input_bytes_per_device",
                            "algorithm_GBps_per_device_median": per_device_median,
                            "algorithm_GBps_per_device_best": per_device_best,
                            "ring_equivalent_factor": ring_factor,
                            "ring_equivalent_bus_GBps_per_device_median": (
                                per_device_median * ring_factor
                            ),
                            "ring_equivalent_bus_GBps_per_device_best": (
                                per_device_best * ring_factor
                            ),
                            "host_algorithm_GBps_median": (
                                per_device_median * participant_count
                            ),
                            "host_algorithm_GBps_best": (
                                per_device_best * participant_count
                            ),
                            "host_ring_equivalent_bus_GBps_median": (
                                per_device_median * ring_factor * participant_count
                            ),
                            "host_ring_equivalent_bus_GBps_best": (
                                per_device_best * ring_factor * participant_count
                            ),
                            "per_physical_chip_ring_equivalent_bus_GBps_median": (
                                per_device_median
                                * ring_factor
                                * participant_count
                                / chips_per_slice
                            ),
                            "per_physical_chip_ring_equivalent_bus_GBps_best": (
                                per_device_best
                                * ring_factor
                                * participant_count
                                / chips_per_slice
                            ),
                        }
                    )
                else:
                    row.update(
                        {
                            "per_device_tx_GBps_median": per_device_median,
                            "per_device_tx_GBps_best": per_device_best,
                            "host_tx_GBps_median": (
                                per_device_median * participant_count
                            ),
                            "host_tx_GBps_best": (
                                per_device_best * participant_count
                            ),
                            "per_physical_chip_tx_GBps_median": (
                                per_device_median
                                * participant_count
                                / chips_per_slice
                            ),
                            "per_physical_chip_tx_GBps_best": (
                                per_device_best
                                * participant_count
                                / chips_per_slice
                            ),
                        }
                    )
                results.append(row)
                log("DCN_GOOGLE_BASELINE_RESULT " + json.dumps(row, sort_keys=True))
            del x

    environment["network_after"] = network_snapshot()
    summary: dict[str, dict[str, float]] = {}
    for variant in variants:
        candidates = [
            row
            for row in results
            if row["variant"] == variant
            and row["participants_per_slice"] == participants_per_slice
        ]
        selection_key = (
            "host_ring_equivalent_bus_GBps_best"
            if variant == "all_reduce"
            else "host_tx_GBps_best"
        )
        best = max(candidates, key=lambda row: row[selection_key])
        summary[variant] = {
            "matrix_dim": int(best["matrix_dim"]),
            "shard_bytes_per_device": float(best["shard_bytes_per_device"]),
        }
        if variant == "all_reduce":
            summary[variant].update(
                {
                    "ring_equivalent_factor": float(
                        best["ring_equivalent_factor"]
                    ),
                    "algorithm_GBps_per_device_median": float(
                        best["algorithm_GBps_per_device_median"]
                    ),
                    "algorithm_GBps_per_device_best": float(
                        best["algorithm_GBps_per_device_best"]
                    ),
                    "ring_equivalent_bus_GBps_per_device_median": float(
                        best["ring_equivalent_bus_GBps_per_device_median"]
                    ),
                    "ring_equivalent_bus_GBps_per_device_best": float(
                        best["ring_equivalent_bus_GBps_per_device_best"]
                    ),
                    "host_algorithm_GBps_median": float(
                        best["host_algorithm_GBps_median"]
                    ),
                    "host_algorithm_GBps_best": float(
                        best["host_algorithm_GBps_best"]
                    ),
                    "host_ring_equivalent_bus_GBps_median": float(
                        best["host_ring_equivalent_bus_GBps_median"]
                    ),
                    "host_ring_equivalent_bus_GBps_best": float(
                        best["host_ring_equivalent_bus_GBps_best"]
                    ),
                    "per_physical_chip_ring_equivalent_bus_GBps_median": float(
                        best[
                            "per_physical_chip_ring_equivalent_bus_GBps_median"
                        ]
                    ),
                    "per_physical_chip_ring_equivalent_bus_GBps_best": float(
                        best["per_physical_chip_ring_equivalent_bus_GBps_best"]
                    ),
                }
            )
        else:
            summary[variant].update(
                {
                    "host_tx_GBps_median": float(best["host_tx_GBps_median"]),
                    "host_tx_GBps_best": float(best["host_tx_GBps_best"]),
                    "per_device_tx_GBps_median": float(
                        best["per_device_tx_GBps_median"]
                    ),
                    "per_device_tx_GBps_best": float(
                        best["per_device_tx_GBps_best"]
                    ),
                    "per_physical_chip_tx_GBps_median": float(
                        best["per_physical_chip_tx_GBps_median"]
                    ),
                    "per_physical_chip_tx_GBps_best": float(
                        best["per_physical_chip_tx_GBps_best"]
                    ),
                }
            )

    payload = {"environment": environment, "summary": summary, "results": results}
    (benchmark_dir / "google_baseline_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (benchmark_dir / "metrics.jsonl").open("w", encoding="utf-8") as output:
        for row in results:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    (profiling_dir / "google_baseline_environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log("DCN_GOOGLE_BASELINE_SUMMARY " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
