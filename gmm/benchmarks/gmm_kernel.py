"""Uniform-distribution benchmark for the standalone GMM kernel."""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

_DEFAULT_LIBTPU_INIT_ARGS = (
    "--xla_enable_custom_call_region_trace=true "
    "--xla_xprof_register_llo_debug_info=true "
    "--xla_tpu_dvfs_p_state=7"
)


def _merge_libtpu_init_args(*args: str) -> str:
  return " ".join(arg.strip() for arg in args if arg and arg.strip())


os.environ["LIBTPU_INIT_ARGS"] = _merge_libtpu_init_args(
    os.environ.get("LIBTPU_INIT_ARGS", ""),
    _DEFAULT_LIBTPU_INIT_ARGS,
)

import jax
import jax.numpy as jnp
import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[1]
if str(REPRO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPRO_ROOT))

from gmm import gmm  # pylint: disable=wrong-import-position

MARKER = "GMM"
DEFAULT_M = 4 * 1024
DEFAULT_K = 2560
DEFAULT_N = 768
DEFAULT_E = 8
DEFAULT_TRACE_ROOT = "/tmp/gmm_kernel_benchmark_trace"
DEFAULT_TILING = (2048, 512, 4)

def time_kernel_grouped(
    cases: list[tuple[str, Callable[[], jax.Array]]],
    trace_name: str,
    *,
    iters: int,
    warmup: int,
    trace_root: str,
) -> dict[str, float | None]:
  trace_dir = Path(trace_root) / f"{trace_name}_{int(time.time() * 1000)}"
  trace_dir.mkdir(parents=True, exist_ok=True)
  options = jax.profiler.ProfileOptions()
  options.advanced_configuration = {
      "tpu_enable_periodic_counter_sampling": True,
      "tpu_tc_perf_counter_sampling_options": (
          "interval_us:1 scaling:0 counter_size_bits:1 "
          "indices:1 indices:3 indices:4 indices:10 indices:11 "
          "indices:31 indices:32 indices:33 indices:34 indices:35 "
          "indices:37 indices:38 indices:56 indices:57 indices:58 "
          "indices:73 indices:74 indices:75 indices:105"
      ),
      "num_tensor_cores_to_trace_per_device": 1,
  }

  for _, jit_fn in cases:
    for _ in range(warmup):
      jax.block_until_ready(jit_fn())

  with jax.profiler.trace(str(trace_dir), profiler_options=options):
    for task, jit_fn in cases:
      for i in range(iters):
        with jax.profiler.StepTraceAnnotation(f"{MARKER}:{task}", step_num=i):
          jax.block_until_ready(jit_fn())


def make_group_sizes_uniform(*, m: int, e: int) -> jax.Array:
  if m % e != 0:
    raise ValueError(f"Uniform group sizes require m % e == 0, got {m=} {e=}.")
  return jnp.full((e,), m // e, dtype=jnp.int32)


def make_inputs(m: int, k: int, n: int, e: int, dtype: jnp.dtype) -> tuple[jax.Array, jax.Array]:
  lhs = jax.random.normal(jax.random.PRNGKey(0), (m, k), dtype=dtype)
  rhs = jax.random.normal(jax.random.PRNGKey(1), (e, k, n), dtype=dtype)
  return lhs, rhs


def configure_jax_compilation_cache(cache_dir: str | None) -> None:
  if not cache_dir:
    return
  os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
  jax.config.update("jax_compilation_cache_dir", cache_dir)
  jax.config.update("jax_enable_compilation_cache", True)


def make_gmm_call(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    tiling: tuple[int, int, int],
    dtype: jnp.dtype,
) -> Callable[[], jax.Array]:
  @functools.partial(jax.jit, static_argnames=("tiling",))
  def fn(lhs, rhs, group_sizes, tiling):
    return gmm(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type=dtype,
        tiling=tiling,
        transpose_rhs=False,
    )

  return functools.partial(fn, lhs, rhs, group_sizes, tiling)


def make_task_name(*, m: int, k: int, n: int, e: int, tiling: tuple[int, int, int]) -> str:
  return (
      "uniform_rhs_gmm"
      f"_M{m}_K{k}_N{n}_E{e}_tm{tiling[0]}_tk{tiling[1]}_tn{tiling[2]}"
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--m", type=int, default=DEFAULT_M)
  parser.add_argument("--k", type=int, default=DEFAULT_K)
  parser.add_argument("--n", type=int, default=DEFAULT_N)
  parser.add_argument("--experts", type=int, default=DEFAULT_E)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--warmup", type=int, default=2)
  parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
  parser.add_argument("--trace-root", default=DEFAULT_TRACE_ROOT)
  parser.add_argument(
      "--jax-compilation-cache-dir",
      default=os.environ.get("JAX_COMPILATION_CACHE_DIR"),
      help="Directory for JAX persistent compilation cache. Defaults to JAX_COMPILATION_CACHE_DIR if set.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  configure_jax_compilation_cache(args.jax_compilation_cache_dir)

  dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
  tiling = DEFAULT_TILING
  group_sizes = make_group_sizes_uniform(m=args.m, e=args.experts)
  lhs, rhs = make_inputs(args.m, args.k, args.n, args.experts, dtype)
  task = make_task_name(m=args.m, k=args.k, n=args.n, e=args.experts, tiling=tiling)
  call = make_gmm_call(lhs, rhs, group_sizes, tiling=tiling, dtype=dtype)

  time_kernel_grouped(
      [(task, call)],
      trace_name="uniform_all_configs",
      iters=args.iters,
      warmup=args.warmup,
      trace_root=args.trace_root,
  )



if __name__ == "__main__":
  main()
