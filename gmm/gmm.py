from __future__ import annotations

from collections.abc import Callable
import functools
import json
from typing import Any, Optional

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp

try:
  import qwix.pallas as qpl
except ModuleNotFoundError:
  class _QArray:  # Minimal fallback; this reproduce only uses normal JAX arrays.
    pass

  class _QplModule:
    QArray = _QArray

  qpl = _QplModule()


def _validate_args(
    *,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    expected_rhs_dims: int = 3,
) -> jnp.ndarray:
  """Validates the arguments for the gmm function."""
  # Validate 'lhs'.
  if lhs.ndim != 2:
    raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")

  # Validate 'rhs'.
  if rhs.ndim != expected_rhs_dims:
    raise ValueError(f"Expected {expected_rhs_dims}-tensor for 'rhs' but got" f" {rhs.ndim}-tensor.")
  # Validate 'group_sizes'.
  if group_sizes.dtype != jnp.int32:
    raise ValueError(f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}.")

  return group_sizes


def _calculate_num_tiles(x: int, tx: int) -> int:
  tiles, rem = divmod(x, tx)
  if rem:
    raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
  return tiles


GroupMetadata = Any  # TODO(enriqueps): Clean this up and use a namedtuple


def make_group_metadata_gmm(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    tms: int,
    te: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
) -> tuple[GroupMetadata, jnp.ndarray]:
  """Create packed active-tile metadata and a precomputed GMM DMA schedule."""
  group_ends = jnp.cumsum(group_sizes)
  group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])
  active_group_offsets = jnp.arange(num_nonzero_groups, dtype=jnp.int32)
  active_group_starts = lax.dynamic_slice_in_dim(
      group_offsets,
      start_group,
      num_nonzero_groups,
  )
  active_group_ends = lax.dynamic_slice_in_dim(
      group_offsets,
      start_group + 1,
      num_nonzero_groups,
  )
  active_group_sizes = lax.dynamic_slice_in_dim(
      group_sizes,
      start_group,
      num_nonzero_groups,
  )
  first_m_tile_ids = active_group_starts // tms
  group_tiles = (active_group_ends + tms - 1) // tms - first_m_tile_ids
  group_tiles = jnp.where(active_group_sizes == 0, 0, group_tiles).astype(jnp.int32)
  num_tiles = group_tiles.sum()

  tiles_m = _calculate_num_tiles(m, tms)
  metadata_len = tiles_m + num_nonzero_groups - 1
  active_i = jnp.arange(metadata_len, dtype=jnp.int32)
  valid = active_i < num_tiles
  group_tile_offsets = jnp.cumsum(group_tiles) - group_tiles
  group_tile_ends = group_tile_offsets + group_tiles
  group_before = group_tile_ends[:, None] <= active_i[None, :]
  group_id_offsets = group_before.sum(axis=0).astype(jnp.int32)
  group_id_offsets = jnp.minimum(group_id_offsets, num_nonzero_groups - 1)
  group_ids = start_group + group_id_offsets
  m_tile_id_base = first_m_tile_ids - group_tile_offsets
  m_tile_id_deltas = m_tile_id_base[1:] - m_tile_id_base[:-1]
  m_tile_id_offsets = m_tile_id_base[0] + jnp.where(
      group_before[:-1],
      m_tile_id_deltas[:, None],
      jnp.zeros((num_nonzero_groups - 1, 1), dtype=jnp.int32),
  ).sum(axis=0)
  m_tile_ids = m_tile_id_offsets + active_i
  m_tile_ids = jnp.where(valid, m_tile_ids, tiles_m - 1).astype(jnp.int32)

  tiles_m_cache = _calculate_num_tiles(m, tm)
  sub_tiles_per_cache = tm // tms
  lhs_cache_tile_ids = m_tile_ids // sub_tiles_per_cache
  rhs_block_ids = group_id_offsets // te

  max_rhs_blocks = _calculate_num_tiles(num_nonzero_groups, te)
  padded_group_tiles = jnp.pad(
      group_tiles,
      (0, max_rhs_blocks * te - num_nonzero_groups),
  ).reshape((max_rhs_blocks, te))
  rhs_block_has_tiles = padded_group_tiles.sum(axis=1) > 0
  rhs_block_iota = jnp.arange(max_rhs_blocks, dtype=jnp.int32)

  def _dense_rhs_schedule():
    rhs_active_te_starts = jnp.concatenate(
        [
            rhs_block_iota * te,
            -jnp.ones((2,), dtype=jnp.int32),
        ]
    )
    return rhs_block_ids, rhs_active_te_starts

  def _sparse_rhs_schedule():
    rhs_block_ordinals = jnp.cumsum(rhs_block_has_tiles.astype(jnp.int32)) - 1
    rhs_schedule_block_ids = jnp.maximum(rhs_block_ordinals[rhs_block_ids], 0)
    num_rhs_blocks = rhs_block_has_tiles.astype(jnp.int32).sum()
    rhs_active_te_starts = jnp.full((max_rhs_blocks,), 0, dtype=jnp.int32)
    rhs_active_te_starts = rhs_active_te_starts.at[
        jnp.maximum(rhs_block_ordinals, 0)
    ].max(
        jnp.where(
            rhs_block_has_tiles,
            rhs_block_iota * te,
            -jnp.ones((max_rhs_blocks,), dtype=jnp.int32),
        )
    )
    rhs_active_te_starts = jnp.where(
        rhs_block_iota < num_rhs_blocks,
        rhs_active_te_starts,
        -jnp.ones((max_rhs_blocks,), dtype=jnp.int32),
    )
    rhs_active_te_starts = jnp.concatenate(
        [
            rhs_active_te_starts,
            -jnp.ones((2,), dtype=jnp.int32),
        ]
    )
    return rhs_schedule_block_ids, rhs_active_te_starts

  rhs_schedule_block_ids, rhs_active_te_starts = lax.cond(
      rhs_block_has_tiles.all(),
      _dense_rhs_schedule,
      _sparse_rhs_schedule,
  )

  return (
      group_offsets,
      group_ids.astype(jnp.int32),
      m_tile_ids.astype(jnp.int32),
      rhs_schedule_block_ids.astype(jnp.int32),
      rhs_active_te_starts,
  ), num_tiles


def _zero_uninitialized_memory(
    out: jnp.ndarray,
    *,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> jnp.ndarray:
  """Zero out uninitialized memory from output."""
  group_offsets = group_metadata[0]
  group_start = group_offsets[start_group]
  group_end = group_offsets[start_group + num_nonzero_groups]
  valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
  valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
  return jnp.where(valid_mask[:, None], out, 0)


def _calculate_bytes(x: jax.Array | qpl.QArray) -> int:
  total_bytes = 0
  for leaf in jax.tree.leaves(x):
    total_bytes += leaf.dtype.itemsize * leaf.size
  return total_bytes


LutFn = Callable[[int, int, int], Optional[tuple[int, int, int]]]


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "transpose_rhs",
        "interpret",
    ],
)
def gmm(
    lhs: jnp.ndarray | qpl.QArray,
    rhs: jnp.ndarray | qpl.QArray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 32, 2),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray:
  """Metadata-driven GMM schedule for VMEM cache layout."""

  if isinstance(lhs, qpl.QArray) or isinstance(rhs, qpl.QArray):
    raise NotImplementedError(
        "gmm does not support QArray inputs yet."
    )
  if existing_out is not None:
    raise NotImplementedError(
        "gmm does not support existing_out."
    )
  if group_offset is None:
    group_offset = jnp.array([0], dtype=jnp.int32)
  else:
    if group_offset.shape:
      raise ValueError(
          f"group_offset must be a ()-shaped array. Got: {group_offset.shape}."
      )
    group_offset = group_offset[None]
  num_current_groups = rhs.shape[0]
  num_total_groups = group_sizes.shape[0]
  group_sizes = _validate_args(lhs=lhs, rhs=rhs, group_sizes=group_sizes)

  m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
  if transpose_rhs:
    n = rhs.shape[1]

  if callable(tiling):
    tiling = tiling(m, k, n)

  if tiling is None:
    raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

  tm, tms, te = tiling
  if tm % tms != 0:
    raise ValueError(f"tm ({tm}) must be divisible by tms ({tms}).")
  if num_current_groups % te != 0:
    raise ValueError(f"rhs.shape[0] ({num_current_groups}) must be divisible by te ({te}).")
  tiles_m_cache = _calculate_num_tiles(m, tm)
  sub_tiles_per_cache = tm // tms
  max_rhs_blocks = _calculate_num_tiles(num_current_groups, te)
  metadata_len = _calculate_num_tiles(m, tms) + num_current_groups - 1

  with jax.named_scope(f"make_group_metadata_gmm"):
    group_metadata, num_active_tiles = make_group_metadata_gmm(  # pylint: disable=unbalanced-tuple-unpacking
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        tms=tms,
        te=te,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
    )

  def kernel(
      group_offsets,
      group_ids,
      m_tile_ids,
      rhs_schedule_block_ids,
      rhs_active_te_starts,
      group_offset,
      num_active_tiles,
      lhs: jax.Array | qpl.QArray,
      rhs: jax.Array | qpl.QArray,
      out,
      lhs_scratch,
      rhs_scratch,
      acc_scratch,
      lhs_dma_sems,
      rhs_dma_sems,
      out_store_sems,
  ):
    group_offset_value = group_offset[0]
    initial_lhs_0 = m_tile_ids[0] // sub_tiles_per_cache
    initial_lhs_1 = initial_lhs_0 + 1
    initial_rhs_0 = rhs_active_te_starts[0]
    initial_rhs_1 = rhs_active_te_starts[jnp.minimum(1, max_rhs_blocks - 1)]
    final_i = jnp.maximum(num_active_tiles[()] - 1, 0)
    final_lhs_relative_cache_id = (
        m_tile_ids[final_i] // sub_tiles_per_cache
    ) - initial_lhs_0

    def _start_lhs_fetch(lhs_cache_tile_id: int, slot: int):
      tile_start = lhs_cache_tile_id * tm
      pltpu.make_async_copy(
          src_ref=lhs.at[pl.ds(tile_start, tm), pl.ds(0, k)],
          dst_ref=lhs_scratch.at[slot],
          sem=lhs_dma_sems.at[slot],
      ).start()

    def _wait_lhs_fetch(slot: int):
      pltpu.make_async_copy(
          src_ref=lhs_scratch.at[slot],
          dst_ref=lhs_scratch.at[slot],
          sem=lhs_dma_sems.at[slot],
      ).wait()

    def _start_rhs_fetch(te_start: int, slot: int):
      if transpose_rhs:
        rhs_ref = rhs.at[pl.ds(te_start, te), pl.ds(0, n), pl.ds(0, k)]
      else:
        rhs_ref = rhs.at[pl.ds(te_start, te), pl.ds(0, k), pl.ds(0, n)]
      pltpu.make_async_copy(
          src_ref=rhs_ref,
          dst_ref=rhs_scratch.at[slot],
          sem=rhs_dma_sems.at[slot],
      ).start()

    def _wait_rhs_fetch(slot: int):
      pltpu.make_async_copy(
          src_ref=rhs_scratch.at[slot],
          dst_ref=rhs_scratch.at[slot],
          sem=rhs_dma_sems.at[slot],
      ).wait()

    def _start_acc_store(acc_slot: int, lhs_cache_tile_id: int):
      tile_start = lhs_cache_tile_id * tm
      pltpu.make_async_copy(
          src_ref=acc_scratch.at[acc_slot],
          dst_ref=out.at[pl.ds(tile_start, tm), pl.ds(0, n)],
          sem=out_store_sems.at[acc_slot],
      ).start()

    def _wait_acc_store(acc_slot: int):
      pltpu.make_async_copy(
          src_ref=acc_scratch.at[acc_slot],
          dst_ref=acc_scratch.at[acc_slot],
          sem=out_store_sems.at[acc_slot],
      ).wait()

    with jax.named_scope("gmm_start_lhs_s0"):
      _start_lhs_fetch(initial_lhs_0, jnp.array(0, dtype=jnp.int32))

    @pl.when(initial_lhs_1 < tiles_m_cache)
    def _prefetch_initial_lhs_slot1():
      with jax.named_scope("gmm_start_lhs_s1"):
        _start_lhs_fetch(initial_lhs_1, jnp.array(1, dtype=jnp.int32))

    with jax.named_scope("gmm_start_rhs_s0"):
      _start_rhs_fetch(initial_rhs_0, jnp.array(0, dtype=jnp.int32))

    @pl.when(initial_rhs_1 >= 0)
    def _prefetch_initial_rhs_slot1():
      with jax.named_scope("gmm_start_rhs_s1"):
        _start_rhs_fetch(initial_rhs_1, jnp.array(1, dtype=jnp.int32))

    if transpose_rhs:
      dot_general_dims = (((1,), (1,)), ((), ()))
    else:
      dot_general_dims = (((1,), (0,)), ((), ()))

    def _active_body(active_i, carry):
      del carry

      @pl.when(active_i < num_active_tiles[()])
      def _do_active_tile():
        group_id = group_ids[active_i]
        m_tile_id = m_tile_ids[active_i]
        lhs_cache_tile_id = m_tile_id // sub_tiles_per_cache
        sub_offset = (m_tile_id % sub_tiles_per_cache) * tms
        group_id_local = group_id - group_offset_value
        rhs_expert_offset = group_id_local % te
        rhs_te_start = (group_id_local // te) * te
        lhs_relative_cache_id = lhs_cache_tile_id - initial_lhs_0
        lhs_slot = (lhs_cache_tile_id - initial_lhs_0) % 2
        rhs_schedule_block_id = rhs_schedule_block_ids[active_i]
        rhs_slot = rhs_schedule_block_id % 2
        acc_slot = lhs_slot
        prefetch_lhs_cache_tile_id = lhs_cache_tile_id + 2
        prefetch_rhs_te_start = rhs_active_te_starts[rhs_schedule_block_id + 2]
        safe_prev_i = jnp.maximum(active_i - 1, 0)
        safe_next_i = jnp.minimum(active_i + 1, num_active_tiles[()] - 1)
        prev_m_tile_id = m_tile_ids[safe_prev_i]
        next_m_tile_id = m_tile_ids[safe_next_i]
        prev_lhs_cache_tile_id = prev_m_tile_id // sub_tiles_per_cache
        next_lhs_cache_tile_id = next_m_tile_id // sub_tiles_per_cache
        prev_group_id = group_ids[safe_prev_i]
        prev_group_id_local = prev_group_id - group_offset_value
        prev_rhs_te_start = (prev_group_id_local // te) * te
        next_group_id = group_ids[safe_next_i]
        next_group_id_local = next_group_id - group_offset_value
        next_rhs_te_start = (next_group_id_local // te) * te
        first_for_lhs = jnp.logical_or(active_i == 0, lhs_cache_tile_id != prev_lhs_cache_tile_id)
        first_for_rhs = jnp.logical_or(active_i == 0, rhs_te_start != prev_rhs_te_start)
        last_for_lhs = jnp.logical_or(
            active_i == num_active_tiles[()] - 1,
            lhs_cache_tile_id != next_lhs_cache_tile_id,
        )
        last_for_rhs = jnp.logical_or(
            active_i == num_active_tiles[()] - 1,
            rhs_te_start != next_rhs_te_start,
        )
        prefetch_lhs_after_dot = jnp.logical_and(
            last_for_lhs,
            prefetch_lhs_cache_tile_id < tiles_m_cache,
        )
        prefetch_rhs_after_dot = jnp.logical_and(last_for_rhs, prefetch_rhs_te_start >= 0)
        group_start = group_offsets[group_id]
        group_end = group_offsets[group_id + 1]
        m_id = m_tile_id * tms
        store_full_tile = jnp.logical_and(
            group_start <= m_id,
            m_id + tms <= group_end,
        )

        @pl.when(jnp.logical_and(first_for_lhs, lhs_relative_cache_id >= 2))
        def _wait_acc_before_dot():
          with jax.named_scope(f"gmm_wait_acc_before_dot_s{acc_slot}"):
            _wait_acc_store(acc_slot)

        @pl.when(first_for_lhs)
        def _wait_lhs_before_dot():
          with jax.named_scope(f"gmm_wait_lhs_before_dot_s{lhs_slot}"):
            _wait_lhs_fetch(lhs_slot)

        @pl.when(first_for_rhs)
        def _wait_rhs_before_dot():
          with jax.named_scope(f"gmm_wait_rhs_before_dot_s{rhs_slot}"):
            _wait_rhs_fetch(rhs_slot)

        with jax.named_scope(f"gmm_dot_loff{sub_offset}_roff{rhs_expert_offset}"):
          partial = jax.lax.dot_general(
              lhs_scratch[lhs_slot, pl.dslice(sub_offset, tms), :],
              rhs_scratch[rhs_slot, rhs_expert_offset, :, :],
              preferred_element_type=jnp.float32,
              dimension_numbers=dot_general_dims,
          )

        @pl.when(prefetch_lhs_after_dot)
        def _prefetch_lhs_after_dot():
          with jax.named_scope(f"gmm_prefetch_lhs_tid{prefetch_lhs_cache_tile_id}_s{lhs_slot}"):
            _start_lhs_fetch(prefetch_lhs_cache_tile_id, lhs_slot)

        @pl.when(prefetch_rhs_after_dot)
        def _prefetch_rhs_after_dot():
          with jax.named_scope(f"gmm_prefetch_rhs_ts{prefetch_rhs_te_start}_s{rhs_slot}"):
            _start_rhs_fetch(prefetch_rhs_te_start, rhs_slot)

        with jax.named_scope(f"gmm_mask_acc_s{acc_slot}"):
          @pl.when(store_full_tile)
          def _store_full_tile():
            with jax.named_scope(f"gmm_store_full_acc_s{acc_slot}"):
              acc_scratch[acc_slot, pl.dslice(sub_offset, tms), :] = (
                  partial.astype(preferred_element_type)
              )

          @pl.when(jnp.logical_not(store_full_tile))
          def _store_partial_tile():
            with jax.named_scope(f"gmm_store_mask_acc_i{active_i}"):
              store_iota = jax.lax.broadcasted_iota(jnp.int32, (tms, n), 0) + m_id
              store_mask = jnp.logical_and(store_iota >= group_start, store_iota < group_end)
            with jax.named_scope(f"gmm_store_masked_acc_s{acc_slot}"):
              acc_slice = acc_scratch[acc_slot, pl.dslice(sub_offset, tms), :]
              acc_scratch[acc_slot, pl.dslice(sub_offset, tms), :] = jnp.where(
                  store_mask,
                  partial.astype(preferred_element_type),
                  acc_slice,
              )

        @pl.when(last_for_lhs)
        def _start_acc_store_after_commit():
          with jax.named_scope(f"gmm_start_acc_store_tid{lhs_cache_tile_id}_s{acc_slot}"):
            _start_acc_store(acc_slot, lhs_cache_tile_id)

      return jnp.array(0, dtype=jnp.int32)

    final_carry = lax.fori_loop(
        0,
        metadata_len,
        _active_body,
        jnp.array(0, dtype=jnp.int32),
        unroll=True,
    )
    del final_carry

    with jax.named_scope("gmm_wait_out_store_0"):
      _wait_acc_store(jnp.array(0, dtype=jnp.int32))

    @pl.when(final_lhs_relative_cache_id >= 1)
    def _drain_out_store_slot1():
      with jax.named_scope("gmm_wait_out_store_1"):
        _wait_acc_store(jnp.array(1, dtype=jnp.int32))

  hbm_block_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

  lhs_bytes = _calculate_bytes(lhs)
  rhs_bytes = (num_current_groups * k * n) * rhs.itemsize
  out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize
  bytes_accessed = lhs_bytes + rhs_bytes + out_bytes
  flops = 2 * m * k * n
  cost_estimate = pl.CostEstimate(
      flops=flops, bytes_accessed=bytes_accessed, transcendentals=0
  )
  metadata = {
      "preferred_element_type": jnp.dtype(preferred_element_type).name,
      "tiling": {"tile_m_cache": tm, "tile_m_compute": tms, "tile_experts": te},
      "transpose_rhs": transpose_rhs,
      "variant": "gmm",
      "input_buffer_count": {
          "lhs": 2,
          "rhs": 2,
          "implementation": "metadata_driven_manual_async_copy",
      },
      "accumulator": {
          "buffer_count": 2,
          "shape": [tm, n],
          "dtype": jnp.dtype(preferred_element_type).name,
          "store": "whole_cache_tile",
      },
      "loop_order": ["active_tms_tile"],
  }
  rhs_scratch_shape = (2, te, n, k) if transpose_rhs else (2, te, k, n)

  call_gmm = pl.pallas_call(
      kernel,
      out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=7,
          in_specs=[
              hbm_block_spec,
              hbm_block_spec,
          ],
          out_specs=hbm_block_spec,
          scratch_shapes=[
              pltpu.VMEM((2, tm, k), lhs.dtype),
              pltpu.VMEM(rhs_scratch_shape, rhs.dtype),
              pltpu.VMEM((2, tm, n), preferred_element_type),
              pltpu.SemaphoreType.DMA((2,)),
              pltpu.SemaphoreType.DMA((2,)),
              pltpu.SemaphoreType.DMA((2,)),
          ],
      ),
      compiler_params=pltpu.CompilerParams(
          dimension_semantics=(),
          has_side_effects=True,
          vmem_limit_bytes=64 * 1024 * 1024,
      ),
      interpret=interpret,
      cost_estimate=cost_estimate,
          metadata={"xprof_metadata": json.dumps(metadata)},
  )

  args = [
      *group_metadata,
      group_offset,
      num_active_tiles,
      lhs,
      rhs,
  ]
  out = call_gmm(*args)
  if num_current_groups < num_total_groups:
    group_offsets, group_ids, m_tile_ids = group_metadata[:3]
    out = _zero_uninitialized_memory(
        out,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        group_metadata=(group_offsets, group_ids, m_tile_ids),
    )
  return out
