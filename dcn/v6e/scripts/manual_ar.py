"""Can a hand-written all-reduce beat the fused MegaScale one?

The lowering is inside libtpu and cannot be patched, but what XLA is asked to
emit is ours to choose. At DP=2 an all-reduce is exactly a bidirectional
exchange plus a local add:

    psum(x, "dcn") == x + ppermute(x, "dcn", perm=[(0,1),(1,0)])

The right-hand side goes down the ONE_TO_ONE transport path, which measures
237.5 Gbps clean, instead of the fused ALL_REDUCE host transfer, which measures
199.4. The local add is HBM-bound and should be a few ms against ~172 ms.

Loads benchmark.py through runpy so it stays byte-identical to the pinned SHA,
the same trick timing_benchmark.py uses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import statistics

SOURCE = os.environ.get(
    "DCN_SOURCE_BENCHMARK", str(Path(__file__).with_name("benchmark.py"))
)
src = runpy.run_path(SOURCE, run_name="dcn_manual_ar_source")

jax = src["jax"]
jnp = src["jnp"]
np = src["np"]
P = src["P"]
smap = src["smap"]
build_mesh = src["build_mesh"]
make_input = src["make_input"]
timed_batches = src["timed_batches"]
network_snapshot = src["network_snapshot"]
network_delta = src["network_delta"]
log = src["log"]


def make_psum(mesh):
    spec = P("dcn", None)

    @jax.jit
    def op(x):
        return smap(lambda v: jax.lax.psum(v, "dcn"), mesh, spec, spec)(x)

    return op


def make_exchange_add(mesh):
    """x + ppermute(x, [(0,1),(1,0)]) — identical result at DP=2."""
    spec = P("dcn", None)

    @jax.jit
    def op(x):
        def body(v):
            return v + jax.lax.ppermute(v, "dcn", perm=[(0, 1), (1, 0)])

        return smap(body, mesh, spec, spec)(x)

    return op


def make_chunked_psum(mesh, nchunks):
    """Split the shard and issue nchunks psums, to hand-roll the pipelining the
    fused path does not do."""
    spec = P("dcn", None)

    @jax.jit
    def op(x):
        def body(v):
            parts = jnp.split(v, nchunks, axis=0)
            return jnp.concatenate(
                [jax.lax.psum(p, "dcn") for p in parts], axis=0
            )

        return smap(body, mesh, spec, spec)(x)

    return op


def make_chunked_exchange_add(mesh, nchunks):
    spec = P("dcn", None)

    @jax.jit
    def op(x):
        def body(v):
            parts = jnp.split(v, nchunks, axis=0)
            out = [
                p + jax.lax.ppermute(p, "dcn", perm=[(0, 1), (1, 0)])
                for p in parts
            ]
            return jnp.concatenate(out, axis=0)

        return smap(body, mesh, spec, spec)(x)

    return op



# ---------------------------------------------------------------------------
# General n. The DP=2 identity does not extend: sending a full copy to each of
# the n-1 peers costs (n-1)*S per rank, while a ring costs 2(n-1)/n*S. They
# coincide only at n=2.
# ---------------------------------------------------------------------------

def make_direct_add(mesh, n):
    """Naive: receive from every peer and add. (n-1)*S per rank, so it should
    lose at n>2 on bytes alone. Included as the bytes-suboptimal reference."""
    spec = P("dcn", None)

    @jax.jit
    def op(x):
        def body(v):
            acc = v
            for k in range(1, n):
                acc = acc + jax.lax.ppermute(
                    v, "dcn", perm=[(i, (i + k) % n) for i in range(n)]
                )
            return acc

        return smap(body, mesh, spec, spec)(x)

    return op


def make_ring_ar(mesh, n):
    """Hand-written ring: reduce-scatter then all-gather, ppermute only.
    2(n-1)/n*S per rank, the same bytes the fused all-reduce nominally moves."""
    spec = P("dcn", None)
    fwd = [(i, (i + 1) % n) for i in range(n)]

    @jax.jit
    def op(x):
        def body(v):
            rows, cols = v.shape
            c = v.reshape(n, rows // n, cols)
            idx = jax.lax.axis_index("dcn")

            # reduce-scatter: after n-1 steps rank idx owns the complete sum of
            # chunk (idx+1) % n
            for k in range(n - 1):
                si = (idx - k) % n
                send = jax.lax.dynamic_index_in_dim(c, si, axis=0, keepdims=False)
                recv = jax.lax.ppermute(send, "dcn", perm=fwd)
                ri = (idx - k - 1) % n
                cur = jax.lax.dynamic_index_in_dim(c, ri, axis=0, keepdims=False)
                c = jax.lax.dynamic_update_index_in_dim(c, cur + recv, ri, axis=0)

            # all-gather: circulate the owned chunks back around
            for k in range(n - 1):
                si = (idx + 1 - k) % n
                send = jax.lax.dynamic_index_in_dim(c, si, axis=0, keepdims=False)
                recv = jax.lax.ppermute(send, "dcn", perm=fwd)
                ri = (idx - k) % n
                c = jax.lax.dynamic_update_index_in_dim(c, recv, ri, axis=0)

            return c.reshape(rows, cols)

        return smap(body, mesh, spec, spec)(x)

    return op


def make_rs_ag(mesh, n):
    """psum_scatter + all_gather. Two fused MegaScale ops, but neither is the
    ALL_REDUCE one."""
    spec = P("dcn", None)

    @jax.jit
    def op(x):
        def body(v):
            r = jax.lax.psum_scatter(v, "dcn", scatter_dimension=0, tiled=True)
            return jax.lax.all_gather(r, "dcn", axis=0, tiled=True)

        return smap(body, mesh, spec, spec)(x)

    return op


N_SLICES = int(os.environ.get("MA_SLICES", "2"))

VARIANTS = {
    "psum": lambda m: make_psum(m),
    "exchange_add": lambda m: make_exchange_add(m),
    "chunked_psum_4": lambda m: make_chunked_psum(m, 4),
    "chunked_psum_8": lambda m: make_chunked_psum(m, 8),
    "chunked_exchange_add_4": lambda m: make_chunked_exchange_add(m, 4),
    "direct_add": lambda m: make_direct_add(m, N_SLICES),
    "ring_ar": lambda m: make_ring_ar(m, N_SLICES),
    "rs_ag": lambda m: make_rs_ag(m, N_SLICES),
}


def verify(mesh, participants):
    """Every variant must produce the same tensor as psum.

    Reduce to a replicated scalar before pulling it to the host: in a
    multi-controller job a rank cannot device_get another rank's shards.
    """
    dim = 512
    rep = jax.sharding.NamedSharding(mesh, P())
    host = np.zeros((dim, dim), dtype=np.float32)
    rows = dim // len(mesh.devices)
    for i in range(len(mesh.devices)):
        host[i * rows:(i + 1) * rows, :] = i + 1
    x = jax.device_put(host, jax.sharding.NamedSharding(mesh, P("dcn", None)))

    def checksum(fn):
        return float(np.asarray(
            jax.jit(lambda v: fn(v).astype(jnp.float32).mean(),
                    out_shardings=rep)(x)
        ))

    ref = checksum(make_psum(mesh))
    out = {"psum_checksum": ref}
    for name in os.environ.get("MA_VARIANTS", ",".join(VARIANTS)).split(","):
        factory = VARIANTS[name]
        got = checksum(factory(mesh))
        ok = abs(got - ref) < 1e-3
        out[name] = ok
        if not ok:
            raise AssertionError(f"{name}: checksum {got} != psum {ref}")
    return out


def main() -> None:
    dims = [int(v) for v in os.environ.get("MA_DIMS", "32768").split(",")]
    names = os.environ.get("MA_VARIANTS", ",".join(VARIANTS)).split(",")
    warmups = int(os.environ.get("MA_WARMUPS", "200"))
    reps = int(os.environ.get("MA_REPS", "10"))
    batch = int(os.environ.get("MA_BATCH", "10"))
    n_slices = N_SLICES

    groups = src["devices_by_slice"]()
    participants = min(len(g) for g in groups)
    mesh, _ = build_mesh(n_slices, participants)

    checks = verify(mesh, participants)
    log("MANUAL_AR_VERIFY " + json.dumps(checks, sort_keys=True))

    itemsize = jnp.dtype(jnp.bfloat16).itemsize
    rows = []
    for dim in dims:
        x = make_input(mesh, dim)
        shard_bytes = dim * dim * itemsize / n_slices
        for name in names:
            before = network_snapshot()
            samples = timed_batches(
                VARIANTS[name](mesh), x,
                label=f"manual_{name}_d{dim}",
                batch=batch, reps=reps, warmups=warmups,
            )
            after = network_snapshot()
            best = min(samples)
            med = statistics.median(samples)
            row = {
                "benchmark": "manual_ar",
                "variant": name,
                "matrix_dim": dim,
                "shard_bytes_per_device": shard_bytes,
                "participants_per_slice": participants,
                "warmups": warmups, "reps": reps, "batch": batch,
                "time_ms_best": best, "time_ms_median": med,
                "time_ms_all": samples,
                # same bus-bandwidth convention as benchmark.py: at DP=2 the
                # ring factor 2(n-1)/n is exactly 1
                "host_gbps_best": shard_bytes / 1e9 / (best / 1e3) * participants * 8,
                "host_gbps_median": shard_bytes / 1e9 / (med / 1e3) * participants * 8,
                "network_delta": network_delta(before, after),
            }
            rows.append(row)
            log("MANUAL_AR_RESULT " + json.dumps(row, sort_keys=True))
        del x

    log("MANUAL_AR_SUMMARY " + json.dumps(
        {f"{r['variant']}@{r['matrix_dim']}": round(r["host_gbps_best"], 1) for r in rows},
        sort_keys=True))


if __name__ == "__main__":
    main()
