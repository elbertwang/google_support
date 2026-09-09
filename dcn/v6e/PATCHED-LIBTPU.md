# A patched libtpu removing the `send_done → recv_done` dependency: no effect

Artifact under test: a libtpu built from an internal snapshot with one line
removed in the MegaScale cross-slice rewrite pass —

```cpp
if (!fusion_data.has_value()) {
  CHECK(send_d);
- RETURN_IF_ERROR(send_d->AddControlDependencyTo(recv_d));
+ // bypassing this control dependency should unlock simultaneous send/recv
}
```

The stated theory was that this dependency serialises send against receive and
degrades the duplex link to near half-duplex.

**It does not.** The patch is confirmed active at the HLO level and moves
performance by +1.6%, inside the noise. The real cause was host-side reduction —
see [`HOST-REDUCTION.md`](HOST-REDUCTION.md).

Recorded here because the negative result is worth as much as the positive one,
and because the verification method generalises.

## The patch is loaded

`importlib.metadata` still reports the pip package version (`libtpu 0.0.44`) and
tells you nothing about which `.so` is mapped. Check the process:

```
$ kubectl exec $POD -- sh -c 'for p in /proc/[0-9]*/maps; do
    grep -ho "/[^ ]*libtpu[^ ]*\.so" $p; done | sort -u'
/tmp/libtpu_patched.so
```

## The patch is active, and it is selective

From `after_optimizations_before_buffer_assignment` (the dependency is absorbed
by scheduling before `after_optimizations`, so it is invisible there):

| module | transfer type | `recv-done` control-predecessors |
|---|---|---|
| ALL_GATHER  | fused | `{%send}` |
| ALL_REDUCE ×4 | fused | `{%send}` |
| ONE_TO_ONE ×2 | `ppermute` | `{%send-done, %send}` |

The patch lands only on the fused path. `ONE_TO_ONE` keeps the dependency.

That is convenient: within one binary and one process, `psum` is the treatment
and `exchange_add` is a same-snapshot, unpatched control. It supplies the
comparison we otherwise could not build, since we have no unpatched build of the
same source snapshot.

## No performance effect

3 interleaved rounds, tpu7x, DP=2, dim 32768, warmups 200:

```
psum         (ALL_REDUCE, patched path)
   stock    161.1  159.1  166.1   mean=162.1  sd=3.6
   patched  164.8  167.8  161.5   mean=164.7  sd=3.2    +1.6%
exchange_add (ONE_TO_ONE, control)
   stock    276.5  288.6  298.2   mean=287.8  sd=10.9
   patched  305.1  281.5  281.9   mean=289.5  sd=13.5    +0.6%
```

Welch t = 0.94, p ≈ 0.4, 95% CI on the difference [-3.1%, +6.4%]. Any real effect
is under ~6%; the gap this was meant to explain is 43%. The control moving +0.6%
also rules out a generic "newer snapshot is faster" effect.

A single first run gave `psum` 171.2, which looks like +6.8%. It was the top of
the noise band and did not survive repetition. At σ ≈ 3.5 one sample cannot
resolve this.

## Independent evidence against the theory

In the stock dumps, the fast and slow variants carry the **same** dependency:

```
ONE_TO_ONE   control-predecessors={%send-done, %send}   ppermute_uni, 313-321 Gbps
ONE_TO_ONE   control-predecessors={%send-done, %send}   ppermute_bidi,     237.5
ALL_GATHER   control-predecessors={%send-done, %send}                      221.0
ALL_REDUCE   control-predecessors={%send-done, %send}                      199.4
```

Something present in every variant cannot be what separates them.

## Reproduce

```bash
scripts/patched-ab.sh    # interleaved A/B, 3 rounds each arm
```
